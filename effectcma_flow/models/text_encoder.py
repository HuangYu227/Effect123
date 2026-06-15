from __future__ import annotations

import hashlib
import re
from typing import Any
import warnings

import torch
from torch import nn


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HashTextEncoder(nn.Module):
    """Deterministic offline text encoder with a trainable projection."""

    def __init__(self, raw_dim: int, d_model: int) -> None:
        super().__init__()
        self.raw_dim = int(raw_dim)
        self.proj = nn.Linear(self.raw_dim, d_model)

    def forward(self, text_condition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        slots = _expect_slots(text_condition)
        raw, mask = _hash_slots(slots, self.raw_dim, self.proj.weight.device, self.proj.weight.dtype)
        return self.proj(raw), mask


class PrecomputedTextEncoder(nn.Module):
    """Projection wrapper for precomputed text embeddings."""

    def __init__(self, input_dim: int, d_model: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.proj = nn.Linear(self.input_dim, d_model)

    def forward(self, text_condition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(text_condition, dict):
            embeddings = text_condition["embeddings"]
            mask = text_condition.get("mask")
        else:
            embeddings = text_condition
            mask = None
        if embeddings.ndim == 2:
            embeddings = embeddings[:, None, :]
        if embeddings.ndim != 3:
            raise ValueError(f"precomputed embeddings must be [B, J, D] or [B, D], got {tuple(embeddings.shape)}")
        embeddings = embeddings.to(self.proj.weight.device, dtype=self.proj.weight.dtype)
        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(f"Expected embedding dim {self.input_dim}, got {embeddings.shape[-1]}")
        if mask is None:
            mask = torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=embeddings.dtype)
        else:
            mask = mask.to(embeddings.device, dtype=embeddings.dtype)
            if mask.shape != embeddings.shape[:2]:
                raise ValueError(f"precomputed mask shape must be {tuple(embeddings.shape[:2])}, got {tuple(mask.shape)}")
        return self.proj(embeddings), mask


class HFTextEncoder(nn.Module):
    """Frozen HuggingFace encoder with mean pooling and trainable projection."""

    def __init__(self, model_name: str, d_model: int, *, local_files_only: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        hidden_size = int(self.backbone.config.hidden_size)
        self.proj = nn.Linear(hidden_size, d_model)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, text_condition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        slots = _expect_slots(text_condition)
        device = self.proj.weight.device
        self.backbone.eval()
        flat, offsets, max_slots = _flatten_slots(slots)
        if not flat:
            batch_size = len(slots)
            dtype = self.proj.weight.dtype
            z = torch.zeros(batch_size, 1, self.proj.out_features, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            return z, mask
        tokens = self.tokenizer(flat, padding=True, truncation=True, return_tensors="pt")
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            out = self.backbone(**tokens)
            raw = _mean_pool(out.last_hidden_state, tokens["attention_mask"])
        emb = self.proj(raw)
        z = torch.zeros(len(slots), max_slots, emb.shape[-1], device=device, dtype=emb.dtype)
        mask = torch.zeros(len(slots), max_slots, device=device, dtype=emb.dtype)
        for row, (start, end) in enumerate(offsets):
            if end > start:
                width = end - start
                z[row, :width] = emb[start:end]
                mask[row, :width] = 1.0
        return z, mask


class CLIPTextProjectionEncoder(nn.Module):
    """Frozen CLIP/LongCLIP text encoder.

    ``output_type='pooled'`` preserves the old behavior: one projected text
    embedding per caption slot. ``output_type='tokens'`` returns token-level
    hidden states packed per sample, which gives downstream cross-modal
    modules real semantic tokens to attend over.
    """

    def __init__(
        self,
        model_name: str,
        d_model: int,
        *,
        local_files_only: bool = False,
        output_type: str = "pooled",
        max_output_tokens: int = 128,
    ) -> None:
        super().__init__()
        from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

        self.output_type = str(output_type).lower()
        if self.output_type not in {"pooled", "tokens"}:
            raise ValueError(f"CLIPTextProjectionEncoder output_type must be 'pooled' or 'tokens', got {output_type!r}")
        self.max_output_tokens = int(max_output_tokens)
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        model_key = str(model_name)
        if "longclip" in model_key.lower():
            clip_config = CLIPTextConfig.from_pretrained(model_name, local_files_only=local_files_only)
            clip_config.max_position_embeddings = max(int(getattr(clip_config, "max_position_embeddings", 77)), 248)
            self.backbone = CLIPTextModelWithProjection.from_pretrained(
                model_name,
                config=clip_config,
                local_files_only=local_files_only,
            )
        else:
            self.backbone = CLIPTextModelWithProjection.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        if self.output_type == "tokens":
            hidden_size = int(self.backbone.config.hidden_size)
        else:
            hidden_size = int(getattr(self.backbone.config, "projection_dim", self.backbone.config.hidden_size))
        self.max_length = int(getattr(self.backbone.config, "max_position_embeddings", 77))
        self.proj = nn.Linear(hidden_size, d_model)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, text_condition: Any) -> tuple[torch.Tensor, torch.Tensor]:
        slots = _expect_slots(text_condition)
        device = self.proj.weight.device
        self.backbone.eval()
        flat, offsets, max_slots = _flatten_slots(slots)
        if not flat:
            batch_size = len(slots)
            dtype = self.proj.weight.dtype
            z = torch.zeros(batch_size, 1, self.proj.out_features, device=device, dtype=dtype)
            mask = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            return z, mask

        input_ids, attention_mask, owners = self._tokenize_chunks(flat, device)
        with torch.no_grad():
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.output_type == "tokens":
            return self._pack_token_outputs(
                last_hidden=out.last_hidden_state,
                attention_mask=attention_mask,
                owners=owners,
                offsets=offsets,
                batch_size=len(slots),
            )

        chunk_emb = out.text_embeds
        raw = torch.zeros(len(flat), chunk_emb.shape[-1], device=device, dtype=chunk_emb.dtype)
        counts = torch.zeros(len(flat), 1, device=device, dtype=chunk_emb.dtype)
        raw.index_add_(0, owners, chunk_emb)
        counts.index_add_(0, owners, torch.ones(owners.shape[0], 1, device=device, dtype=chunk_emb.dtype))
        raw = raw / counts.clamp_min(1.0)
        emb = self.proj(raw)
        z = torch.zeros(len(slots), max_slots, emb.shape[-1], device=device, dtype=emb.dtype)
        mask = torch.zeros(len(slots), max_slots, device=device, dtype=emb.dtype)
        for row, (start, end) in enumerate(offsets):
            if end > start:
                width = end - start
                z[row, :width] = emb[start:end]
                mask[row, :width] = 1.0
        return z, mask

    def _pack_token_outputs(
        self,
        *,
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        owners: torch.Tensor,
        offsets: list[tuple[int, int]],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.proj(last_hidden)
        per_text: list[torch.Tensor] = []
        empty = projected.new_zeros((0, projected.shape[-1]))
        flat_count = max((end for _, end in offsets), default=0)
        for flat_idx in range(flat_count):
            chunks = torch.nonzero(owners == flat_idx, as_tuple=False).flatten()
            pieces = []
            for chunk_idx in chunks.tolist():
                valid = attention_mask[chunk_idx].to(dtype=torch.bool)
                if valid.any():
                    pieces.append(projected[chunk_idx, valid])
            per_text.append(torch.cat(pieces, dim=0) if pieces else empty)

        rows: list[torch.Tensor] = []
        max_len = 1
        for start, end in offsets:
            pieces = [per_text[idx] for idx in range(start, end) if idx < len(per_text) and per_text[idx].numel() > 0]
            row = torch.cat(pieces, dim=0) if pieces else empty
            if row.shape[0] > self.max_output_tokens:
                row = row[: self.max_output_tokens]
            rows.append(row)
            max_len = max(max_len, int(row.shape[0]))
        max_len = min(max_len, self.max_output_tokens)
        z = projected.new_zeros((batch_size, max_len, projected.shape[-1]))
        mask = projected.new_zeros((batch_size, max_len))
        for row_idx, row in enumerate(rows):
            width = min(int(row.shape[0]), max_len)
            if width > 0:
                z[row_idx, :width] = row[:width]
                mask[row_idx, :width] = 1.0
        return z, mask

    def _tokenize_chunks(self, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        special_len = len(self.tokenizer.build_inputs_with_special_tokens([]))
        chunk_payload = max(1, self.max_length - special_len)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0
        rows: list[list[int]] = []
        masks: list[list[int]] = []
        owners: list[int] = []
        for owner, text in enumerate(texts):
            token_ids = self.tokenizer.encode(str(text), add_special_tokens=False)
            if not token_ids:
                token_chunks = [[]]
            else:
                token_chunks = [token_ids[start : start + chunk_payload] for start in range(0, len(token_ids), chunk_payload)]
            for chunk in token_chunks:
                full = self.tokenizer.build_inputs_with_special_tokens(chunk)
                if len(full) > self.max_length:
                    full = full[: self.max_length]
                attn = [1] * len(full)
                pad = self.max_length - len(full)
                if pad > 0:
                    full = full + [int(pad_id)] * pad
                    attn = attn + [0] * pad
                rows.append([int(x) for x in full])
                masks.append(attn)
                owners.append(owner)
        input_ids = torch.tensor(rows, device=device, dtype=torch.long)
        attention_mask = torch.tensor(masks, device=device, dtype=torch.long)
        owner_tensor = torch.tensor(owners, device=device, dtype=torch.long)
        return input_ids, attention_mask, owner_tensor


def build_text_encoder(config: dict[str, Any], d_model: int) -> nn.Module:
    mode = str(config.get("mode", "hash")).lower()
    if mode == "hash":
        return HashTextEncoder(raw_dim=int(config.get("hash_dim", 256)), d_model=d_model)
    if mode == "precomputed":
        return PrecomputedTextEncoder(input_dim=int(config.get("precomputed_dim", 128)), d_model=d_model)
    if mode == "hf":
        model_name = str(config.get("hf_model_name", "BAAI/bge-small-en-v1.5"))
        local_files_only = bool(config.get("local_files_only", False))
        try:
            return HFTextEncoder(
                model_name=model_name,
                d_model=d_model,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            fallback_mode = str(config.get("fallback_mode", "error")).lower()
            if fallback_mode == "hash":
                warnings.warn(
                    f"Failed to load HuggingFace text encoder {model_name!r}; falling back to HashTextEncoder. "
                    f"Set local_files_only={local_files_only!r} and verify the model cache for server training. "
                    f"Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return HashTextEncoder(raw_dim=int(config.get("fallback_hash_dim", config.get("hash_dim", 256))), d_model=d_model)
            raise RuntimeError(
                f"Failed to load HuggingFace text encoder {model_name!r}. "
                f"local_files_only={local_files_only!r}. Install/cache the model or set text_encoder.mode=hash "
                f"for offline smoke tests. Original error: {exc}"
            ) from exc
    if mode in {"longclip", "clip_hf"}:
        model_name = str(config.get("longclip_model_name", "save/Longclip"))
        local_files_only = bool(config.get("longclip_local_files_only", config.get("local_files_only", False)))
        try:
            return CLIPTextProjectionEncoder(
                model_name=model_name,
                d_model=d_model,
                local_files_only=local_files_only,
                output_type=str(config.get("output_type", "pooled")),
                max_output_tokens=int(config.get("max_output_tokens", 128)),
            )
        except Exception as exc:
            fallback_mode = str(config.get("fallback_mode", "error")).lower()
            if fallback_mode == "hash":
                warnings.warn(
                    f"Failed to load CLIP/LongCLIP text encoder {model_name!r}; falling back to HashTextEncoder. "
                    f"Verify longclip_model_name and local_files_only={local_files_only!r}. Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return HashTextEncoder(raw_dim=int(config.get("fallback_hash_dim", config.get("hash_dim", 256))), d_model=d_model)
            raise RuntimeError(
                f"Failed to load CLIP/LongCLIP text encoder {model_name!r}. "
                f"Verify longclip_model_name and local_files_only={local_files_only!r}. Original error: {exc}"
            ) from exc
    raise ValueError(f"Unknown text encoder mode {mode!r}")


def _expect_slots(text_condition: Any) -> list[list[str]]:
    if isinstance(text_condition, dict):
        if "slots" not in text_condition:
            raise ValueError("Text condition dict must contain 'slots'")
        text_condition = text_condition["slots"]
    if not isinstance(text_condition, list):
        raise ValueError("Slot text condition must be a list[list[str]]")
    return [[str(x) for x in one] for one in text_condition]


def _hash_slots(slots: list[list[str]], dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, max_slots = _flatten_slots(slots)
    max_slots = max(max_slots, 1)
    raw = torch.zeros(len(slots), max_slots, dim, device=device, dtype=dtype)
    mask = torch.zeros(len(slots), max_slots, device=device, dtype=dtype)
    for i, one in enumerate(slots):
        for j, text in enumerate(one):
            raw[i, j] = _hash_text(text, dim, device, dtype)
            mask[i, j] = 1.0
    return raw, mask


def _hash_text(text: str, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vec = torch.zeros(dim, device=device, dtype=dtype)
    tokens = TOKEN_RE.findall(text.lower()) or [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = torch.linalg.vector_norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _flatten_slots(slots: list[list[str]]) -> tuple[list[str], list[tuple[int, int]], int]:
    flat: list[str] = []
    offsets: list[tuple[int, int]] = []
    max_slots = 0
    for one in slots:
        start = len(flat)
        flat.extend(one)
        end = len(flat)
        offsets.append((start, end))
        max_slots = max(max_slots, end - start)
    return flat, offsets, max_slots


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask[:, :, None].to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom
