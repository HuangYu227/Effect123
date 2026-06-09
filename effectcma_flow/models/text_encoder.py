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
    """Frozen CLIP/LongCLIP text encoder using projected text embeddings."""

    def __init__(self, model_name: str, d_model: int, *, local_files_only: bool = False) -> None:
        super().__init__()
        from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

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

        tokens = self.tokenizer(flat, padding=True, return_tensors="pt")
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        chunks = []
        for start in range(0, input_ids.shape[1], self.max_length):
            chunk_ids = input_ids[:, start : start + self.max_length]
            if chunk_ids.shape[1] == 0:
                continue
            chunk_mask = attention_mask[:, start : start + self.max_length] if attention_mask is not None else None
            kwargs = {"input_ids": chunk_ids}
            if chunk_mask is not None:
                kwargs["attention_mask"] = chunk_mask
            with torch.no_grad():
                chunks.append(self.backbone(**kwargs).text_embeds)
        raw = torch.stack(chunks, dim=0).mean(dim=0)
        emb = self.proj(raw)
        z = torch.zeros(len(slots), max_slots, emb.shape[-1], device=device, dtype=emb.dtype)
        mask = torch.zeros(len(slots), max_slots, device=device, dtype=emb.dtype)
        for row, (start, end) in enumerate(offsets):
            if end > start:
                width = end - start
                z[row, :width] = emb[start:end]
                mask[row, :width] = 1.0
        return z, mask


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
