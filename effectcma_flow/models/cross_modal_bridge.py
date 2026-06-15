from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CrossModalConditionBridge(nn.Module):
    """Bidirectional text-state bridge for text-to-series generation.

    Text tokens interact with state tokens built from the current flow state.
    The bridge keeps token-level alignment inside the velocity field instead of
    reducing text to a single vector before the generator sees the state.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_channels: int,
        sequence_length: int,
        num_experts: int,
        patch_size: int = 6,
        num_heads: int = 4,
        dropout: float = 0.0,
        num_spectral_tokens: int = 3,
        alignment_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_channels <= 0 or sequence_length <= 0 or num_experts <= 0:
            raise ValueError("d_model, num_channels, sequence_length, and num_experts must be positive")
        if patch_size <= 0 or num_spectral_tokens <= 0:
            raise ValueError("patch_size and num_spectral_tokens must be positive")
        heads = max(1, min(int(num_heads), int(d_model)))
        while d_model % heads != 0 and heads > 1:
            heads -= 1

        self.d_model = int(d_model)
        self.num_channels = int(num_channels)
        self.sequence_length = int(sequence_length)
        self.num_experts = int(num_experts)
        self.patch_size = int(patch_size)
        self.num_spectral_tokens = int(num_spectral_tokens)
        self.alignment_temperature = float(alignment_temperature)

        self.time_patch_proj = nn.Linear(self.patch_size * self.num_channels, self.d_model)
        self.channel_stat_proj = nn.Linear(3, self.d_model)
        self.spectral_proj = nn.Linear(1, self.d_model)
        self.channel_embed = nn.Embedding(self.num_channels, self.d_model)
        self.spectral_embed = nn.Embedding(self.num_spectral_tokens, self.d_model)
        # 0=time patch, 1=channel summary, 2=spectral band.
        self.type_embed = nn.Embedding(3, self.d_model)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        self.text_norm = nn.LayerNorm(self.d_model)
        self.state_norm = nn.LayerNorm(self.d_model)
        self.text_to_state = nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True)
        self.state_to_text = nn.MultiheadAttention(self.d_model, heads, dropout=dropout, batch_first=True)
        self.text_ffn = _ffn(self.d_model, dropout)
        self.state_ffn = _ffn(self.d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        self.state_to_context = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        self.context_gate = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.d_model),
            nn.Sigmoid(),
        )
        self.context_norm = nn.LayerNorm(self.d_model)
        self.expert_context = nn.Sequential(
            nn.LayerNorm(2 * self.d_model),
            nn.Linear(2 * self.d_model, self.num_experts * self.d_model),
        )
        self.text_align = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
        self.state_align = nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))

    def forward(
        self,
        *,
        slot_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if slot_tokens.ndim != 3:
            raise ValueError(f"slot_tokens must be [B,J,D], got {tuple(slot_tokens.shape)}")
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,L,C], got {tuple(x_t.shape)}")
        batch, length, channels = x_t.shape
        if slot_tokens.shape[0] != batch:
            raise ValueError("slot_tokens and x_t must have the same batch size")
        if slot_tokens.shape[-1] != self.d_model:
            raise ValueError(f"Expected slot token dim {self.d_model}, got {slot_tokens.shape[-1]}")
        if (length, channels) != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected x_t shape [B,{self.sequence_length},{self.num_channels}], got {tuple(x_t.shape)}")
        if t.shape != (batch,):
            raise ValueError(f"t must have shape [B], got {tuple(t.shape)}")
        if slot_mask.shape != slot_tokens.shape[:2]:
            raise ValueError(f"slot_mask must have shape {tuple(slot_tokens.shape[:2])}, got {tuple(slot_mask.shape)}")

        dtype = slot_tokens.dtype
        device = slot_tokens.device
        x_t = x_t.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype)
        slot_mask = slot_mask.to(device=device, dtype=dtype)
        state_tokens = self._build_state_tokens(x_t, t)

        text_query = self.text_norm(slot_tokens)
        state_query = self.state_norm(state_tokens)
        text_delta, text_attn = self.text_to_state(
            query=text_query,
            key=state_query,
            value=state_query,
            need_weights=True,
            average_attn_weights=False,
        )
        bridged_text = slot_tokens + self.dropout(text_delta)
        bridged_text = bridged_text + self.text_ffn(bridged_text)
        bridged_text = bridged_text * slot_mask[:, :, None].clamp(0.0, 1.0)

        text_key_padding_mask = _safe_key_padding_mask(slot_mask)
        state_delta, state_attn = self.state_to_text(
            query=state_query,
            key=self.text_norm(slot_tokens),
            value=slot_tokens,
            key_padding_mask=text_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        bridged_state = state_tokens + self.dropout(state_delta)
        bridged_state = bridged_state + self.state_ffn(bridged_state)

        text_context = _masked_mean(bridged_text, slot_mask)
        time_token_count = _num_patch_tokens(length, self.patch_size)
        channel_context = bridged_state[:, time_token_count : time_token_count + channels]
        if channel_context.shape != (batch, channels, self.d_model):
            raise RuntimeError(
                f"internal channel_context shape {tuple(channel_context.shape)} "
                f"must be {(batch, channels, self.d_model)}"
            )

        state_context = self.state_to_context(bridged_state.mean(dim=1))
        gate = self.context_gate(torch.cat([text_context, state_context], dim=-1))
        bridge_context = self.context_norm(text_context + gate * state_context)
        expert_context = self.expert_context(torch.cat([bridge_context, state_context], dim=-1))
        expert_context = expert_context.view(batch, self.num_experts, self.d_model)

        alignment_loss, alignment_aux = self._alignment_loss(text_context, state_context)
        valid_slot_count = slot_mask.sum(dim=1)
        text_entropy, text_max, text_entropy_norm = _attention_summary(text_attn, query_mask=slot_mask)
        state_entropy, state_max, state_entropy_norm = _attention_summary(state_attn, key_mask=slot_mask)
        aux = {
            "bridge_alignment_loss": alignment_loss,
            "text_slot_count": valid_slot_count.detach().mean(),
            "text_slot_count_min": valid_slot_count.detach().min(),
            "text_slot_count_max": valid_slot_count.detach().max(),
            "bridge_text_to_state_entropy": text_entropy.detach(),
            "bridge_text_to_state_entropy_norm": text_entropy_norm.detach(),
            "bridge_text_to_state_max_prob": text_max.detach(),
            "bridge_state_to_text_entropy": state_entropy.detach(),
            "bridge_state_to_text_entropy_norm": state_entropy_norm.detach(),
            "bridge_state_to_text_max_prob": state_max.detach(),
            "bridge_context_norm": bridge_context.detach().norm(dim=-1).mean(),
            "bridge_text_context_norm": text_context.detach().norm(dim=-1).mean(),
            "bridge_state_context_norm": state_context.detach().norm(dim=-1).mean(),
            "bridge_expert_context_norm": expert_context.detach().norm(dim=-1).mean(),
            "bridge_channel_context_norm": channel_context.detach().norm(dim=-1).mean(),
            **alignment_aux,
        }
        return bridged_text, bridge_context, expert_context, channel_context, aux

    def _build_state_tokens(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x_t.shape
        dtype = x_t.dtype
        time_code = self.time_embed(t[:, None])[:, None, :]

        patch_size = min(self.patch_size, length)
        pad = (patch_size - length % patch_size) % patch_size
        if pad:
            x_patch = F.pad(x_t, (0, 0, 0, pad))
        else:
            x_patch = x_t
        num_patches = x_patch.shape[1] // patch_size
        patches = x_patch.reshape(batch, num_patches, patch_size * channels)
        if patch_size != self.patch_size:
            patches = F.pad(patches, (0, (self.patch_size - patch_size) * channels))
        time_tokens = self.time_patch_proj(patches)
        time_tokens = time_tokens + self.type_embed.weight[0].to(dtype=dtype)[None, None, :] + time_code

        mean = x_t.mean(dim=1)
        std = x_t.std(dim=1, unbiased=False)
        if length > 1:
            roughness = (x_t[:, 1:] - x_t[:, :-1]).square().mean(dim=1).sqrt()
        else:
            roughness = torch.zeros_like(std)
        channel_stats = torch.stack([mean, std, roughness], dim=-1)
        channel_tokens = self.channel_stat_proj(channel_stats)
        channel_ids = torch.arange(channels, device=x_t.device)
        channel_tokens = channel_tokens + self.channel_embed(channel_ids)[None, :, :].to(dtype=dtype)
        channel_tokens = channel_tokens + self.type_embed.weight[1].to(dtype=dtype)[None, None, :] + time_code

        spectral_stats = self._spectral_stats(x_t)
        spectral_tokens = self.spectral_proj(spectral_stats[:, :, None])
        spectral_ids = torch.arange(self.num_spectral_tokens, device=x_t.device)
        spectral_tokens = spectral_tokens + self.spectral_embed(spectral_ids)[None, :, :].to(dtype=dtype)
        spectral_tokens = spectral_tokens + self.type_embed.weight[2].to(dtype=dtype)[None, None, :] + time_code

        return self.state_norm(torch.cat([time_tokens, channel_tokens, spectral_tokens], dim=1))

    def _spectral_stats(self, x_t: torch.Tensor) -> torch.Tensor:
        centered = x_t - x_t.mean(dim=1, keepdim=True)
        fft_source = centered.transpose(1, 2)
        fft_dtype = torch.float32 if fft_source.dtype in {torch.float16, torch.bfloat16} else fft_source.dtype
        freq = torch.fft.rfft(fft_source.to(fft_dtype), dim=-1)
        power = freq.abs().square().mean(dim=1)
        freq_len = power.shape[-1]
        bands = []
        for idx in range(self.num_spectral_tokens):
            start = int(math.floor(idx * freq_len / self.num_spectral_tokens))
            end = int(math.floor((idx + 1) * freq_len / self.num_spectral_tokens))
            end = max(start + 1, min(end, freq_len))
            bands.append(power[:, start:end].mean(dim=-1))
        stats = torch.stack(bands, dim=-1)
        stats = torch.log1p(stats)
        return stats.to(dtype=x_t.dtype)

    def _alignment_loss(self, text_context: torch.Tensor, state_context: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        text = F.normalize(self.text_align(text_context), dim=-1)
        state = F.normalize(self.state_align(state_context), dim=-1)
        logits = text @ state.transpose(0, 1)
        logits = logits / max(self.alignment_temperature, 1e-4)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))
        pos = logits.diag()
        return loss, {
            "bridge_alignment_logit_pos": pos.detach().mean(),
            "bridge_alignment_logit_std": logits.detach().std(unbiased=False),
        }


def _ffn(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(dropout),
    )


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=tokens.device, dtype=tokens.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask[:, :, None]).sum(dim=1) / denom


def _safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    valid = mask > 0
    if valid.shape[1] == 0:
        raise ValueError("slot_mask must contain at least one slot")
    safe_valid = valid.clone()
    empty = ~safe_valid.any(dim=1)
    if empty.any():
        safe_valid[empty, 0] = True
    return ~safe_valid


def _num_patch_tokens(length: int, patch_size: int) -> int:
    effective = min(int(patch_size), int(length))
    return int(math.ceil(float(length) / float(max(effective, 1))))


def _attention_summary(
    attn: torch.Tensor,
    *,
    query_mask: torch.Tensor | None = None,
    key_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # attn: [B, heads, query_tokens, key_tokens]
    prob = attn.detach()
    batch, heads, queries, keys = prob.shape
    dtype = prob.dtype
    device = prob.device
    if key_mask is not None:
        key_mask = key_mask.to(device=device, dtype=dtype)
        if key_mask.shape != (batch, keys):
            raise ValueError(f"key_mask must have shape {(batch, keys)}, got {tuple(key_mask.shape)}")
        key_weight = key_mask[:, None, None, :]
        prob_for_entropy = prob * key_weight
        valid_key_count = key_mask.sum(dim=-1).clamp_min(2.0)
        norm_denom = valid_key_count.log()[:, None, None]
    else:
        key_weight = None
        prob_for_entropy = prob
        norm_denom = torch.full((batch, 1, 1), math.log(float(max(keys, 2))), device=device, dtype=dtype)

    p = prob_for_entropy.clamp_min(eps)
    entropy_per_query = -(p * p.log()).sum(dim=-1)
    if key_weight is not None:
        entropy_per_query = entropy_per_query * (key_mask.sum(dim=-1) > 0).to(dtype)[:, None, None]
        max_per_query = (prob * key_weight).max(dim=-1).values
    else:
        max_per_query = prob.max(dim=-1).values
    entropy_norm_per_query = entropy_per_query / norm_denom.clamp_min(eps)

    if query_mask is not None:
        query_mask = query_mask.to(device=device, dtype=dtype)
        if query_mask.shape != (batch, queries):
            raise ValueError(f"query_mask must have shape {(batch, queries)}, got {tuple(query_mask.shape)}")
        query_weight = query_mask[:, None, :]
        denom = (query_weight.sum() * float(heads)).clamp_min(1.0)
        entropy = (entropy_per_query * query_weight).sum() / denom
        max_prob = (max_per_query * query_weight).sum() / denom
        entropy_norm = (entropy_norm_per_query * query_weight).sum() / denom
    else:
        entropy = entropy_per_query.mean()
        max_prob = max_per_query.mean()
        entropy_norm = entropy_norm_per_query.mean()
    return entropy, max_prob, entropy_norm
