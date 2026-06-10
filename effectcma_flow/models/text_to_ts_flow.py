from __future__ import annotations

import torch
from torch import nn

from effectcma_flow.models.effect_mapper import EffectMapper
from effectcma_flow.models.operator_bank import ResidualOperatorBank
from effectcma_flow.models.ts_encoder import ChannelEncoder, TimePatchEncoder


class TextToTSFlow(nn.Module):
    """Text-conditioned continuous flow from Gaussian noise to time series.

    Unlike `EffectCMAFlow`, this model never receives a real base trajectory as
    condition. The current flow state `x_t` supplies state tokens, while caption
    tokens define a time-channel-operator generation field.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        num_channels: int,
        patch_len: int,
        d_model: int,
        num_operators: int,
        text_encoder: nn.Module,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        operator_hidden: int = 128,
        operator_t_dim: int = 32,
        operator_max_velocity: float = 5.0,
        operator_depth: int = 2,
        operator_kernel_size: int = 3,
        operator_dropout: float = 0.0,
        channel_patch_len: int | None = None,
        channel_stride: int | None = None,
        channel_temporal_layers: int = 1,
        channel_layers: int = 1,
        channel_heads: int | None = None,
        channel_dropout: float = 0.0,
        mapper_field_rank: int = 4,
        mapper_slot_layers: int = 1,
        mapper_slot_heads: int | None = None,
        mapper_dropout: float = 0.0,
        mapper_normalizer: str = "softmax",
        mapper_bounded_field_gate: bool = True,
        mapper_operator_router: str = "text",
        mapper_time_segment_scales: tuple[int, ...] | list[int] | None = None,
        mapper_router_epsilon: float = 1e-4,
        mapper_flow_time_condition: bool = True,
        operator_context_film: bool = True,
        operator_context_mode: str = "global",
        operator_norm: str = "group",
        operator_architecture: str = "homogeneous",
        operator_channel_heads: int = 4,
    ) -> None:
        super().__init__()
        if str(operator_architecture).lower() == "structural" and int(num_operators) != 3:
            raise ValueError("operator_architecture='structural' requires num_operators=3")
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.patch_len = int(patch_len)
        self.time_encoder = TimePatchEncoder(
            self.sequence_length,
            self.num_channels,
            self.patch_len,
            d_model,
            layers=transformer_layers,
            heads=transformer_heads,
        )
        self.channel_encoder = ChannelEncoder(
            self.sequence_length,
            self.num_channels,
            d_model,
            patch_len=channel_patch_len or self.patch_len,
            stride=channel_stride,
            temporal_layers=channel_temporal_layers,
            channel_layers=channel_layers,
            heads=channel_heads or transformer_heads,
            dropout=channel_dropout,
        )
        self.text_encoder = text_encoder
        self.mapper_flow_time_condition = bool(mapper_flow_time_condition)
        self.flow_time_proj = None
        if self.mapper_flow_time_condition:
            self.flow_time_proj = nn.Sequential(
                nn.Linear(1, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            nn.init.normal_(self.flow_time_proj[-1].weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.flow_time_proj[-1].bias)
        self.mapper = EffectMapper(
            d_model=d_model,
            num_operators=num_operators,
            field_rank=mapper_field_rank,
            slot_layers=mapper_slot_layers,
            slot_heads=mapper_slot_heads or transformer_heads,
            dropout=mapper_dropout,
            normalizer=mapper_normalizer,
            bounded_field_gate=mapper_bounded_field_gate,
            operator_router=mapper_operator_router,
            series_stats_dim=7,
            router_epsilon=mapper_router_epsilon,
            time_segment_scales=mapper_time_segment_scales,
        )
        self.operator_bank = ResidualOperatorBank(
            num_channels=self.num_channels,
            num_operators=num_operators,
            hidden=operator_hidden,
            t_dim=operator_t_dim,
            max_velocity=operator_max_velocity,
            depth=operator_depth,
            kernel_size=operator_kernel_size,
            dropout=operator_dropout,
            context_dim=d_model,
            context_film=operator_context_film,
            context_mode=operator_context_mode,
            norm_type=operator_norm,
            architecture=operator_architecture,
            channel_heads=operator_channel_heads,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_condition,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B, L, C], got {tuple(x_t.shape)}")
        if x_t.shape[1:] != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected x_t shape [B, {self.sequence_length}, {self.num_channels}], got {tuple(x_t.shape)}")
        if t.shape != (x_t.shape[0],):
            raise ValueError(f"t must have shape [batch], got {tuple(t.shape)} for batch {x_t.shape[0]}")
        if t.device != x_t.device:
            raise ValueError(f"t and x_t must be on the same device, got {t.device} and {x_t.device}")

        time_tokens = self.time_encoder(x_t)
        channel_tokens = self.channel_encoder(x_t)
        slot_tokens, slot_mask = self.text_encoder(text_condition)
        if self.flow_time_proj is not None:
            slot_tokens = slot_tokens + self.flow_time_proj(t[:, None].to(x_t.dtype))[:, None, :]
        text_context = _masked_mean(slot_tokens, slot_mask)
        series_stats = _series_router_stats(x_t)
        g_patch, aux = self.mapper(
            slot_tokens,
            time_tokens,
            channel_tokens,
            slot_mask,
            series_stats=series_stats,
            flow_time=t,
        )
        g = g_patch.repeat_interleave(self.patch_len, dim=1)
        if g.shape[1] < self.sequence_length:
            raise RuntimeError(f"Expanded generation field length {g.shape[1]} is shorter than {self.sequence_length}")
        g = g[:, : self.sequence_length]
        velocities = self.operator_bank(x_t, None, t, context=text_context)
        v_hat = (g * velocities).sum(dim=-1)
        aux = {
            **aux,
            "G_patch": g_patch,
            "G": g,
            "V": velocities,
            "text_context": text_context,
            "series_stats": series_stats.detach(),
        }
        operator_aux = getattr(self.operator_bank, "last_aux", {})
        if operator_aux:
            aux["operator_aux"] = operator_aux
        return v_hat, aux


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    mask = mask.to(tokens.device, dtype=tokens.dtype)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"slot mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}")
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask[:, :, None]).sum(dim=1) / denom


def _series_router_stats(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"x must be [B, L, C], got {tuple(x.shape)}")
    batch, length, channels = x.shape
    dtype = x.dtype
    centered = x - x.mean(dim=1, keepdim=True)
    level = x.mean(dim=(1, 2))
    scale = centered.square().mean(dim=(1, 2)).sqrt()
    if length > 1:
        diff = x[:, 1:] - x[:, :-1]
        roughness = diff.square().mean(dim=(1, 2)).sqrt()
    else:
        roughness = torch.zeros(batch, device=x.device, dtype=dtype)
    fft_source = centered.transpose(1, 2)
    fft_dtype = torch.float32 if fft_source.dtype in {torch.float16, torch.bfloat16} else fft_source.dtype
    freq = torch.fft.rfft(fft_source.to(fft_dtype), dim=-1)
    power = freq.abs().square().mean(dim=1)
    freq_len = power.shape[-1]
    low_end = max(1, int(round(freq_len / 3)))
    mid_end = max(low_end + 1, int(round(2 * freq_len / 3)))
    total = power.sum(dim=-1).clamp_min(1e-8)
    low = power[:, :low_end].sum(dim=-1) / total
    mid = power[:, low_end:mid_end].sum(dim=-1) / total if mid_end > low_end else torch.zeros_like(low)
    high = power[:, mid_end:].sum(dim=-1) / total if mid_end < freq_len else torch.zeros_like(low)
    if channels > 1:
        normed = centered / centered.square().mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
        corr = torch.einsum("blc,bld->bcd", normed, normed) / float(length)
        eye = torch.eye(channels, device=x.device, dtype=torch.bool)
        channel_corr = corr.masked_select(~eye[None]).reshape(batch, -1).abs().mean(dim=-1)
    else:
        channel_corr = torch.zeros(batch, device=x.device, dtype=dtype)
    stats = torch.stack([level, scale, roughness, low.to(dtype), mid.to(dtype), high.to(dtype), channel_corr], dim=-1)
    transformed = torch.log1p(stats.float().abs()) * stats.float().sign()
    return transformed.to(dtype)
