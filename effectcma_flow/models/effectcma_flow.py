from __future__ import annotations

import torch
from torch import nn

from effectcma_flow.models.effect_mapper import EffectMapper
from effectcma_flow.models.operator_bank import ResidualOperatorBank
from effectcma_flow.models.ts_encoder import ChannelEncoder, TimePatchEncoder


class EffectCMAFlow(nn.Module):
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
        mapper_field_gate_mode: str | None = None,
        mapper_field_max_amplitude: float = 8.0,
        mapper_relative_time_position: bool = False,
        mapper_channel_identity: bool = False,
        mapper_flow_time_condition: bool = True,
        operator_context_film: bool = True,
        operator_norm: str = "group",
        operator_architecture: str = "homogeneous",
        operator_types: list[str] | tuple[str, ...] | None = None,
        base_velocity_branch: bool = False,
    ) -> None:
        super().__init__()
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
        self.channel_identity = None
        if mapper_channel_identity:
            self.channel_identity = nn.Parameter(torch.randn(self.num_channels, d_model) * 0.02)
        self.mapper = EffectMapper(
            d_model=d_model,
            num_operators=num_operators,
            field_rank=mapper_field_rank,
            slot_layers=mapper_slot_layers,
            slot_heads=mapper_slot_heads or transformer_heads,
            dropout=mapper_dropout,
            normalizer=mapper_normalizer,
            bounded_field_gate=mapper_bounded_field_gate,
            field_gate_mode=mapper_field_gate_mode,
            field_max_amplitude=mapper_field_max_amplitude,
            relative_time_position=mapper_relative_time_position,
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
            norm_type=operator_norm,
            architecture=operator_architecture,
            operator_types=operator_types,
        )
        self.base_velocity_branch = bool(base_velocity_branch)
        self.base_operator = None
        if self.base_velocity_branch:
            self.base_operator = ResidualOperatorBank(
                num_channels=self.num_channels,
                num_operators=1,
                hidden=operator_hidden,
                t_dim=operator_t_dim,
                max_velocity=operator_max_velocity,
                depth=operator_depth,
                kernel_size=operator_kernel_size,
                dropout=operator_dropout,
                norm_type=operator_norm,
                architecture="homogeneous",
            )

    def forward(
        self,
        base: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_condition,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if base.ndim != 3:
            raise ValueError(f"base must be [B, L, C], got {tuple(base.shape)}")
        if x_t.shape != base.shape:
            raise ValueError(f"x_t must have the same shape as base, got {tuple(x_t.shape)} and {tuple(base.shape)}")
        if x_t.device != base.device:
            raise ValueError(f"x_t and base must be on the same device, got {x_t.device} and {base.device}")
        if x_t.dtype != base.dtype:
            raise ValueError(f"x_t and base must have the same dtype, got {x_t.dtype} and {base.dtype}")
        if t.shape != (base.shape[0],):
            raise ValueError(f"t must have shape [batch], got {tuple(t.shape)} for batch {base.shape[0]}")
        if t.device != base.device:
            raise ValueError(f"t and base must be on the same device, got {t.device} and {base.device}")
        if base.shape[1:] != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected base shape [B, {self.sequence_length}, {self.num_channels}], got {tuple(base.shape)}")
        time_tokens = self.time_encoder(base)
        channel_tokens = self.channel_encoder(base)
        if self.channel_identity is not None:
            channel_tokens = channel_tokens + self.channel_identity[None].to(channel_tokens.dtype)
        slot_tokens, slot_mask = self.text_encoder(text_condition)
        if self.flow_time_proj is not None:
            slot_tokens = slot_tokens + self.flow_time_proj(t[:, None].to(x_t.dtype))[:, None, :]
        text_context = _masked_mean(slot_tokens, slot_mask)
        g_patch, aux = self.mapper(slot_tokens, time_tokens, channel_tokens, slot_mask)
        g = g_patch.repeat_interleave(self.patch_len, dim=1)
        if g.shape[1] < self.sequence_length:
            raise RuntimeError(f"Expanded effect field length {g.shape[1]} is shorter than {self.sequence_length}")
        g = g[:, : self.sequence_length]
        velocities = self.operator_bank(x_t, base, t, context=text_context)
        residual_v = (g * velocities).sum(dim=-1)
        base_v = None
        if self.base_operator is not None:
            base_v = self.base_operator(x_t, base, t).squeeze(-1)
            v_hat = base_v + residual_v
        else:
            v_hat = residual_v
        aux = {**aux, "G_patch": g_patch, "G": g, "V": velocities, "text_context": text_context}
        if base_v is not None:
            aux["base_v"] = base_v
            aux["residual_v"] = residual_v
        return v_hat, aux


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    mask = mask.to(tokens.device, dtype=tokens.dtype)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"slot mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}")
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask[:, :, None]).sum(dim=1) / denom
