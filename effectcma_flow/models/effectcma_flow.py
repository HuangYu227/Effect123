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
        mapper_gate_rescale: str | float = "auto",
        mapper_flow_time_condition: bool = True,
        operator_context_film: bool = True,
        operator_norm: str = "group",
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
        self.mapper = EffectMapper(
            d_model=d_model,
            num_operators=num_operators,
            field_rank=mapper_field_rank,
            slot_layers=mapper_slot_layers,
            slot_heads=mapper_slot_heads or transformer_heads,
            dropout=mapper_dropout,
            normalizer=mapper_normalizer,
            bounded_field_gate=mapper_bounded_field_gate,
            gate_rescale=mapper_gate_rescale,
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
            norm_type=operator_norm,
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
        slot_tokens, slot_mask = self.text_encoder(text_condition)
        g_patch, aux = self.mapper(slot_tokens, time_tokens, channel_tokens, slot_mask)
        g = g_patch.repeat_interleave(self.patch_len, dim=1)
        if g.shape[1] < self.sequence_length:
            raise RuntimeError(f"Expanded effect field length {g.shape[1]} is shorter than {self.sequence_length}")
        g = g[:, : self.sequence_length]
        velocities = self.operator_bank(x_t, base, t)
        v_hat = (g * velocities).sum(dim=-1)
        aux = {**aux, "G_patch": g_patch, "G": g, "V": velocities}
        return v_hat, aux
