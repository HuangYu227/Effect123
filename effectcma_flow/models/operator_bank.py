from __future__ import annotations

import torch
from torch import nn


class DilatedResidualBlock(nn.Module):
    def __init__(
        self,
        width: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        pointwise_groups: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=width),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=1, groups=pointwise_groups),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if y.shape[-1] != x.shape[-1]:
            y = y[..., : x.shape[-1]]
        return self.norm(x + y)


class TemporalOperatorExpert(nn.Module):
    def __init__(
        self,
        *,
        num_channels: int,
        hidden: int,
        depth: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        width = int(num_channels) * int(hidden)
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    width,
                    kernel_size=kernel_size,
                    dilation=2**layer,
                    dropout=dropout,
                    pointwise_groups=int(num_channels),
                )
                for layer in range(max(1, int(depth)))
            ]
        )
        self.head = nn.Conv1d(width, int(num_channels), kernel_size=1, groups=int(num_channels))
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(features))


class ResidualOperatorBank(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_operators: int,
        *,
        hidden: int = 128,
        t_dim: int = 32,
        max_velocity: float = 5.0,
        depth: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_operators = int(num_operators)
        self.hidden = int(hidden)
        self.max_velocity = float(max_velocity)
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.in_proj = nn.Linear(4 + t_dim, hidden)
        width = self.hidden * self.num_channels
        self.channel_mixer = nn.Linear(self.num_channels, self.num_channels, bias=False)
        self.shared_temporal = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width),
            nn.SiLU(),
        )
        self.experts = nn.ModuleList(
            [
                TemporalOperatorExpert(
                    num_channels=self.num_channels,
                    hidden=self.hidden,
                    depth=depth,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(self.num_operators)
            ]
        )

    def forward(self, x_t: torch.Tensor, base: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.shape != base.shape:
            raise ValueError(f"x_t and base shapes must match, got {tuple(x_t.shape)} and {tuple(base.shape)}")
        batch, length, channels = x_t.shape
        if t.shape != (batch,):
            raise ValueError(f"t must have shape [batch], got {tuple(t.shape)}")
        if t.device != x_t.device:
            raise ValueError(f"t and x_t must be on the same device, got {t.device} and {x_t.device}")
        if base.dtype != x_t.dtype:
            raise ValueError(f"base and x_t must have same dtype, got {base.dtype} and {x_t.dtype}")
        if channels != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got {channels}")
        t_emb = self.t_embed(t[:, None].to(x_t.dtype))
        t_emb = t_emb[:, None, None, :].expand(batch, length, channels, -1)
        pos = torch.linspace(-1.0, 1.0, length, device=x_t.device, dtype=x_t.dtype)
        pos = pos[None, :, None].expand(batch, length, channels)
        feat = torch.stack([x_t, base, x_t - base, pos], dim=-1)
        feat = torch.cat([feat, t_emb], dim=-1)
        h = self.in_proj(feat)
        h = h + self.channel_mixer(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        h2 = h.permute(0, 2, 3, 1).reshape(batch, channels * self.hidden, length)
        h2 = self.shared_temporal(h2)
        expert_outputs = [expert(h2).transpose(1, 2) for expert in self.experts]
        velocities = torch.stack(expert_outputs, dim=-1)
        return torch.tanh(velocities) * self.max_velocity
