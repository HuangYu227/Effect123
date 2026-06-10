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
        norm_type: str = "group",
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
        if norm_type == "group":
            self.norm = nn.GroupNorm(1, width)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(width)
        else:
            raise ValueError(f"norm_type must be 'group' or 'batch', got {norm_type!r}")

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
        norm_type: str,
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
                    norm_type=norm_type,
                )
                for layer in range(max(1, int(depth)))
            ]
        )
        self.head = nn.Conv1d(width, int(num_channels), kernel_size=1, groups=int(num_channels))
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(features))


class TrendOperatorExpert(nn.Module):
    """Low-frequency expert for level and trend-like dynamics."""

    def __init__(self, *, num_channels: int, hidden: int, depth: int, kernel_size: int, dropout: float, norm_type: str) -> None:
        super().__init__()
        width = int(num_channels) * int(hidden)
        smooth_kernel = max(5, int(kernel_size) * 2 + 1)
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        self.lowpass = nn.AvgPool1d(smooth_kernel, stride=1, padding=smooth_kernel // 2, count_include_pad=False)
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    width,
                    kernel_size=max(3, int(kernel_size)),
                    dilation=2 ** min(layer, 3),
                    dropout=dropout,
                    pointwise_groups=int(num_channels),
                    norm_type=norm_type,
                )
                for layer in range(max(1, int(depth)))
            ]
        )
        self.head = nn.Conv1d(width, int(num_channels), kernel_size=1, groups=int(num_channels))
        _init_small_head(self.head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.lowpass(features)))


class LocalEventOperatorExpert(nn.Module):
    """Local expert for spike/drop-like short-range events."""

    def __init__(self, *, num_channels: int, hidden: int, depth: int, kernel_size: int, dropout: float, norm_type: str) -> None:
        super().__init__()
        width = int(num_channels) * int(hidden)
        local_kernel = 3 if int(kernel_size) <= 3 else int(kernel_size)
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    width,
                    kernel_size=local_kernel,
                    dilation=1,
                    dropout=dropout,
                    pointwise_groups=int(num_channels),
                    norm_type=norm_type,
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.head = nn.Conv1d(width, int(num_channels), kernel_size=1, groups=int(num_channels))
        _init_small_head(self.head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(features))


class VolatilityOperatorExpert(nn.Module):
    """Amplitude-modulation expert for variance and volatility shifts."""

    def __init__(self, *, num_channels: int, hidden: int, depth: int, kernel_size: int, dropout: float, norm_type: str) -> None:
        super().__init__()
        width = int(num_channels) * int(hidden)
        self.lowpass = nn.AvgPool1d(5, stride=1, padding=2, count_include_pad=False)
        self.envelope = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width),
            nn.SiLU(),
            nn.Conv1d(width, width, kernel_size=1, groups=int(num_channels)),
            nn.Sigmoid(),
        )
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    width,
                    kernel_size=max(3, int(kernel_size)),
                    dilation=2**layer,
                    dropout=dropout,
                    pointwise_groups=int(num_channels),
                    norm_type=norm_type,
                )
                for layer in range(max(1, int(depth)))
            ]
        )
        self.head = nn.Conv1d(width, int(num_channels), kernel_size=1, groups=int(num_channels))
        _init_small_head(self.head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features - self.lowpass(features)
        modulated = features + residual * self.envelope(features)
        return self.head(self.blocks(modulated))


class SpectralOperatorExpert(nn.Module):
    """Multi-scale expert that separates smooth and high-frequency components."""

    def __init__(self, *, num_channels: int, hidden: int, depth: int, kernel_size: int, dropout: float, norm_type: str) -> None:
        super().__init__()
        width = int(num_channels) * int(hidden)
        smooth_kernel = max(7, int(kernel_size) * 4 + 1)
        if smooth_kernel % 2 == 0:
            smooth_kernel += 1
        self.lowpass = nn.AvgPool1d(smooth_kernel, stride=1, padding=smooth_kernel // 2, count_include_pad=False)
        self.mix = nn.Conv1d(2 * width, width, kernel_size=1, groups=int(num_channels))
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    width,
                    kernel_size=max(3, int(kernel_size)),
                    dilation=2 ** min(layer, 4),
                    dropout=dropout,
                    pointwise_groups=int(num_channels),
                    norm_type=norm_type,
                )
                for layer in range(max(1, int(depth)))
            ]
        )
        self.head = nn.Conv1d(width, int(num_channels), kernel_size=1, groups=int(num_channels))
        _init_small_head(self.head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        low = self.lowpass(features)
        high = features - low
        return self.head(self.blocks(self.mix(torch.cat([low, high], dim=1))))


class ChannelPropagationOperatorExpert(nn.Module):
    """Cross-channel expert for spatial or variable-interaction effects."""

    def __init__(self, *, num_channels: int, hidden: int, depth: int, kernel_size: int, dropout: float, norm_type: str) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.hidden = int(hidden)
        width = self.num_channels * self.hidden
        self.channel_mix = nn.Linear(self.num_channels, self.num_channels, bias=False)
        self.blocks = nn.Sequential(
            *[
                DilatedResidualBlock(
                    width,
                    kernel_size=max(3, int(kernel_size)),
                    dilation=2 ** min(layer, 3),
                    dropout=dropout,
                    pointwise_groups=1,
                    norm_type=norm_type,
                )
                for layer in range(max(1, int(depth)))
            ]
        )
        self.head = nn.Conv1d(width, self.num_channels, kernel_size=1)
        _init_small_head(self.head)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, _, length = features.shape
        h = features.reshape(batch, self.num_channels, self.hidden, length)
        mixed = self.channel_mix(h.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        h = (h + mixed).reshape(batch, self.num_channels * self.hidden, length)
        return self.head(self.blocks(h))


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
        context_dim: int | None = None,
        context_film: bool = True,
        norm_type: str = "group",
        architecture: str = "homogeneous",
        operator_types: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_operators = int(num_operators)
        self.hidden = int(hidden)
        self.max_velocity = float(max_velocity)
        self.architecture = str(architecture).lower()
        if self.architecture not in {"homogeneous", "heterogeneous"}:
            raise ValueError(f"architecture must be 'homogeneous' or 'heterogeneous', got {architecture!r}")
        if operator_types is None:
            default_types = ["trend", "local", "volatility", "spectral", "channel"]
            self._operator_type_names = [default_types[i % len(default_types)] for i in range(self.num_operators)]
        else:
            names = [str(name).lower() for name in operator_types]
            if not names:
                raise ValueError("operator_types must not be empty")
            self._operator_type_names = [names[i % len(names)] for i in range(self.num_operators)]
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.in_proj = nn.Linear(4 + t_dim, hidden)
        width = self.hidden * self.num_channels
        self.channel_mixer = nn.Linear(self.num_channels, self.num_channels, bias=False)
        self.context_proj = nn.Linear(int(context_dim), self.hidden) if context_dim is not None else None
        self.context_film = nn.Linear(int(context_dim) + t_dim, 2 * self.hidden) if context_dim is not None and context_film else None
        if self.context_film is not None:
            nn.init.zeros_(self.context_film.weight)
            nn.init.zeros_(self.context_film.bias)
        self.shared_temporal = nn.Sequential(
            nn.Conv1d(width, width, kernel_size=3, padding=1, groups=width),
            nn.SiLU(),
        )
        self.experts = nn.ModuleList(
            [
                _make_expert(
                    "temporal" if self.architecture == "homogeneous" else self.operator_type_names[i],
                    num_channels=self.num_channels,
                    hidden=self.hidden,
                    depth=depth,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(self.num_operators)
            ]
        )

    @property
    def operator_type_names(self) -> list[str]:
        return self._operator_type_names

    def forward(self, x_t: torch.Tensor, base: torch.Tensor | None, t: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if base is None:
            base = torch.zeros_like(x_t)
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
        t_code = self.t_embed(t[:, None].to(x_t.dtype))
        t_emb = t_code[:, None, None, :].expand(batch, length, channels, -1)
        pos = torch.linspace(-1.0, 1.0, length, device=x_t.device, dtype=x_t.dtype)
        pos = pos[None, :, None].expand(batch, length, channels)
        feat = torch.stack([x_t, base, x_t - base, pos], dim=-1)
        feat = torch.cat([feat, t_emb], dim=-1)
        h = self.in_proj(feat)
        if context is not None:
            if self.context_proj is None:
                raise ValueError("context was provided but this ResidualOperatorBank was created without context_dim")
            if context.shape != (batch, self.context_proj.in_features):
                raise ValueError(
                    f"context must have shape [batch, {self.context_proj.in_features}], got {tuple(context.shape)}"
                )
            context = context.to(x_t.dtype)
            h = h + self.context_proj(context)[:, None, None, :]
            if self.context_film is not None:
                film = self.context_film(torch.cat([context, t_code], dim=-1))
                gamma, beta = film.chunk(2, dim=-1)
                h = h * (1.0 + 0.1 * torch.tanh(gamma)[:, None, None, :]) + 0.1 * beta[:, None, None, :]
        elif self.context_proj is not None:
            h = h + self.context_proj.weight.new_zeros((batch, 1, 1, self.context_proj.out_features))
        h = h + self.channel_mixer(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        h2 = h.permute(0, 2, 3, 1).reshape(batch, channels * self.hidden, length)
        h2 = self.shared_temporal(h2)
        expert_outputs = [expert(h2).transpose(1, 2) for expert in self.experts]
        velocities = torch.stack(expert_outputs, dim=-1)
        return torch.tanh(velocities) * self.max_velocity


def _init_small_head(head: nn.Conv1d) -> None:
    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


def _make_expert(
    kind: str,
    *,
    num_channels: int,
    hidden: int,
    depth: int,
    kernel_size: int,
    dropout: float,
    norm_type: str,
) -> nn.Module:
    kwargs = dict(
        num_channels=num_channels,
        hidden=hidden,
        depth=depth,
        kernel_size=kernel_size,
        dropout=dropout,
        norm_type=norm_type,
    )
    if kind in {"temporal", "tcn", "homogeneous"}:
        return TemporalOperatorExpert(**kwargs)
    if kind in {"trend", "level", "level_trend"}:
        return TrendOperatorExpert(**kwargs)
    if kind in {"local", "spike", "drop", "event"}:
        return LocalEventOperatorExpert(**kwargs)
    if kind in {"volatility", "variance", "amplitude"}:
        return VolatilityOperatorExpert(**kwargs)
    if kind in {"spectral", "frequency", "multiscale"}:
        return SpectralOperatorExpert(**kwargs)
    if kind in {"channel", "propagation", "spatial"}:
        return ChannelPropagationOperatorExpert(**kwargs)
    raise ValueError(f"Unknown operator expert type {kind!r}")
