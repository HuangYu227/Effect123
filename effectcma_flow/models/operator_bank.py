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


class AdaFeatureNorm(nn.Module):
    def __init__(self, hidden: int, t_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_dim, 2 * hidden),
        )
        nn.init.zeros_(self.to_scale_shift[-1].weight)
        nn.init.zeros_(self.to_scale_shift[-1].bias)

    def forward(self, x: torch.Tensor, t_code: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale, shift = self.to_scale_shift(t_code).chunk(2, dim=-1)
        y = self.norm(x) * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]
        return y, scale, shift


class TemporalSegmentExpert(nn.Module):
    """Velocity expert constrained to the time dimension with multi-scale dilated convolutions."""

    def __init__(self, *, num_channels: int, hidden: int, t_dim: int, dropout: float, norm_type: str) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.hidden = int(hidden)
        self.ada_norm = AdaFeatureNorm(hidden, t_dim)
        self.local = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.mid = nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2, groups=hidden)
        self.long = nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4, groups=hidden)
        self.mix = nn.Sequential(
            nn.Conv1d(3 * hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)
        _init_small_head(self.head)

    def forward(self, h: torch.Tensor, t_code: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels, hidden = h.shape
        h_n, scale, shift = self.ada_norm(h, t_code)
        x = h_n.permute(0, 2, 3, 1).reshape(batch * channels, hidden, length)
        y = self.mix(torch.cat([self.local(x), self.mid(x), self.long(x)], dim=1))
        out = self.head(y).reshape(batch, channels, length).transpose(1, 2)
        aux = {
            "time_film_scale_rms": scale.detach().square().mean().sqrt(),
            "time_film_shift_rms": shift.detach().square().mean().sqrt(),
        }
        return out, aux


class ChannelInteractionExpert(nn.Module):
    """Velocity expert whose main operation is self-attention over channels."""

    def __init__(self, *, num_channels: int, hidden: int, t_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.hidden = int(hidden)
        heads = max(1, min(int(heads), hidden))
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        self.ada_norm = AdaFeatureNorm(hidden, t_dim)
        self.channel_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: torch.Tensor, t_code: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels, hidden = h.shape
        h_n, scale, shift = self.ada_norm(h, t_code)
        x = h_n.reshape(batch * length, channels, hidden)
        attended, attn_weights = self.channel_attn(x, x, x, need_weights=True, average_attn_weights=False)
        y = attended + self.ffn(attended)
        out = self.head(y).reshape(batch, length, channels)
        aux = {
            "channel_attn_entropy": _entropy(attn_weights.detach().mean(dim=1), dim=-1).mean(),
            "channel_film_scale_rms": scale.detach().square().mean().sqrt(),
            "channel_film_shift_rms": shift.detach().square().mean().sqrt(),
        }
        return out, aux


class FrequencyBandExpert(nn.Module):
    """Velocity expert constrained by fixed low/mid/high FFT band decomposition."""

    def __init__(self, *, num_channels: int, hidden: int, t_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.hidden = int(hidden)
        self.ada_norm = AdaFeatureNorm(hidden, t_dim)
        self.band_mixer = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: torch.Tensor, t_code: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels, hidden = h.shape
        h_n, scale, shift = self.ada_norm(h, t_code)
        x = h_n.permute(0, 2, 3, 1).reshape(batch * channels, hidden, length)
        fft_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        freq = torch.fft.rfft(x.to(fft_dtype), dim=-1)
        bands = _fixed_frequency_bands(freq.shape[-1], freq.device)
        components = []
        energy = []
        for mask in bands:
            mask_t = mask[None, None, :]
            filtered = torch.fft.irfft(freq * mask_t, n=length, dim=-1)
            components.append(filtered.to(x.dtype))
            energy.append((freq.abs().square() * mask_t).mean(dim=(1, 2)))
        stacked = torch.cat(components, dim=1)
        y = self.band_mixer(stacked.transpose(1, 2)).transpose(1, 2)
        out = self.head(y).reshape(batch, channels, length).transpose(1, 2)
        band_energy = torch.stack(energy, dim=-1).reshape(batch, channels, 3).mean(dim=1).to(h.dtype)
        aux = {
            "frequency_band_energy": band_energy.detach(),
            "frequency_film_scale_rms": scale.detach().square().mean().sqrt(),
            "frequency_film_shift_rms": shift.detach().square().mean().sqrt(),
        }
        return out, aux


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
        context_mode: str = "global",
        norm_type: str = "group",
        architecture: str = "homogeneous",
        channel_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_operators = int(num_operators)
        self.hidden = int(hidden)
        self.max_velocity = float(max_velocity)
        self.architecture = str(architecture).lower()
        if self.architecture not in {"homogeneous", "structural"}:
            raise ValueError(f"architecture must be 'homogeneous' or 'structural', got {architecture!r}")
        if self.architecture == "structural":
            if self.num_operators != 3:
                raise ValueError("structural operator bank requires num_operators=3")
            self.operator_type_names = ["time", "channel", "frequency"]
        else:
            self.operator_type_names = [f"temporal_{i}" for i in range(self.num_operators)]
        self.context_mode = str(context_mode).lower()
        if self.context_mode not in {"global", "none"}:
            raise ValueError(f"context_mode must be 'global' or 'none', got {context_mode!r}")
        self.t_embed = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.in_proj = nn.Linear(4 + t_dim, hidden)
        self.channel_mixer = (
            nn.Linear(self.num_channels, self.num_channels, bias=False)
            if self.architecture == "homogeneous"
            else None
        )
        use_context = self.context_mode == "global" and context_dim is not None
        self.context_proj = nn.Linear(int(context_dim), self.hidden) if use_context else None
        self.context_film = nn.Linear(int(context_dim) + t_dim, 2 * self.hidden) if use_context and context_film else None
        if self.context_film is not None:
            nn.init.zeros_(self.context_film.weight)
            nn.init.zeros_(self.context_film.bias)
        if self.architecture == "homogeneous":
            width = self.hidden * self.num_channels
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
                        norm_type=norm_type,
                    )
                    for _ in range(self.num_operators)
                ]
            )
        else:
            self.shared_temporal = None
            self.experts = nn.ModuleList(
                [
                    TemporalSegmentExpert(
                        num_channels=self.num_channels,
                        hidden=self.hidden,
                        t_dim=t_dim,
                        dropout=dropout,
                        norm_type=norm_type,
                    ),
                    ChannelInteractionExpert(
                        num_channels=self.num_channels,
                        hidden=self.hidden,
                        t_dim=t_dim,
                        heads=channel_heads,
                        dropout=dropout,
                    ),
                    FrequencyBandExpert(
                        num_channels=self.num_channels,
                        hidden=self.hidden,
                        t_dim=t_dim,
                        dropout=dropout,
                    ),
                ]
            )
        self.last_aux: dict[str, torch.Tensor] = {}

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
        if context is not None and self.context_mode == "global":
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
        if self.channel_mixer is not None:
            h = h + self.channel_mixer(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        if self.architecture == "structural":
            expert_outputs = []
            expert_aux: dict[str, torch.Tensor] = {}
            for name, expert in zip(self.operator_type_names, self.experts):
                out, one_aux = expert(h, t_code)
                expert_outputs.append(out)
                for key, value in one_aux.items():
                    expert_aux[f"{name}_{key}"] = value
            velocities = torch.stack(expert_outputs, dim=-1)
            self.last_aux = expert_aux
            return torch.tanh(velocities) * self.max_velocity
        h2 = h.permute(0, 2, 3, 1).reshape(batch, channels * self.hidden, length)
        if self.shared_temporal is None:
            raise RuntimeError("shared_temporal is not initialized")
        h2 = self.shared_temporal(h2)
        expert_outputs = [expert(h2).transpose(1, 2) for expert in self.experts]
        velocities = torch.stack(expert_outputs, dim=-1)
        self.last_aux = {}
        return torch.tanh(velocities) * self.max_velocity


def _init_small_head(head: nn.Conv1d) -> None:
    nn.init.normal_(head.weight, mean=0.0, std=1e-3)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


def _fixed_frequency_bands(freq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(freq_len, device=device)
    low_end = max(1, int(round(freq_len / 3)))
    mid_end = max(low_end + 1, int(round(2 * freq_len / 3)))
    low = (idx < low_end).to(torch.float32)
    mid = ((idx >= low_end) & (idx < mid_end)).to(torch.float32)
    high = (idx >= mid_end).to(torch.float32)
    if high.sum() == 0:
        high[-1] = 1.0
    return low, mid, high


def _entropy(prob: torch.Tensor, *, dim: int, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)
