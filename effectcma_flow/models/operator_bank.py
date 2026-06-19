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

    def __init__(
        self,
        *,
        num_channels: int,
        hidden: int,
        t_dim: int,
        dropout: float,
        norm_type: str,
        long_range_mode: str = "none",
        long_range_scales: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.hidden = int(hidden)
        self.long_range_mode = str(long_range_mode).lower()
        if self.long_range_mode not in {"none", "multiscale"}:
            raise ValueError(f"long_range_mode must be 'none' or 'multiscale', got {long_range_mode!r}")
        self.ada_norm = AdaFeatureNorm(hidden, t_dim)
        self.local = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.mid = nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2, groups=hidden)
        self.long = nn.Conv1d(hidden, hidden, kernel_size=7, padding=9, dilation=3, groups=hidden)
        self.long_range = (
            MultiScaleTemporalMixer(hidden=hidden, scales=long_range_scales or (2, 4, 8), dropout=dropout)
            if self.long_range_mode == "multiscale"
            else None
        )
        mix_inputs = 4 * hidden if self.long_range is not None else 3 * hidden
        self.mix = nn.Sequential(
            nn.Conv1d(mix_inputs, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)
        _init_small_head(self.head)

    def forward(self, h: torch.Tensor, t_code: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels, hidden = h.shape
        h_n, scale, shift = self.ada_norm(h, t_code)
        x = h_n.permute(0, 2, 3, 1).reshape(batch * channels, hidden, length)
        branches = [self.local(x), self.mid(x), self.long(x)]
        if self.long_range is not None:
            branches.append(self.long_range(x))
        y = self.mix(torch.cat(branches, dim=1))
        out = self.head(y).reshape(batch, channels, length).transpose(1, 2)
        aux = {
            "time_film_scale_rms": scale.detach().square().mean().sqrt(),
            "time_film_shift_rms": shift.detach().square().mean().sqrt(),
        }
        if self.long_range is not None:
            aux["long_range_rms"] = branches[-1].detach().square().mean().sqrt()
        return out, aux


class MultiScaleTemporalMixer(nn.Module):
    """Dependency-free TimeMixer-style coarse-to-fine temporal context branch.

    The branch downsamples each variable's hidden trajectory to several coarse
    scales, applies lightweight temporal mixing there, and interpolates the
    result back to the original length. It gives the temporal expert an explicit
    long-range path without adding S4/Mamba custom kernels or changing the
    operator count.
    """

    def __init__(self, *, hidden: int, scales: tuple[int, ...] | list[int], dropout: float) -> None:
        super().__init__()
        clean_scales = []
        for value in scales:
            scale = int(value)
            if scale > 1 and scale not in clean_scales:
                clean_scales.append(scale)
        if not clean_scales:
            raise ValueError("MultiScaleTemporalMixer requires at least one scale > 1")
        self.scales = tuple(clean_scales)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
                    nn.GELU(),
                    nn.Conv1d(hidden, hidden, kernel_size=1),
                    nn.Dropout(dropout),
                )
                for _ in self.scales
            ]
        )
        self.mix = nn.Sequential(
            nn.Conv1d(len(self.scales) * hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B*C,H,L] temporal features, got {tuple(x.shape)}")
        length = int(x.shape[-1])
        branches = []
        for scale, block in zip(self.scales, self.blocks):
            kernel = min(int(scale), length)
            pooled = torch.nn.functional.avg_pool1d(x, kernel_size=kernel, stride=kernel, ceil_mode=True)
            mixed = block(pooled)
            up = torch.nn.functional.interpolate(mixed, size=length, mode="linear", align_corners=False)
            branches.append(up)
        return self.mix(torch.cat(branches, dim=1))


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
    """Velocity expert constrained by adaptive FFT band decomposition."""

    def __init__(
        self,
        *,
        num_channels: int,
        hidden: int,
        t_dim: int,
        dropout: float,
        frequency_band_mode: str = "gaussian",
        frequency_topk_frac: float = 0.15,
        frequency_temp_start: float = 1.0,
        frequency_temp_end: float = 0.1,
        frequency_anneal_steps: int = 10000,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.hidden = int(hidden)
        self.frequency_band_mode = str(frequency_band_mode).lower()
        if self.frequency_band_mode not in {"gaussian", "soft_topk"}:
            raise ValueError(
                f"frequency_band_mode must be 'gaussian' or 'soft_topk', got {frequency_band_mode!r}"
            )
        self.frequency_topk_frac = float(frequency_topk_frac)
        self.frequency_temp_start = float(frequency_temp_start)
        self.frequency_temp_end = float(frequency_temp_end)
        self.frequency_anneal_steps = max(1, int(frequency_anneal_steps))
        # Optional direct temperature override (None -> use the annealed schedule).
        # Tests set this to keep the soft top-K mask deterministic.
        self.frequency_temperature_override: float | None = None
        self.ada_norm = AdaFeatureNorm(hidden, t_dim)
        centers = torch.tensor([0.08, 0.32, 0.72], dtype=torch.float32).clamp(1e-3, 0.999)
        widths = torch.tensor([0.12, 0.18, 0.22], dtype=torch.float32).clamp_min(1e-3)
        self.band_center_logits = nn.Parameter(torch.logit(centers))
        self.band_log_widths = nn.Parameter(widths.log())
        # Only the soft top-K path needs a training-progress signal. Registering
        # the buffer exclusively in that mode keeps the default ("gaussian")
        # state_dict byte-for-byte identical to the pre-upgrade module.
        if self.frequency_band_mode == "soft_topk":
            self.register_buffer("frequency_anneal_step", torch.zeros((), dtype=torch.long))
        self.band_gate = nn.Sequential(
            nn.LayerNorm(hidden + t_dim),
            nn.Linear(hidden + t_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )
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
        pooled = h_n.mean(dim=(1, 2))
        band_gate = torch.softmax(self.band_gate(torch.cat([pooled, t_code], dim=-1)), dim=-1).to(fft_dtype)
        if self.frequency_band_mode == "soft_topk":
            # Amplitude-ranked, temperature-annealed soft spectral mask. Computed
            # from the RAW rFFT magnitude (not the gated energy below) so there is
            # no gradient self-loop, and fully differentiable (no argmax/topk
            # index masking). Shared across the 3 band slots; the per-sample
            # band_gate then mixes the same sparse spectrum into the mixer inputs.
            magnitude = freq.abs()
            temperature = self._frequency_temperature(magnitude.dtype, magnitude.device)
            soft_mask = torch.softmax(magnitude / temperature, dim=-1)
            # softmax sums to 1 over frequency; scaling by ~topk_frac * F gives an
            # average per-bin weight of frequency_topk_frac, i.e. a soft top-K gain
            # that "keeps" roughly that fraction of bins by energy. Stays >= 0.
            soft_mask = soft_mask * (self.frequency_topk_frac * magnitude.shape[-1])
            bands = (soft_mask, soft_mask, soft_mask)
        else:
            bands = _adaptive_frequency_bands(
                freq.shape[-1],
                freq.device,
                dtype=fft_dtype,
                center_logits=self.band_center_logits,
                log_widths=self.band_log_widths,
            )
            bands = (bands[0][None, None, :], bands[1][None, None, :], bands[2][None, None, :])
        components = []
        energy = []
        for band_idx, mask_t in enumerate(bands):
            sample_gate = band_gate[:, band_idx].repeat_interleave(channels)[:, None, None]
            mask_t = mask_t * sample_gate
            filtered = torch.fft.irfft(freq * mask_t, n=length, dim=-1)
            components.append(filtered.to(x.dtype))
            energy.append((freq.abs().square() * mask_t).mean(dim=(1, 2)))
        stacked = torch.cat(components, dim=1)
        y = self.band_mixer(stacked.transpose(1, 2)).transpose(1, 2)
        out = self.head(y.transpose(1, 2)).squeeze(-1).reshape(batch, channels, length).transpose(1, 2)
        band_energy = torch.stack(energy, dim=-1).reshape(batch, channels, 3).mean(dim=1).to(h.dtype)
        aux = {
            "frequency_band_energy": band_energy.detach(),
            "frequency_band_gate": band_gate.detach().to(h.dtype),
            "frequency_band_centers": torch.sigmoid(self.band_center_logits).detach().to(h.dtype),
            "frequency_band_widths": torch.exp(self.band_log_widths).detach().to(h.dtype),
            "frequency_film_scale_rms": scale.detach().square().mean().sqrt(),
            "frequency_film_shift_rms": shift.detach().square().mean().sqrt(),
        }
        return out, aux

    def _frequency_temperature(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Temperature for the soft top-K spectral mask.

        Uses ``frequency_temperature_override`` when set (deterministic for
        tests); otherwise anneals linearly from ``frequency_temp_start`` down to
        ``frequency_temp_end`` over ``frequency_anneal_steps``. The step counter
        buffer is advanced only on training-mode forwards so eval/sampling is
        reproducible.
        """
        if self.frequency_temperature_override is not None:
            value = float(self.frequency_temperature_override)
        else:
            step = float(self.frequency_anneal_step.item()) if hasattr(self, "frequency_anneal_step") else 0.0
            frac = min(1.0, max(0.0, step / float(self.frequency_anneal_steps)))
            value = self.frequency_temp_end + (self.frequency_temp_start - self.frequency_temp_end) * (1.0 - frac)
            if self.training and hasattr(self, "frequency_anneal_step"):
                self.frequency_anneal_step += 1
        value = max(value, 1e-4)
        return torch.tensor(value, dtype=dtype, device=device)


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
        multiview_context: bool = False,
        norm_type: str = "group",
        architecture: str = "homogeneous",
        channel_heads: int = 4,
        frequency_band_mode: str = "gaussian",
        frequency_topk_frac: float = 0.15,
        frequency_temp_start: float = 1.0,
        frequency_temp_end: float = 0.1,
        frequency_anneal_steps: int = 10000,
        temporal_long_range_mode: str = "none",
        temporal_long_range_scales: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.num_operators = int(num_operators)
        self.hidden = int(hidden)
        self.t_dim = int(t_dim)
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
        self.multiview_context = bool(multiview_context)
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
        self.expert_context_proj = (
            nn.ModuleList([nn.Linear(int(context_dim), self.hidden) for _ in range(self.num_operators)])
            if use_context
            else None
        )
        self.expert_context_scale = nn.Parameter(torch.full((self.num_operators,), 0.1)) if use_context else None
        use_multiview = use_context and self.multiview_context
        self.channel_context_proj = nn.Linear(int(context_dim), self.hidden) if use_multiview else None
        self.channel_context_scale = nn.Parameter(torch.tensor(0.1)) if use_multiview else None
        self.expert_t_proj = (
            nn.ModuleList([nn.Linear(int(context_dim), self.t_dim) for _ in range(self.num_operators)])
            if use_multiview
            else None
        )
        self.expert_t_scale = nn.Parameter(torch.full((self.num_operators,), 0.1)) if use_multiview else None
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
                        long_range_mode=temporal_long_range_mode,
                        long_range_scales=temporal_long_range_scales,
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
                        frequency_band_mode=frequency_band_mode,
                        frequency_topk_frac=frequency_topk_frac,
                        frequency_temp_start=frequency_temp_start,
                        frequency_temp_end=frequency_temp_end,
                        frequency_anneal_steps=frequency_anneal_steps,
                    ),
                ]
            )
        self.last_aux: dict[str, torch.Tensor] = {}

    def forward(
        self,
        x_t: torch.Tensor,
        base: torch.Tensor | None,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
        expert_context: torch.Tensor | None = None,
        channel_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B, L, C], got {tuple(x_t.shape)}")
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
        if expert_context is not None:
            if self.expert_context_proj is None or self.context_proj is None:
                raise ValueError("expert_context was provided but this ResidualOperatorBank was created without context_dim")
            expected = (batch, self.num_operators, self.context_proj.in_features)
            if expert_context.shape != expected:
                raise ValueError(f"expert_context must have shape {expected}, got {tuple(expert_context.shape)}")
            expert_context = expert_context.to(device=x_t.device, dtype=x_t.dtype)
        if channel_context is not None:
            if self.channel_context_proj is None or self.context_proj is None:
                raise ValueError(
                    "channel_context was provided but this ResidualOperatorBank was created without multiview context"
                )
            expected_channel = (batch, channels, self.context_proj.in_features)
            if channel_context.shape != expected_channel:
                raise ValueError(f"channel_context must have shape {expected_channel}, got {tuple(channel_context.shape)}")
            channel_context = channel_context.to(device=x_t.device, dtype=x_t.dtype)
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
        if channel_context is not None:
            if self.channel_context_proj is None or self.channel_context_scale is None:
                raise RuntimeError("channel_context projection is not initialized")
            channel_delta = self.channel_context_proj(channel_context)[:, None, :, :]
            h = h + self.channel_context_scale.to(device=h.device, dtype=h.dtype) * channel_delta
        if self.channel_mixer is not None:
            h = h + self.channel_mixer(h.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        if self.architecture == "structural":
            expert_outputs = []
            expert_aux: dict[str, torch.Tensor] = {}
            for idx, (name, expert) in enumerate(zip(self.operator_type_names, self.experts)):
                expert_h = self._apply_expert_context(h, expert_context, idx)
                expert_t_code = self._apply_expert_time_context(t_code, expert_context, idx)
                out, one_aux = expert(expert_h, expert_t_code)
                expert_outputs.append(out)
                for key, value in one_aux.items():
                    expert_aux[f"{name}_{key}"] = value
            if expert_context is not None:
                expert_aux["expert_context_norm"] = expert_context.detach().norm(dim=-1).mean()
            if channel_context is not None:
                expert_aux["channel_context_norm"] = channel_context.detach().norm(dim=-1).mean()
            if expert_context is not None and self.expert_t_proj is not None:
                expert_aux["expert_time_context_norm"] = expert_context.detach().norm(dim=-1).mean()
            velocities = torch.stack(expert_outputs, dim=-1)
            self.last_aux = expert_aux
            return torch.tanh(velocities) * self.max_velocity
        if self.shared_temporal is None:
            raise RuntimeError("shared_temporal is not initialized")
        if expert_context is None:
            h2 = h.permute(0, 2, 3, 1).reshape(batch, channels * self.hidden, length)
            h2 = self.shared_temporal(h2)
            expert_outputs = [expert(h2).transpose(1, 2) for expert in self.experts]
            self.last_aux = {}
        else:
            expert_outputs = []
            for idx, expert in enumerate(self.experts):
                expert_h = self._apply_expert_context(h, expert_context, idx)
                h2 = expert_h.permute(0, 2, 3, 1).reshape(batch, channels * self.hidden, length)
                h2 = self.shared_temporal(h2)
                expert_outputs.append(expert(h2).transpose(1, 2))
            self.last_aux = {"expert_context_norm": expert_context.detach().norm(dim=-1).mean()}
            if channel_context is not None:
                self.last_aux["channel_context_norm"] = channel_context.detach().norm(dim=-1).mean()
        velocities = torch.stack(expert_outputs, dim=-1)
        return torch.tanh(velocities) * self.max_velocity

    def _apply_expert_context(self, h: torch.Tensor, expert_context: torch.Tensor | None, idx: int) -> torch.Tensor:
        if expert_context is None:
            return h
        if self.expert_context_proj is None or self.expert_context_scale is None:
            return h
        delta = self.expert_context_proj[idx](expert_context[:, idx])[:, None, None, :]
        scale = self.expert_context_scale[idx].to(device=h.device, dtype=h.dtype)
        return h + scale * delta

    def _apply_expert_time_context(
        self,
        t_code: torch.Tensor,
        expert_context: torch.Tensor | None,
        idx: int,
    ) -> torch.Tensor:
        if expert_context is None:
            return t_code
        if self.expert_t_proj is None or self.expert_t_scale is None:
            return t_code
        delta = self.expert_t_proj[idx](expert_context[:, idx])
        scale = self.expert_t_scale[idx].to(device=t_code.device, dtype=t_code.dtype)
        return t_code + scale * delta


def _init_small_head(head: nn.Module) -> None:
    if hasattr(head, "weight") and head.weight is not None:
        nn.init.normal_(head.weight, mean=0.0, std=1e-3)
    if hasattr(head, "bias") and head.bias is not None:
        nn.init.zeros_(head.bias)


def _adaptive_frequency_bands(
    freq_len: int,
    device: torch.device,
    *,
    dtype: torch.dtype,
    center_logits: torch.Tensor,
    log_widths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if freq_len <= 0:
        raise ValueError("freq_len must be positive")
    grid = torch.linspace(0.0, 1.0, freq_len, device=device, dtype=dtype)
    centers = torch.sigmoid(center_logits).to(device=device, dtype=dtype)
    widths = torch.exp(log_widths).to(device=device, dtype=dtype).clamp_min(1e-3)
    logits = -((grid[None, :] - centers[:, None]) / widths[:, None]).square()
    bands = torch.softmax(logits, dim=0)
    return bands[0], bands[1], bands[2]


def _entropy(prob: torch.Tensor, *, dim: int, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)
