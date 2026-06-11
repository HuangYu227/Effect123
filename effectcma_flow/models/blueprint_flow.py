from __future__ import annotations

from typing import Any

import torch
from torch import nn


COMPONENT_NAMES = ("trend", "seasonal", "channel", "residual")


class BlueprintTextToTSFlow(nn.Module):
    """Text-to-series flow with a text-derived structural source distribution.

    The model keeps the public TextToTSFlow interface: forward receives only
    the current ODE state, flow time, and text condition. For training/sampling,
    `initial_state` replaces pure Gaussian noise with
    `text_blueprint + structured_noise`; the blueprint is predicted from text,
    not read from a real time series.
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        num_channels: int,
        d_model: int,
        text_encoder: nn.Module,
        blueprint_trend_degree: int = 3,
        blueprint_num_frequencies: int = 8,
        blueprint_rank: int = 4,
        flow_hidden: int = 96,
        flow_levels: int = 3,
        flow_blocks_per_level: int = 1,
        flow_t_dim: int = 32,
        flow_dropout: float = 0.0,
        flow_max_velocity: float = 5.0,
        flow_channel_heads: int = 4,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.d_model = int(d_model)
        self.component_names = COMPONENT_NAMES
        self.text_encoder = text_encoder
        self.component_router = nn.Linear(self.d_model, len(COMPONENT_NAMES))
        self.blueprint_prior = TextBlueprintPrior(
            sequence_length=self.sequence_length,
            num_channels=self.num_channels,
            text_dim=self.d_model,
            trend_degree=blueprint_trend_degree,
            num_frequencies=blueprint_num_frequencies,
            rank=blueprint_rank,
        )
        self.flow = ComponentResidualUNet(
            sequence_length=self.sequence_length,
            num_channels=self.num_channels,
            hidden=flow_hidden,
            text_dim=self.d_model,
            t_dim=flow_t_dim,
            levels=flow_levels,
            blocks_per_level=flow_blocks_per_level,
            dropout=flow_dropout,
            max_velocity=flow_max_velocity,
            channel_heads=flow_channel_heads,
        )

    def prepare_condition(self, text_condition: Any, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        if isinstance(text_condition, dict) and text_condition.get("_blueprint_prepared", False):
            return _move_prepared(text_condition, device=device, dtype=dtype)
        slot_tokens, slot_mask = self.text_encoder(text_condition)
        slot_tokens = slot_tokens.to(device=device, dtype=dtype)
        slot_mask = slot_mask.to(device=device, dtype=dtype) if slot_mask is not None else None
        text_context = _masked_mean(slot_tokens, slot_mask)
        component_weights = torch.softmax(self.component_router(text_context), dim=-1)
        return {
            "_blueprint_prepared": True,
            "slot_tokens": slot_tokens,
            "slot_mask": slot_mask if slot_mask is not None else torch.ones(slot_tokens.shape[:2], device=device, dtype=dtype),
            "text_context": text_context,
            "component_weights": component_weights,
        }

    def initial_state(
        self,
        shape_like: torch.Tensor,
        text_condition: Any,
        *,
        noise_scale: float = 1.0,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if shape_like.ndim != 3:
            raise ValueError(f"shape_like must be [B, L, C], got {tuple(shape_like.shape)}")
        if shape_like.shape[1:] != (self.sequence_length, self.num_channels):
            raise ValueError(
                f"Expected shape_like [B, {self.sequence_length}, {self.num_channels}], got {tuple(shape_like.shape)}"
            )
        prepared = self.prepare_condition(text_condition, device=shape_like.device, dtype=shape_like.dtype)
        blueprint, aux = self.blueprint_prior(prepared["text_context"])
        if noise is None:
            base_noise = torch.randn(
                shape_like.shape,
                device=shape_like.device,
                dtype=shape_like.dtype,
                generator=generator,
            )
        else:
            base_noise = noise.to(device=shape_like.device, dtype=shape_like.dtype)
        if base_noise.shape != shape_like.shape:
            raise ValueError(f"noise must have shape {tuple(shape_like.shape)}, got {tuple(base_noise.shape)}")
        structured = self.blueprint_prior.structure_noise(base_noise, prepared["text_context"])
        scale = aux["residual_scale"]
        return blueprint + float(noise_scale) * scale * structured

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, text_condition: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B, L, C], got {tuple(x_t.shape)}")
        if x_t.shape[1:] != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected x_t [B, {self.sequence_length}, {self.num_channels}], got {tuple(x_t.shape)}")
        if t.shape != (x_t.shape[0],):
            raise ValueError(f"t must have shape [batch], got {tuple(t.shape)}")
        if t.device != x_t.device:
            raise ValueError(f"t and x_t must be on the same device, got {t.device} and {x_t.device}")
        prepared = self.prepare_condition(text_condition, device=x_t.device, dtype=x_t.dtype)
        blueprint, blueprint_aux = self.blueprint_prior(prepared["text_context"])
        v_hat, flow_aux = self.flow(
            x_t=x_t,
            t=t,
            blueprint=blueprint,
            components=blueprint_aux["components"],
            text_context=prepared["text_context"],
            component_weights=prepared["component_weights"],
        )
        aux = {
            "A_o": prepared["component_weights"][:, None, :],
            "A_o_text": prepared["component_weights"][:, None, :],
            "blueprint": blueprint,
            "blueprint_trend": blueprint_aux["components"]["trend"],
            "blueprint_seasonal": blueprint_aux["components"]["seasonal"],
            "blueprint_channel": blueprint_aux["components"]["channel"],
            "residual_scale": blueprint_aux["residual_scale"],
            "noise_band_weights": blueprint_aux["noise_band_weights"],
            "text_context": prepared["text_context"],
            "component_names": COMPONENT_NAMES,
            **flow_aux,
        }
        return v_hat, aux


class TextBlueprintPrior(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        num_channels: int,
        text_dim: int,
        trend_degree: int,
        num_frequencies: int,
        rank: int,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.trend_degree = max(0, int(trend_degree))
        self.num_frequencies = max(1, int(num_frequencies))
        self.rank = max(1, int(rank))
        time = torch.linspace(-1.0, 1.0, self.sequence_length)
        trend_basis = torch.stack([time.pow(i) for i in range(self.trend_degree + 1)], dim=0)
        freqs = torch.arange(1, self.num_frequencies + 1, dtype=time.dtype)[:, None]
        phase = torch.pi * (time[None, :] + 1.0) * freqs
        fourier_basis = torch.cat([torch.sin(phase), torch.cos(phase)], dim=0)
        basis = torch.cat([trend_basis, fourier_basis], dim=0)
        self.register_buffer("trend_basis", trend_basis)
        self.register_buffer("fourier_basis", fourier_basis)
        self.register_buffer("combined_basis", basis)
        trend_dim = self.trend_degree + 1
        seasonal_dim = 2 * self.num_frequencies
        basis_dim = trend_dim + seasonal_dim
        self.trend_head = _projection_head(text_dim, self.num_channels * trend_dim)
        self.seasonal_head = _projection_head(text_dim, self.num_channels * seasonal_dim)
        self.lowrank_basis_head = _projection_head(text_dim, self.rank * basis_dim)
        self.lowrank_channel_head = _projection_head(text_dim, self.num_channels * self.rank)
        self.amplitude_head = _projection_head(text_dim, self.num_channels)
        self.residual_scale_head = _projection_head(text_dim, self.num_channels)
        self.noise_band_head = _projection_head(text_dim, self.num_channels * 3)

    def forward(self, text_context: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if text_context.ndim != 2:
            raise ValueError(f"text_context must be [B, D], got {tuple(text_context.shape)}")
        batch = text_context.shape[0]
        dtype = text_context.dtype
        trend_basis = self.trend_basis.to(device=text_context.device, dtype=dtype)
        fourier_basis = self.fourier_basis.to(device=text_context.device, dtype=dtype)
        combined_basis = self.combined_basis.to(device=text_context.device, dtype=dtype)
        trend_coeff = self.trend_head(text_context).view(batch, self.num_channels, -1)
        seasonal_coeff = self.seasonal_head(text_context).view(batch, self.num_channels, -1)
        trend = torch.einsum("bcd,dl->blc", trend_coeff, trend_basis)
        seasonal = torch.einsum("bcf,fl->blc", seasonal_coeff, fourier_basis) / (self.num_frequencies ** 0.5)
        latent_coeff = self.lowrank_basis_head(text_context).view(batch, self.rank, -1)
        latent_time = torch.einsum("brd,dl->blr", latent_coeff, combined_basis) / (combined_basis.shape[0] ** 0.5)
        channel_load = torch.tanh(self.lowrank_channel_head(text_context).view(batch, self.num_channels, self.rank))
        channel = torch.einsum("blr,bcr->blc", latent_time, channel_load) / (self.rank ** 0.5)
        amplitude = torch.nn.functional.softplus(self.amplitude_head(text_context)).view(batch, 1, self.num_channels) + 0.05
        residual_scale = torch.nn.functional.softplus(self.residual_scale_head(text_context)).view(batch, 1, self.num_channels) + 0.05
        noise_band_weights = torch.softmax(self.noise_band_head(text_context).view(batch, self.num_channels, 3), dim=-1)
        blueprint = amplitude * (trend + seasonal + channel)
        return blueprint, {
            "components": {
                "trend": amplitude * trend,
                "seasonal": amplitude * seasonal,
                "channel": amplitude * channel,
            },
            "residual_scale": residual_scale,
            "noise_band_weights": noise_band_weights,
        }

    def structure_noise(self, noise: torch.Tensor, text_context: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 3:
            raise ValueError(f"noise must be [B, L, C], got {tuple(noise.shape)}")
        if noise.shape[1:] != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected noise [B, {self.sequence_length}, {self.num_channels}], got {tuple(noise.shape)}")
        _, aux = self.forward(text_context)
        band_weights = aux["noise_band_weights"].to(device=noise.device, dtype=noise.dtype)
        x = noise - noise.mean(dim=1, keepdim=True)
        fft_dtype = torch.float32 if x.dtype in {torch.float16, torch.bfloat16} else x.dtype
        freq = torch.fft.rfft(x.transpose(1, 2).to(fft_dtype), dim=-1)
        masks = _fixed_frequency_bands(freq.shape[-1], freq.device, freq.real.dtype)
        pieces = []
        for mask in masks:
            filtered = torch.fft.irfft(freq * mask[None, None, :], n=self.sequence_length, dim=-1)
            pieces.append(filtered.transpose(1, 2).to(noise.dtype))
        stacked = torch.stack(pieces, dim=-1)
        colored = (stacked * band_weights[:, None, :, :]).sum(dim=-1)
        scale = colored.var(dim=1, keepdim=True, unbiased=False).sqrt().clamp_min(1e-4)
        return colored / scale


class ComponentResidualUNet(nn.Module):
    def __init__(
        self,
        *,
        sequence_length: int,
        num_channels: int,
        hidden: int,
        text_dim: int,
        t_dim: int,
        levels: int,
        blocks_per_level: int,
        dropout: float,
        max_velocity: float,
        channel_heads: int,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.hidden = int(hidden)
        self.max_velocity = float(max_velocity)
        self.t_embed = nn.Sequential(nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.in_proj = nn.Linear(7 + t_dim, hidden)
        self.component_proj = nn.Linear(3, hidden)
        self.levels = max(1, int(levels))
        blocks = max(1, int(blocks_per_level))
        self.encoder_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ComponentFlowBlock(
                            hidden=hidden,
                            text_dim=text_dim,
                            t_dim=t_dim,
                            dropout=dropout,
                            channel_heads=channel_heads,
                            dilation=2 ** min(i + j, 4),
                        )
                        for j in range(blocks)
                    ]
                )
                for i in range(self.levels)
            ]
        )
        self.down_proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(max(0, self.levels - 1))])
        self.bottleneck = ComponentFlowBlock(
            hidden=hidden,
            text_dim=text_dim,
            t_dim=t_dim,
            dropout=dropout,
            channel_heads=channel_heads,
            dilation=2 ** min(self.levels, 4),
        )
        self.up_merge = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(max(0, self.levels - 1))])
        self.up_blocks = nn.ModuleList(
            [
                ComponentFlowBlock(
                    hidden=hidden,
                    text_dim=text_dim,
                    t_dim=t_dim,
                    dropout=dropout,
                    channel_heads=channel_heads,
                    dilation=2 ** min(i, 4),
                )
                for i in reversed(range(max(0, self.levels - 1)))
            ]
        )
        self.head = nn.Linear(hidden, 1)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        blueprint: torch.Tensor,
        components: dict[str, torch.Tensor],
        text_context: torch.Tensor,
        component_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, length, channels = x_t.shape
        if (length, channels) != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected x_t [B, {self.sequence_length}, {self.num_channels}], got {tuple(x_t.shape)}")
        pos = torch.linspace(-1.0, 1.0, length, device=x_t.device, dtype=x_t.dtype)
        pos = pos[None, :, None].expand(batch, length, channels)
        t_code = self.t_embed(t[:, None].to(x_t.dtype))
        t_feat = t_code[:, None, None, :].expand(batch, length, channels, -1)
        base_feat = torch.stack(
            [
                x_t,
                blueprint,
                x_t - blueprint,
                components["trend"],
                components["seasonal"],
                components["channel"],
                pos,
            ],
            dim=-1,
        )
        component_map = torch.stack([components["trend"], components["seasonal"], components["channel"]], dim=-1)
        h = self.in_proj(torch.cat([base_feat, t_feat], dim=-1)) + self.component_proj(component_map)
        skips: list[torch.Tensor] = []
        component_pyramid = component_map
        for level, level_blocks in enumerate(self.encoder_blocks):
            for block in level_blocks:
                h = block(
                    h,
                    text_context=text_context,
                    t_code=t_code,
                    component_weights=component_weights,
                    component_map=component_pyramid,
                )
            skips.append(h)
            if level < self.levels - 1:
                h = self.down_proj[level](_pool_time(h))
                component_pyramid = _pool_time(component_pyramid)
        h = self.bottleneck(
            h,
            text_context=text_context,
            t_code=t_code,
            component_weights=component_weights,
            component_map=component_pyramid,
        )
        component_skips: list[torch.Tensor] = []
        component_cursor = component_map
        for _ in range(self.levels):
            component_skips.append(component_cursor)
            if len(component_skips) < self.levels:
                component_cursor = _pool_time(component_cursor)
        for index, skip in enumerate(reversed(skips[:-1])):
            h = _upsample_time(h, skip.shape[1])
            h = self.up_merge[index](torch.cat([h, skip], dim=-1))
            component_up = component_skips[-2 - index]
            h = self.up_blocks[index](
                h,
                text_context=text_context,
                t_code=t_code,
                component_weights=component_weights,
                component_map=component_up,
            )
        v = self.head(h).squeeze(-1)
        return torch.tanh(v) * self.max_velocity, {
            "flow_hidden_rms": h.detach().square().mean().sqrt(),
        }


class ComponentFlowBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        text_dim: int,
        t_dim: int,
        dropout: float,
        channel_heads: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.norm = RoutedAdaLayerNorm(hidden=hidden, text_dim=text_dim, t_dim=t_dim)
        padding = int(dilation)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=padding, dilation=dilation, groups=hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.Dropout(dropout),
        )
        heads = max(1, min(int(channel_heads), hidden))
        while hidden % heads != 0 and heads > 1:
            heads -= 1
        self.channel_attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.channel_ffn = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 2 * hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.frequency_mix = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.branch_gate = nn.Linear(text_dim, 3)
        self.out_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        x: torch.Tensor,
        *,
        text_context: torch.Tensor,
        t_code: torch.Tensor,
        component_weights: torch.Tensor,
        component_map: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, channels, hidden = x.shape
        if component_map.shape[:3] != (batch, length, channels) or component_map.shape[-1] != 3:
            raise ValueError(
                f"component_map must be [B, L, C, 3], got {tuple(component_map.shape)} for hidden {tuple(x.shape)}"
            )
        y = self.norm(x, text_context=text_context, t_code=t_code, component_weights=component_weights)
        temporal_in = y.permute(0, 2, 3, 1).reshape(batch * channels, hidden, length)
        temporal = self.temporal(temporal_in).reshape(batch, channels, hidden, length).permute(0, 3, 1, 2)
        channel_in = y.reshape(batch * length, channels, hidden)
        channel, _ = self.channel_attn(channel_in, channel_in, channel_in, need_weights=False)
        channel = (channel + self.channel_ffn(channel)).reshape(batch, length, channels, hidden)
        frequency = self._frequency_branch(y)
        gate = torch.softmax(self.branch_gate(text_context), dim=-1)
        map_gate = torch.softmax(component_map.abs(), dim=-1)
        effective_gate = torch.softmax(gate[:, None, None, :] + map_gate.clamp_min(1e-6).log(), dim=-1)
        residual = (
            effective_gate[..., 0, None] * temporal
            + effective_gate[..., 1, None] * channel
            + effective_gate[..., 2, None] * frequency
        )
        return self.out_norm(x + residual)

    def _frequency_branch(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels, hidden = x.shape
        series = x.permute(0, 2, 3, 1).reshape(batch * channels, hidden, length)
        fft_dtype = torch.float32 if series.dtype in {torch.float16, torch.bfloat16} else series.dtype
        freq = torch.fft.rfft(series.to(fft_dtype), dim=-1)
        masks = _fixed_frequency_bands(freq.shape[-1], freq.device, freq.real.dtype)
        pieces = []
        for mask in masks:
            filtered = torch.fft.irfft(freq * mask[None, None, :], n=length, dim=-1)
            pieces.append(filtered.to(series.dtype))
        stacked = torch.cat(pieces, dim=1)
        mixed = self.frequency_mix(stacked.transpose(1, 2)).reshape(batch, channels, length, hidden)
        return mixed.permute(0, 2, 1, 3)


class RoutedAdaLayerNorm(nn.Module):
    def __init__(self, *, hidden: int, text_dim: int, t_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.to_component = nn.Linear(text_dim + t_dim, len(COMPONENT_NAMES) * 2 * hidden)
        nn.init.zeros_(self.to_component.weight)
        nn.init.zeros_(self.to_component.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        text_context: torch.Tensor,
        t_code: torch.Tensor,
        component_weights: torch.Tensor,
    ) -> torch.Tensor:
        params = self.to_component(torch.cat([text_context, t_code], dim=-1))
        params = params.view(x.shape[0], len(COMPONENT_NAMES), 2, x.shape[-1])
        scale, shift = params[:, :, 0], params[:, :, 1]
        scale = torch.einsum("bk,bkh->bh", component_weights, scale)
        shift = torch.einsum("bk,bkh->bh", component_weights, shift)
        return self.norm(x) * (1.0 + scale[:, None, None, :]) + shift[:, None, None, :]


def _projection_head(text_dim: int, out_dim: int) -> nn.Sequential:
    head = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, 2 * text_dim), nn.SiLU(), nn.Linear(2 * text_dim, out_dim))
    nn.init.normal_(head[-1].weight, mean=0.0, std=1e-3)
    nn.init.zeros_(head[-1].bias)
    return head


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"slot mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}")
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask[:, :, None]).sum(dim=1) / denom


def _move_prepared(prepared: dict[str, Any], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {"_blueprint_prepared": True}
    for key, value in prepared.items():
        if key == "_blueprint_prepared":
            continue
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value
    return moved


def _pool_time(x: torch.Tensor) -> torch.Tensor:
    batch, length, channels, hidden = x.shape
    series = x.permute(0, 2, 3, 1).reshape(batch * channels, hidden, length)
    pooled = torch.nn.functional.avg_pool1d(series, kernel_size=2, stride=2, ceil_mode=True)
    new_length = pooled.shape[-1]
    return pooled.reshape(batch, channels, hidden, new_length).permute(0, 3, 1, 2)


def _upsample_time(x: torch.Tensor, length: int) -> torch.Tensor:
    batch, _, channels, hidden = x.shape
    series = x.permute(0, 2, 3, 1).reshape(batch * channels, hidden, -1)
    up = torch.nn.functional.interpolate(series, size=int(length), mode="linear", align_corners=False)
    return up.reshape(batch, channels, hidden, int(length)).permute(0, 3, 1, 2)


def _fixed_frequency_bands(freq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(freq_len, device=device)
    low_end = max(1, int(round(freq_len / 3)))
    mid_end = max(low_end + 1, int(round(2 * freq_len / 3)))
    low = (idx < low_end).to(dtype)
    mid = ((idx >= low_end) & (idx < mid_end)).to(dtype)
    high = (idx >= mid_end).to(dtype)
    if bool((high.sum() == 0).detach().cpu()):
        high[-1] = 1.0
    return low, mid, high
