from __future__ import annotations

import torch
from torch import nn

from effectcma_flow.models.cross_modal_bridge import CrossModalConditionBridge
from effectcma_flow.models.effect_mapper import EffectMapper
from effectcma_flow.models.global_operator_gate import GlobalOperatorGate
from effectcma_flow.models.latent_regime_adapter import LatentRegimeConditionAdapter
from effectcma_flow.models.operator_bank import ResidualOperatorBank
from effectcma_flow.models.spectral_prompt import SpectralPromptGenerator, merge_spectral_expert_context
from effectcma_flow.models.ts_encoder import ChannelEncoder, TimePatchEncoder


class TextToTSFlow(nn.Module):
    """Text-conditioned continuous flow from Gaussian noise to time series.

    Unlike `EffectCMAFlow`, this model never receives a real base trajectory as
    condition. The current flow state `x_t` supplies state tokens, while caption
    tokens define a time-channel-operator generation field.

    V6.1 adds two optional modules:

    * **LatentRegimeConditionAdapter** — infers a soft regime posterior from
      ``x_t``, ``t``, and the text context, producing an enhanced text context
      and an optional appended regime token.
    * **GlobalOperatorGate** — replaces the fine-grained A_t/A_c router with a
      sample-level operator gate (set ``router_mode='global_operator'``).
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
        mapper_gate_rescale: str | float = "auto",
        mapper_operator_router: str = "text",
        mapper_time_segment_scales: tuple[int, ...] | list[int] | None = None,
        mapper_router_epsilon: float = 1e-4,
        mapper_flow_time_condition: bool = True,
        operator_context_film: bool = True,
        operator_context_mode: str = "global",
        operator_multiview_context: bool = False,
        operator_norm: str = "group",
        operator_architecture: str = "homogeneous",
        operator_channel_heads: int = 4,
        operator_frequency_band_mode: str = "gaussian",
        operator_frequency_topk_frac: float = 0.15,
        operator_frequency_temp_start: float = 1.0,
        operator_frequency_temp_end: float = 0.1,
        operator_frequency_anneal_steps: int = 10000,
        operator_temporal_long_range_mode: str = "none",
        operator_temporal_long_range_scales: tuple[int, ...] | list[int] | None = None,
        # V6.1 latent regime adapter
        use_latent_regime_adapter: bool = False,
        num_regimes: int = 4,
        regime_dim: int | None = None,
        regime_hidden_dim: int | None = None,
        regime_temperature: float = 0.7,
        regime_posterior_mode: str = "learned",
        regime_append_token: bool = True,
        regime_dropout: float = 0.0,
        regime_state_weight_mode: str = "linear_t",
        # V6.1 routing mode: "legacy" (EffectMapper) or "global_operator"
        router_mode: str = "legacy",
        operator_gate_temperature: float = 1.0,
        operator_gate_dropout: float = 0.0,
        operator_gate_router: str = "mlp",
        operator_gate_heads: int = 4,
        # V6.2 cross-modal condition bridge
        use_cross_modal_bridge: bool = False,
        bridge_num_heads: int = 4,
        bridge_dropout: float = 0.0,
        bridge_patch_size: int | None = None,
        bridge_num_spectral_tokens: int = 3,
        bridge_alignment_temperature: float = 0.07,
        bridge_focal_mode: str = "legacy",
        bridge_num_stage_tokens: int = 3,
        bridge_state_connector: str = "legacy",
        bridge_temporal_merge: int = 4,
        bridge_channel_merge: int = 1,
        bridge_token_budget: int = 96,
        bridge_alignment_mode: str = "auto",
        bridge_alignment_dense: bool = False,
        bridge_alignment_regions: int = 8,
        bridge_alignment_target: str = "state",
        bridge_alignment_clean_prob: float = 1.0,
        bridge_text_agg_tokens: int = 0,
        bridge_diagnostics: bool = False,
        # V6.6-light text-conditioned spectral prompt
        use_spectral_prompt: bool = False,
        spectral_prompt_bands: int = 3,
        spectral_prompt_heads: int = 4,
        spectral_prompt_dropout: float = 0.0,
        spectral_prompt_gate_temperature: float = 1.0,
        spectral_prompt_residual_gate: bool = True,
        # V6.7 TSP-Bridge V2 (temporal semantic pyramid)
        temporal_pyramid_patch_lens: list[int] | tuple[int, ...] | None = None,
        temporal_pyramid_token_budget: int = 128,
        temporal_pyramid_anchor_tokens: int = 8,
        temporal_pyramid_cross_scale_layers: int = 2,
        temporal_pyramid_dropout: float = 0.05,
        temporal_pyramid_use_topdown: bool = True,
        temporal_pyramid_use_bottomup: bool = True,
        temporal_pyramid_use_text_routing: bool = True,
        temporal_pyramid_temporal_bias_tau: float = 0.25,
        temporal_pyramid_gate_temperature: float = 0.7,
        **kwargs,
    ) -> None:
        super().__init__()
        self.sequence_length = int(sequence_length)
        self.num_channels = int(num_channels)
        self.patch_len = int(patch_len)
        self.router_mode = str(router_mode).lower()
        if self.router_mode not in {"legacy", "global_operator"}:
            raise ValueError(f"router_mode must be 'legacy' or 'global_operator', got {router_mode!r}")
        self._use_global_gate = self.router_mode == "global_operator"

        self.text_encoder = text_encoder
        self.mapper_flow_time_condition = bool(mapper_flow_time_condition)
        self.flow_time_proj = None
        self.use_cross_modal_bridge = bool(use_cross_modal_bridge)
        self.cross_modal_bridge = (
            CrossModalConditionBridge(
                d_model=d_model,
                num_channels=self.num_channels,
                sequence_length=self.sequence_length,
                num_experts=num_operators,
                patch_size=int(bridge_patch_size or self.patch_len),
                num_heads=bridge_num_heads,
                dropout=bridge_dropout,
                num_spectral_tokens=bridge_num_spectral_tokens,
                alignment_temperature=bridge_alignment_temperature,
                focal_mode=bridge_focal_mode,
                num_stage_tokens=bridge_num_stage_tokens,
                state_connector=bridge_state_connector,
                temporal_merge=bridge_temporal_merge,
                channel_merge=bridge_channel_merge,
                token_budget=bridge_token_budget,
                alignment_mode=bridge_alignment_mode,
                alignment_dense=bridge_alignment_dense,
                alignment_regions=bridge_alignment_regions,
                alignment_target=bridge_alignment_target,
                alignment_clean_prob=bridge_alignment_clean_prob,
                text_agg_tokens=bridge_text_agg_tokens,
                diagnostics=bridge_diagnostics,
                # TSP-Bridge V2 parameters
                temporal_pyramid_patch_lens=temporal_pyramid_patch_lens or [4, 8, 16, 32],
                temporal_pyramid_token_budget=temporal_pyramid_token_budget,
                temporal_pyramid_anchor_tokens=temporal_pyramid_anchor_tokens,
                temporal_pyramid_cross_scale_layers=temporal_pyramid_cross_scale_layers,
                temporal_pyramid_dropout=temporal_pyramid_dropout,
                temporal_pyramid_use_topdown=temporal_pyramid_use_topdown,
                temporal_pyramid_use_bottomup=temporal_pyramid_use_bottomup,
                temporal_pyramid_use_text_routing=temporal_pyramid_use_text_routing,
                temporal_pyramid_temporal_bias_tau=temporal_pyramid_temporal_bias_tau,
                temporal_pyramid_gate_temperature=temporal_pyramid_gate_temperature,
            )
            if self.use_cross_modal_bridge
            else None
        )
        self.use_spectral_prompt = bool(use_spectral_prompt)
        self.spectral_prompt = (
            SpectralPromptGenerator(
                d_model=d_model,
                num_bands=spectral_prompt_bands,
                num_experts=num_operators,
                num_heads=spectral_prompt_heads,
                dropout=spectral_prompt_dropout,
                gate_temperature=spectral_prompt_gate_temperature,
                use_residual_gate=spectral_prompt_residual_gate,
            )
            if self.use_spectral_prompt
            else None
        )

        if self._use_global_gate:
            # Global operator gate path: legacy encoders/mapper are not needed.
            self.time_encoder = None
            self.channel_encoder = None
            self.mapper = None
        else:
            # Legacy EffectMapper path: create encoders and mapper.
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
            if self.mapper_flow_time_condition:
                self.flow_time_proj = nn.Sequential(
                    nn.Linear(1, d_model),
                    nn.SiLU(),
                    nn.Linear(d_model, d_model),
                )
                nn.init.zeros_(self.flow_time_proj[-1].weight)
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
                gate_rescale=mapper_gate_rescale,
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
            multiview_context=operator_multiview_context,
            norm_type=operator_norm,
            architecture=operator_architecture,
            channel_heads=operator_channel_heads,
            frequency_band_mode=operator_frequency_band_mode,
            frequency_topk_frac=operator_frequency_topk_frac,
            frequency_temp_start=operator_frequency_temp_start,
            frequency_temp_end=operator_frequency_temp_end,
            frequency_anneal_steps=operator_frequency_anneal_steps,
            temporal_long_range_mode=operator_temporal_long_range_mode,
            temporal_long_range_scales=operator_temporal_long_range_scales,
        )

        # V6.1: Latent regime condition adapter (optional)
        # In global_operator mode the appended regime token is never consumed
        # by a downstream mapper, so auto-disable it to avoid dead code.
        effective_append = regime_append_token and not self._use_global_gate
        if use_latent_regime_adapter:
            self.regime_adapter = LatentRegimeConditionAdapter(
                d_model=d_model,
                num_channels=num_channels,
                num_regimes=num_regimes,
                regime_dim=regime_dim,
                hidden_dim=regime_hidden_dim,
                temperature=regime_temperature,
                posterior_mode=regime_posterior_mode,
                append_regime_token=effective_append,
                state_weight_mode=regime_state_weight_mode,
                dropout=regime_dropout,
            )
        else:
            self.regime_adapter = None

        # V6.1: Global operator gate (replaces mapper when active)
        if self._use_global_gate:
            self.global_operator_gate = GlobalOperatorGate(
                d_model=d_model,
                num_channels=num_channels,
                num_operators=num_operators,
                hidden_dim=d_model,
                temperature=operator_gate_temperature,
                dropout=operator_gate_dropout,
                router_type=operator_gate_router,
                num_heads=operator_gate_heads,
            )
        else:
            self.global_operator_gate = None

    def prepare_condition(self, text_condition: list[str] | list[list[str]] | dict[str, torch.Tensor], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        if isinstance(text_condition, dict) and text_condition.get("_text2ts_prepared", False):
            return _move_prepared_condition(text_condition, device=device, dtype=dtype)
        slot_tokens, slot_mask = self.text_encoder(text_condition)
        slot_tokens = slot_tokens.to(device=device, dtype=dtype)
        slot_mask = slot_mask.to(device=device, dtype=dtype) if slot_mask is not None else None
        text_context = _masked_mean(slot_tokens, slot_mask)
        return {
            "_text2ts_prepared": True,
            "slot_tokens": slot_tokens,
            "slot_mask": slot_mask if slot_mask is not None else torch.ones(slot_tokens.shape[:2], device=device, dtype=dtype),
            "text_context": text_context,
        }

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_condition,
        *,
        compute_bridge_alignment: bool = True,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be [B, L, C], got {tuple(x_t.shape)}")
        if x_t.shape[1:] != (self.sequence_length, self.num_channels):
            raise ValueError(f"Expected x_t shape [B, {self.sequence_length}, {self.num_channels}], got {tuple(x_t.shape)}")
        if t.shape != (x_t.shape[0],):
            raise ValueError(f"t must have shape [batch], got {tuple(t.shape)} for batch {x_t.shape[0]}")
        if t.device != x_t.device:
            raise ValueError(f"t and x_t must be on the same device, got {t.device} and {x_t.device}")

        prepared = self.prepare_condition(text_condition, device=x_t.device, dtype=x_t.dtype)
        slot_tokens = prepared["slot_tokens"]
        slot_mask = prepared["slot_mask"]
        if self.flow_time_proj is not None:
            slot_tokens = slot_tokens + self.flow_time_proj(t[:, None].to(x_t.dtype))[:, None, :]
        text_context = _masked_mean(slot_tokens, slot_mask)

        bridge_aux: dict[str, torch.Tensor] = {}
        expert_context: torch.Tensor | None = None
        channel_context: torch.Tensor | None = None
        if self.cross_modal_bridge is not None:
            # slot_tokens: [B,J,D], x_t: [B,L,C],
            # expert_context: [B,K,D], channel_context: [B,C,D].
            slot_tokens, text_context, expert_context, channel_context, bridge_aux = self.cross_modal_bridge(
                slot_tokens=slot_tokens,
                slot_mask=slot_mask,
                x_t=x_t,
                t=t,
                compute_alignment=compute_bridge_alignment,
                target=target,
            )

        # V6.1: optional regime adapter (works with both routing modes)
        regime_aux: dict[str, torch.Tensor] = {}
        if self.regime_adapter is not None:
            slot_tokens, slot_mask, text_context, regime_aux = self.regime_adapter(
                x_t=x_t,
                t=t,
                slot_tokens=slot_tokens,
                slot_mask=slot_mask,
                text_context=text_context,
            )

        spectral_aux: dict[str, torch.Tensor] = {}
        if self.spectral_prompt is not None:
            spectral_out = self.spectral_prompt(slot_tokens, slot_mask, text_context=text_context)
            expert_context = merge_spectral_expert_context(expert_context, spectral_out.expert_delta)
            spectral_aux = spectral_out.aux

        bank_channel_context = channel_context if getattr(self.operator_bank, "multiview_context", False) else None
        if self._use_global_gate:
            # --- V6.1 global operator gate path ---
            velocities = self.operator_bank(
                x_t,
                None,
                t,
                context=text_context,
                expert_context=expert_context,
                channel_context=bank_channel_context,
            )
            _gate, g, gate_aux = self.global_operator_gate(
                x_t=x_t,
                t=t,
                text_context=text_context,
                slot_tokens=slot_tokens,
                slot_mask=slot_mask,
                velocity_shape=velocities.shape,
            )
            v_hat = (g * velocities).sum(dim=-1)
            bank_aux = getattr(self.operator_bank, "last_aux", {})
            aux = {
                **bridge_aux,
                **regime_aux,
                **spectral_aux,
                **bank_aux,
                **gate_aux,
                "G": g,
                "V": velocities,
                "text_context": text_context,
            }
        else:
            # --- Legacy EffectMapper path ---
            time_tokens = self.time_encoder(x_t)
            channel_tokens = self.channel_encoder(x_t)
            series_stats = _series_router_stats(x_t)
            g_patch, aux = self.mapper(
                slot_tokens, time_tokens, channel_tokens, slot_mask,
                series_stats=series_stats, flow_time=t,
            )
            g = g_patch.repeat_interleave(self.patch_len, dim=1)
            if g.shape[1] < self.sequence_length:
                raise RuntimeError(f"Expanded generation field length {g.shape[1]} is shorter than {self.sequence_length}")
            g = g[:, : self.sequence_length]
            velocities = self.operator_bank(
                x_t,
                None,
                t,
                context=text_context,
                expert_context=expert_context,
                channel_context=bank_channel_context,
            )
            v_hat = (g * velocities).sum(dim=-1)
            bank_aux = getattr(self.operator_bank, "last_aux", {})
            aux = {
                **aux,
                **bridge_aux,
                **regime_aux,
                **spectral_aux,
                **bank_aux,
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


def _move_prepared_condition(condition: dict[str, torch.Tensor], *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved = dict(condition)
    for key in ("slot_tokens", "slot_mask", "text_context"):
        if key in moved and torch.is_tensor(moved[key]):
            moved[key] = moved[key].to(device=device, dtype=dtype)
    return moved


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    mask = mask.to(tokens.device, dtype=tokens.dtype)
    if mask.shape != tokens.shape[:2]:
        raise ValueError(f"slot mask must have shape {tuple(tokens.shape[:2])}, got {tuple(mask.shape)}")
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask[:, :, None]).sum(dim=1) / denom


def _series_router_stats(x: torch.Tensor) -> torch.Tensor:
    """Compute 7-dim statistics from current flow state for dual routing."""
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
