from __future__ import annotations

import torch
from torch import nn


class EffectMapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_operators: int,
        *,
        field_rank: int = 4,
        slot_layers: int = 1,
        slot_heads: int = 4,
        dropout: float = 0.0,
        tau_t: float = 1.0,
        tau_c: float = 1.0,
        tau_o: float = 1.0,
        normalizer: str = "softmax",
        bounded_field_gate: bool = True,
        operator_router: str = "text",
        series_stats_dim: int = 7,
        series_router_hidden: int | None = None,
        router_epsilon: float = 1e-4,
        time_segment_scales: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        super().__init__()
        self.field_rank = int(field_rank)
        if self.field_rank <= 0:
            raise ValueError("field_rank must be positive")
        if d_model % slot_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by slot_heads={slot_heads}")
        slot_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=slot_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.slot_mixer = nn.TransformerEncoder(slot_layer, num_layers=max(1, int(slot_layers)))
        self.slot_norm = nn.LayerNorm(d_model)
        self.w_t = nn.Linear(d_model, self.field_rank * d_model)
        self.w_c = nn.Linear(d_model, self.field_rank * d_model)
        self.w_o = nn.Linear(d_model, self.field_rank * d_model)
        self.op_proto = nn.Parameter(torch.randn(num_operators, d_model) * 0.02)
        self.alpha = nn.Linear(d_model, self.field_rank)
        self.field_log_amplitude = nn.Parameter(torch.zeros(()))
        self.operator_router = str(operator_router).lower()
        if self.operator_router not in {"text", "dual"}:
            raise ValueError(f"operator_router must be 'text' or 'dual', got {operator_router!r}")
        router_hidden = int(series_router_hidden or d_model)
        self.series_router = None
        if self.operator_router == "dual":
            self.series_router = nn.Sequential(
                nn.Linear(int(series_stats_dim) + 1, router_hidden),
                nn.SiLU(),
                nn.Linear(router_hidden, num_operators),
            )
            nn.init.normal_(self.series_router[-1].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.series_router[-1].bias)
        self.router_epsilon = float(router_epsilon)
        if self.router_epsilon <= 0.0:
            raise ValueError("router_epsilon must be positive")
        if time_segment_scales is None:
            time_segment_scales = (1,)
        self.time_segment_scales = tuple(sorted({max(1, int(k)) for k in time_segment_scales}))
        self.time_resolution = None
        if len(self.time_segment_scales) > 1:
            self.time_resolution = nn.Linear(d_model, len(self.time_segment_scales))
        self.tau_t = float(tau_t)
        self.tau_c = float(tau_c)
        self.tau_o = float(tau_o)
        self.normalizer = normalizer
        self.bounded_field_gate = bool(bounded_field_gate)
        self.scale = d_model**-0.5

    def forward(
        self,
        slot_tokens: torch.Tensor,
        time_tokens: torch.Tensor,
        channel_tokens: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
        series_stats: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, slots, d_model = slot_tokens.shape
        padding_mask = None
        if slot_mask is not None:
            slot_mask = slot_mask.to(slot_tokens.device, dtype=slot_tokens.dtype)
            if slot_mask.shape != slot_tokens.shape[:2]:
                raise ValueError(f"slot_mask must have shape {tuple(slot_tokens.shape[:2])}, got {tuple(slot_mask.shape)}")
            if bool((slot_mask > 0).any(dim=1).all().detach().cpu()):
                padding_mask = slot_mask <= 0
        mixed_slots = self.slot_mixer(slot_tokens, src_key_padding_mask=padding_mask)
        slot_tokens = self.slot_norm(slot_tokens + mixed_slots)

        q_t = self.w_t(slot_tokens).view(batch, slots, self.field_rank, d_model)
        q_c = self.w_c(slot_tokens).view(batch, slots, self.field_rank, d_model)
        q_o = self.w_o(slot_tokens).view(batch, slots, self.field_rank, d_model)
        score_t = torch.einsum("bjrd,bpd->bjrp", q_t, time_tokens) * self.scale
        score_c = torch.einsum("bjrd,bcd->bjrc", q_c, channel_tokens) * self.scale
        score_o = torch.einsum("bjrd,kd->bjrk", q_o, self.op_proto) * self.scale
        score_t = self._multi_resolution_time_logits(score_t, slot_tokens)

        a_t = _normalize(score_t / self.tau_t, dim=-1, normalizer=self.normalizer)
        a_c = _normalize(score_c / self.tau_c, dim=-1, normalizer=self.normalizer)
        a_o_text = _normalize(score_o / self.tau_o, dim=-1, normalizer=self.normalizer)
        a_o_series = None
        if self.operator_router == "dual":
            if series_stats is None or flow_time is None:
                raise ValueError("dual operator router requires series_stats and flow_time")
            if self.series_router is None:
                raise RuntimeError("series router is not initialized")
            if series_stats.shape[0] != batch:
                raise ValueError(f"series_stats batch must be {batch}, got {series_stats.shape[0]}")
            if series_stats.shape[-1] != self.series_router[0].in_features - 1:
                raise ValueError(
                    f"series_stats last dim must be {self.series_router[0].in_features - 1}, got {series_stats.shape[-1]}"
                )
            if flow_time.shape != (batch,):
                raise ValueError(f"flow_time must have shape [{batch}], got {tuple(flow_time.shape)}")
            series_input = torch.cat(
                [
                    series_stats.to(device=slot_tokens.device, dtype=slot_tokens.dtype),
                    flow_time[:, None].to(device=slot_tokens.device, dtype=slot_tokens.dtype),
                ],
                dim=-1,
            )
            series_logits = self.series_router(series_input).to(score_o.dtype)
            a_o_series = _normalize(series_logits / self.tau_o, dim=-1, normalizer=self.normalizer)
            fused_log = (a_o_text.clamp_min(self.router_epsilon).log() +
                         a_o_series[:, None, None, :].clamp_min(self.router_epsilon).log())
            a_o = torch.softmax(fused_log, dim=-1)
        else:
            a_o = a_o_text
        alpha_logits = self.alpha(slot_tokens)
        if self.bounded_field_gate:
            alpha = self._bounded_alpha(alpha_logits, slot_mask)
            field_amplitude = 2.0 * torch.sigmoid(self.field_log_amplitude)
            alpha = alpha * field_amplitude
        else:
            alpha = torch.nn.functional.softplus(alpha_logits)
            if slot_mask is not None:
                alpha = alpha * slot_mask[:, :, None]
            field_amplitude = alpha.new_tensor(float("nan"))

        g_patch = torch.einsum("bjrp,bjrc,bjrk,bjr->bpck", a_t, a_c, a_o, alpha)
        rank_weight = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        aux = {
            "A_t": torch.einsum("bjrp,bjr->bjp", a_t, rank_weight),
            "A_c": torch.einsum("bjrc,bjr->bjc", a_c, rank_weight),
            "A_o": torch.einsum("bjrk,bjr->bjk", a_o, rank_weight),
            "A_o_text": torch.einsum("bjrk,bjr->bjk", a_o_text, rank_weight),
            "A_t_rank": a_t,
            "A_c_rank": a_c,
            "A_o_rank": a_o,
            "A_o_text_rank": a_o_text,
            "alpha": alpha,
            "field_amplitude": field_amplitude.detach(),
            "field_mass": alpha.sum(dim=(1, 2)).detach(),
        }
        if a_o_series is not None:
            aux["A_o_series"] = a_o_series
        return g_patch, aux

    def _bounded_alpha(self, alpha_logits: torch.Tensor, slot_mask: torch.Tensor | None) -> torch.Tensor:
        batch, slots, _ = alpha_logits.shape
        if slot_mask is None:
            return torch.softmax(alpha_logits.flatten(1), dim=-1).view(batch, slots, self.field_rank)
        alpha_mask = slot_mask[:, :, None].expand_as(alpha_logits) > 0
        masked_logits = alpha_logits.masked_fill(~alpha_mask, torch.finfo(alpha_logits.dtype).min)
        valid = alpha_mask.flatten(1).any(dim=1)
        alpha = torch.zeros_like(alpha_logits)
        if bool(valid.any().detach().cpu()):
            valid_alpha = torch.softmax(masked_logits[valid].flatten(1), dim=-1).view(-1, slots, self.field_rank)
            alpha[valid] = valid_alpha
        return alpha

    def _multi_resolution_time_logits(self, score_t: torch.Tensor, slot_tokens: torch.Tensor) -> torch.Tensor:
        if len(self.time_segment_scales) == 1:
            return score_t
        batch, slots, rank, patches = score_t.shape
        flat = score_t.reshape(batch * slots * rank, 1, patches)
        smoothed = []
        for kernel in self.time_segment_scales:
            if kernel <= 1:
                smoothed.append(flat)
                continue
            pad_left = (kernel - 1) // 2
            pad_right = kernel - 1 - pad_left
            padded = torch.nn.functional.pad(flat, (pad_left, pad_right), mode="replicate")
            smoothed.append(torch.nn.functional.avg_pool1d(padded, kernel_size=kernel, stride=1))
        stacked = torch.stack(smoothed, dim=-1).reshape(batch, slots, rank, patches, -1)
        if self.time_resolution is None:
            raise RuntimeError("time_resolution was not initialized")
        weights = torch.softmax(self.time_resolution(slot_tokens), dim=-1)
        weights = weights[:, :, None, None, :]
        return (stacked * weights).sum(dim=-1)


def _normalize(scores: torch.Tensor, *, dim: int, normalizer: str) -> torch.Tensor:
    if normalizer == "softmax":
        return torch.softmax(scores, dim=dim)
    if normalizer in {"entmax", "entmax15"}:
        try:
            from entmax import entmax15
        except ImportError as exc:
            raise RuntimeError("entmax normalizer requested but entmax is not installed") from exc
        return entmax15(scores, dim=dim)
    raise ValueError(f"Unknown normalizer {normalizer!r}")
