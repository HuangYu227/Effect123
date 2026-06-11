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
        gate_rescale: str | float = "auto",
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
        self.tau_t = float(tau_t)
        self.tau_c = float(tau_c)
        self.tau_o = float(tau_o)
        self.normalizer = normalizer
        self.bounded_field_gate = bool(bounded_field_gate)
        self.gate_rescale = gate_rescale
        self.scale = d_model**-0.5

    def forward(
        self,
        slot_tokens: torch.Tensor,
        time_tokens: torch.Tensor,
        channel_tokens: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
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

        a_t = _normalize(score_t / self.tau_t, dim=-1, normalizer=self.normalizer)
        a_c = torch.softmax(score_c / self.tau_c, dim=-1)
        a_o = torch.softmax(score_o / self.tau_o, dim=-1)
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

        g_patch_raw = torch.einsum("bjrp,bjrc,bjrk,bjr->bpck", a_t, a_c, a_o, alpha)
        gate_scale = self._gate_rescale_factor(g_patch_raw.shape[1], g_patch_raw.shape[2])
        g_patch = g_patch_raw * gate_scale
        rank_weight = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        aux = {
            "A_t": torch.einsum("bjrp,bjr->bjp", a_t, rank_weight),
            "A_c": torch.einsum("bjrc,bjr->bjc", a_c, rank_weight),
            "A_o": torch.einsum("bjrk,bjr->bjk", a_o, rank_weight),
            "A_t_rank": a_t,
            "A_c_rank": a_c,
            "A_o_rank": a_o,
            "alpha": alpha,
            "field_amplitude": field_amplitude.detach(),
            "field_mass": alpha.sum(dim=(1, 2)).detach(),
            "gate_rescale_factor": torch.tensor(float(gate_scale), device=g_patch.device),
            "gate_raw_cell_mass_mean": g_patch_raw.detach().abs().mean(),
            "gate_cell_mass_mean": g_patch.detach().abs().mean(),
        }
        return g_patch, aux

    def _gate_rescale_factor(self, patches: int, channels: int) -> float:
        if isinstance(self.gate_rescale, (int, float)):
            return float(self.gate_rescale)
        if self.gate_rescale == "none":
            return 1.0
        if self.gate_rescale == "sqrt":
            return float((patches * channels) ** 0.5)
        if self.gate_rescale == "auto":
            return float(patches * channels)
        raise ValueError(f"Unknown gate_rescale mode: {self.gate_rescale!r}")

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
