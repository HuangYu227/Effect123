"""Latent Regime Condition Adapter for text-conditioned Flow Matching.

This module is designed for the V6.1 upgrade of a text-to-time-series
Flow-Matching model.  It learns a small set of latent distribution/regime
prototypes and infers a soft regime posterior from the current flow state
``x_t``, flow time ``t``, and the text context.

Key principles:
  * no handcrafted trend/frequency/channel meta labels;
  * one optional structural auxiliary loss: prototype orthogonality;
  * outputs enhanced text context/tokens for the existing velocity network;
  * diagnostics are exposed without forcing additional training objectives.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def regime_orthogonal_loss(regime_bank: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Encourage latent regime prototypes to remain diverse.

    This is intentionally a single compact structural regularizer, not a set of
    meta-information losses.  It can be disabled by setting its weight to zero.

    Returns a differentiable tensor.  Call ``.detach()`` if you only need the
    scalar value (e.g. during inference or logging).
    """
    if regime_bank.ndim != 2:
        raise ValueError(f"regime_bank must be [K, D], got {tuple(regime_bank.shape)}")
    if regime_bank.shape[0] <= 1:
        return regime_bank.new_zeros(())
    bank = F.normalize(regime_bank, dim=-1, eps=eps)
    sim = bank @ bank.t()
    eye = torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    return sim.masked_select(~eye).square().mean()


class LatentRegimeConditionAdapter(nn.Module):
    """Infer latent regimes and enhance text conditions for velocity prediction.

    Args:
        d_model: Dimensionality of the text/model hidden space.
        num_channels: Number of time-series variables/channels in ``x_t``.
        num_regimes: Number of learnable latent regime prototypes.
        regime_dim: Prototype dimension. Defaults to ``d_model``.
        hidden_dim: Internal hidden dimension. Defaults to ``d_model``.
        temperature: Softmax temperature for regime posterior.
        posterior_mode: ``learned`` uses the inferred posterior; ``uniform``
            forces a constant posterior for mechanism ablation.
        append_regime_token: If True, append one regime token to slot tokens.
        state_weight_mode: How much the state branch should affect the regime
            posterior. ``linear_t`` uses the flow time t, so early noisy states
            have weaker influence; ``learned`` learns a scalar gate from t;
            ``constant`` always uses state; ``none`` ignores state.
    """

    def __init__(
        self,
        d_model: int,
        num_channels: int,
        *,
        num_regimes: int = 4,
        regime_dim: int | None = None,
        hidden_dim: int | None = None,
        temperature: float = 0.7,
        posterior_mode: str = "learned",
        append_regime_token: bool = True,
        state_weight_mode: str = "linear_t",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_channels <= 0:
            raise ValueError("d_model and num_channels must be positive")
        if num_regimes <= 0:
            raise ValueError("num_regimes must be positive")
        self.d_model = int(d_model)
        self.num_channels = int(num_channels)
        self.num_regimes = int(num_regimes)
        self.regime_dim = int(regime_dim or d_model)
        self.hidden_dim = int(hidden_dim or d_model)
        self.temperature = float(temperature)
        self.posterior_mode = str(posterior_mode).lower()
        if self.posterior_mode not in {"learned", "uniform"}:
            raise ValueError("posterior_mode must be one of {'learned','uniform'}")
        self.append_regime_token = bool(append_regime_token)
        self.state_weight_mode = str(state_weight_mode).lower()
        if self.state_weight_mode not in {"linear_t", "learned", "constant", "none"}:
            raise ValueError("state_weight_mode must be one of {'linear_t','learned','constant','none'}")

        self.regime_bank = nn.Parameter(torch.empty(self.num_regimes, self.regime_dim))
        nn.init.orthogonal_(self.regime_bank)
        with torch.no_grad():
            self.regime_bank.copy_(F.normalize(self.regime_bank, dim=-1))

        self.state_encoder = nn.Sequential(
            nn.Conv1d(self.num_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # 3 temporal stats (mean/std/max) + 3 spectral stats (low/mid/high energy)
        self.state_stats_proj = nn.Sequential(
            nn.LayerNorm(6 * self.hidden_dim),
            nn.Linear(6 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.query_proj = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.regime_dim),
        )
        self.regime_to_text = nn.Sequential(
            nn.Linear(self.regime_dim, self.d_model),
            nn.LayerNorm(self.d_model),
        )
        self.context_norm = nn.LayerNorm(self.d_model)
        self.token_norm = nn.LayerNorm(self.d_model)
        self.context_gate = nn.Sequential(
            nn.LayerNorm(self.d_model + self.hidden_dim),
            nn.Linear(self.d_model + self.hidden_dim, self.d_model),
            nn.Sigmoid(),
        )
        # Small initial scale keeps the adapter stable while still trainable.
        self.env_scale_logit = nn.Parameter(torch.tensor(-2.0))
        if self.state_weight_mode == "learned":
            self.state_gate = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        else:
            self.state_gate = None

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        slot_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
        text_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_inputs(x_t, t, slot_tokens, slot_mask, text_context)
        dtype = slot_tokens.dtype
        device = slot_tokens.device
        x_t = x_t.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype).view(-1)
        text_context = text_context.to(device=device, dtype=dtype)
        slot_mask = slot_mask.to(device=device, dtype=dtype)

        state_feat = self._encode_state(x_t)  # [B, H]
        text_feat = self.text_proj(text_context)  # [B, H]
        time_feat = self.time_proj(t[:, None])  # [B, H]
        state_weight = self._state_weight(t, dtype=dtype, device=device)  # [B, 1]
        query_in = torch.cat([state_feat * state_weight, text_feat, time_feat], dim=-1)
        query = F.normalize(self.query_proj(query_in), dim=-1)

        bank_norm = F.normalize(self.regime_bank.to(device=device, dtype=dtype), dim=-1)
        scores = query @ bank_norm.t()
        scores = scores / max(self.temperature, 1e-4)
        if self.posterior_mode == "uniform":
            regime_prob = torch.full_like(scores, 1.0 / float(self.num_regimes))
        else:
            regime_prob = torch.softmax(scores, dim=-1)
        regime_context_raw = regime_prob @ bank_norm  # [B, regime_dim]
        regime_token = self.regime_to_text(regime_context_raw)

        gate = self.context_gate(torch.cat([text_context, text_feat], dim=-1))
        scale = torch.sigmoid(self.env_scale_logit).to(dtype=dtype, device=device)
        enhanced_context = self.context_norm(text_context + scale * gate * regime_token)

        if self.append_regime_token:
            token = self.token_norm(regime_token).unsqueeze(1)
            slot_tokens_out = torch.cat([slot_tokens, token], dim=1)
            token_mask = torch.ones(slot_tokens.shape[0], 1, dtype=slot_mask.dtype, device=device)
            slot_mask_out = torch.cat([slot_mask, token_mask], dim=1)
        else:
            slot_tokens_out = slot_tokens
            slot_mask_out = slot_mask

        entropy = _entropy(regime_prob, dim=-1).mean()
        ortho = regime_orthogonal_loss(self.regime_bank) if self.training else regime_prob.new_zeros(())
        usage = regime_prob.mean(dim=0)
        aux = {
            "regime_prob": regime_prob,
            "regime_scores": scores,
            "regime_posterior_uniform": torch.as_tensor(
                float(self.posterior_mode == "uniform"),
                device=device,
                dtype=dtype,
            ).detach(),
            "regime_entropy": entropy.detach(),
            "regime_entropy_norm": (entropy / torch.log(torch.tensor(float(max(self.num_regimes, 2)), device=device, dtype=dtype))).detach(),
            "regime_max_prob": regime_prob.max(dim=-1).values.mean().detach(),
            "regime_usage": usage.detach(),
            "regime_state_weight": state_weight.mean().detach(),
            "regime_context_norm": regime_token.norm(dim=-1).mean().detach(),
            "regime_ortho_loss": ortho,
        }
        return slot_tokens_out, slot_mask_out, enhanced_context, aux

    def _encode_state(self, x_t: torch.Tensor) -> torch.Tensor:
        # x_t: [B, L, C] -> [B, C, L]
        x = x_t.transpose(1, 2).contiguous()
        h = self.state_encoder(x)  # [B, H, L']
        # Temporal statistics
        temporal = torch.cat(
            [h.mean(dim=-1), h.std(dim=-1, unbiased=False), h.amax(dim=-1)],
            dim=-1,
        )  # [B, 3*H]
        # Spectral statistics: low/mid/high energy ratio from Conv features
        fft_dtype = torch.float32 if h.dtype in {torch.float16, torch.bfloat16} else h.dtype
        freq = torch.fft.rfft(h.to(fft_dtype), dim=-1)  # [B, H, F]
        power = freq.abs().square()  # [B, H, F]
        freq_len = power.shape[-1]
        low_end = max(1, freq_len // 3)
        mid_end = max(low_end + 1, 2 * freq_len // 3)
        total = power.sum(dim=-1).clamp_min(1e-8)  # [B, H]
        low = power[..., :low_end].sum(dim=-1) / total  # [B, H]
        mid = power[..., low_end:mid_end].sum(dim=-1) / total if mid_end > low_end else torch.zeros_like(low)
        high = power[..., mid_end:].sum(dim=-1) / total if mid_end < freq_len else torch.zeros_like(low)
        spectral = torch.cat([low.to(h.dtype), mid.to(h.dtype), high.to(h.dtype)], dim=-1)  # [B, 3*H]
        stats = torch.cat([temporal, spectral], dim=-1)  # [B, 6*H]
        return self.state_stats_proj(stats)

    def _state_weight(self, t: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.state_weight_mode == "none":
            return torch.zeros(t.shape[0], 1, dtype=dtype, device=device)
        if self.state_weight_mode == "constant":
            return torch.ones(t.shape[0], 1, dtype=dtype, device=device)
        if self.state_weight_mode == "linear_t":
            return t.clamp(0.0, 1.0).detach().view(-1, 1)
        assert self.state_gate is not None
        return self.state_gate(t.view(-1, 1))

    @staticmethod
    def _validate_inputs(
        x_t: torch.Tensor,
        t: torch.Tensor,
        slot_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
        text_context: torch.Tensor,
    ) -> None:
        if not torch.is_tensor(x_t) or x_t.ndim != 3:
            raise ValueError(f"x_t must be [B, L, C], got {type(x_t)} {getattr(x_t, 'shape', None)}")
        if not torch.is_tensor(t) or t.ndim not in {1, 2}:
            raise ValueError(f"t must be [B] or [B,1], got {type(t)} {getattr(t, 'shape', None)}")
        if not torch.is_tensor(slot_tokens) or slot_tokens.ndim != 3:
            raise ValueError(f"slot_tokens must be [B, J, D], got {getattr(slot_tokens, 'shape', None)}")
        if not torch.is_tensor(slot_mask) or slot_mask.ndim != 2:
            raise ValueError(f"slot_mask must be [B, J], got {getattr(slot_mask, 'shape', None)}")
        if not torch.is_tensor(text_context) or text_context.ndim != 2:
            raise ValueError(f"text_context must be [B, D], got {getattr(text_context, 'shape', None)}")
        b = x_t.shape[0]
        if t.shape[0] != b or slot_tokens.shape[0] != b or slot_mask.shape[0] != b or text_context.shape[0] != b:
            raise ValueError("batch dimensions of x_t/t/slot_tokens/slot_mask/text_context must match")
        if slot_tokens.shape[:2] != slot_mask.shape:
            raise ValueError(f"slot_tokens[:2] {tuple(slot_tokens.shape[:2])} must match slot_mask {tuple(slot_mask.shape)}")


def _entropy(prob: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)
