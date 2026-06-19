from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpectralPromptOutput:
    """Text-conditioned band prompts used to modulate operator experts."""

    band_tokens: torch.Tensor
    band_gates: torch.Tensor
    expert_delta: torch.Tensor
    aux: dict[str, torch.Tensor]


class SpectralPromptGenerator(nn.Module):
    """Generate lightweight spectral prompts from bridged text tokens.

    The module does not predict a target-specific metadata vector and does not
    introduce a supervised auxiliary head. It uses low/mid/high learnable band
    queries to read the existing text-token memory, then converts those band
    prompts into a residual update for the existing operator expert context.
    The final projection is zero-initialised so enabling the module is a no-op
    at step 0 and can be ablated cleanly.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_bands: int = 3,
        num_experts: int = 3,
        num_heads: int = 4,
        dropout: float = 0.0,
        gate_temperature: float = 1.0,
        use_residual_gate: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_bands <= 0:
            raise ValueError(f"num_bands must be positive, got {num_bands}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        self.d_model = int(d_model)
        self.num_bands = int(num_bands)
        self.num_experts = int(num_experts)
        self.num_heads = _compatible_heads(int(d_model), int(num_heads))
        self.gate_temperature = max(float(gate_temperature), 1e-4)
        self.use_residual_gate = bool(use_residual_gate)

        self.band_queries = nn.Parameter(torch.empty(self.num_bands, self.d_model))
        self.memory_norm = nn.LayerNorm(self.d_model)
        self.query_norm = nn.LayerNorm(self.d_model)
        self.band_attn = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.band_ffn = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model * 4, self.d_model),
        )
        self.context_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.num_bands),
        )
        self.band_gate = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )

        self.expert_queries = nn.Parameter(torch.empty(self.num_experts, self.d_model))
        self.expert_attn = nn.MultiheadAttention(
            self.d_model,
            self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.expert_out = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.band_queries, mean=0.0, std=self.d_model ** -0.5)
        nn.init.normal_(self.expert_queries, mean=0.0, std=self.d_model ** -0.5)
        nn.init.zeros_(self.context_gate[-1].bias)
        nn.init.zeros_(self.band_gate[-1].bias)
        nn.init.zeros_(self.expert_out[-1].weight)
        nn.init.zeros_(self.expert_out[-1].bias)

    def forward(
        self,
        slot_tokens: torch.Tensor,
        slot_mask: torch.Tensor | None = None,
        *,
        text_context: torch.Tensor | None = None,
    ) -> SpectralPromptOutput:
        if slot_tokens.ndim != 3:
            raise ValueError(f"slot_tokens must be [B,J,D], got {tuple(slot_tokens.shape)}")
        if slot_tokens.shape[-1] != self.d_model:
            raise ValueError(f"slot token dim {slot_tokens.shape[-1]} does not match d_model={self.d_model}")
        batch, slots, _ = slot_tokens.shape
        if slots <= 0:
            raise ValueError("slot_tokens must contain at least one token")
        mask = _valid_slot_mask(slot_mask, batch=batch, slots=slots, device=slot_tokens.device)
        memory = self.memory_norm(slot_tokens)
        key_padding_mask = ~mask

        if text_context is None:
            text_context = _masked_mean(memory, mask)
        else:
            if text_context.shape != (batch, self.d_model):
                raise ValueError(f"text_context must be [B,{self.d_model}], got {tuple(text_context.shape)}")
            text_context = text_context.to(device=slot_tokens.device, dtype=slot_tokens.dtype)

        band_queries = self.band_queries.unsqueeze(0).expand(batch, -1, -1)
        band_queries = self.query_norm(band_queries + text_context.unsqueeze(1))
        band_tokens, band_attn = self.band_attn(
            band_queries,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        band_tokens = band_tokens + self.band_ffn(band_tokens)

        gate_logits = self.band_gate(band_tokens).squeeze(-1) + self.context_gate(text_context)
        band_gates = F.softmax(gate_logits / self.gate_temperature, dim=-1)
        gated_bands = band_tokens * band_gates.unsqueeze(-1)

        expert_queries = self.expert_queries.unsqueeze(0).expand(batch, -1, -1)
        expert_tokens, _ = self.expert_attn(expert_queries, gated_bands, gated_bands, need_weights=False)
        expert_delta = self.expert_out(expert_tokens)
        if self.use_residual_gate:
            expert_delta = expert_delta * torch.tanh(self.residual_scale)

        aux = self._diagnostics(band_gates, expert_delta, band_attn, mask)
        return SpectralPromptOutput(
            band_tokens=band_tokens,
            band_gates=band_gates,
            expert_delta=expert_delta,
            aux=aux,
        )

    def _diagnostics(
        self,
        band_gates: torch.Tensor,
        expert_delta: torch.Tensor,
        band_attn: torch.Tensor | None,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = torch.finfo(band_gates.dtype).eps
        entropy = -(band_gates.clamp_min(eps) * band_gates.clamp_min(eps).log()).sum(dim=-1).mean()
        denom = band_gates.new_tensor(float(max(self.num_bands, 2))).log()
        aux: dict[str, torch.Tensor] = {
            "spectral_prompt_entropy": entropy,
            "spectral_prompt_entropy_norm": entropy / denom.clamp_min(eps),
            "spectral_prompt_gate_low": band_gates[:, 0].mean(),
            "spectral_prompt_delta_norm": expert_delta.detach().float().square().mean().sqrt().to(band_gates.dtype),
        }
        if self.num_bands > 1:
            aux["spectral_prompt_gate_mid"] = band_gates[:, 1].mean()
        if self.num_bands > 2:
            aux["spectral_prompt_gate_high"] = band_gates[:, 2].mean()
        if band_attn is not None:
            aux["spectral_prompt_attn_entropy_norm"] = _attention_entropy_norm(band_attn, mask)
        return aux


def merge_spectral_expert_context(
    expert_context: torch.Tensor | None,
    spectral_delta: torch.Tensor | None,
) -> torch.Tensor | None:
    """Add spectral expert residuals to the existing operator context."""

    if spectral_delta is None:
        return expert_context
    if spectral_delta.ndim != 3:
        raise ValueError(f"spectral_delta must be [B,K,D], got {tuple(spectral_delta.shape)}")
    if expert_context is None:
        return spectral_delta
    if expert_context.ndim != 3:
        raise ValueError(f"expert_context must be [B,K,D], got {tuple(expert_context.shape)}")
    if expert_context.shape[0] != spectral_delta.shape[0] or expert_context.shape[-1] != spectral_delta.shape[-1]:
        raise ValueError(
            "expert_context and spectral_delta must share batch and feature dimensions, "
            f"got {tuple(expert_context.shape)} and {tuple(spectral_delta.shape)}"
        )
    delta = spectral_delta
    if delta.shape[1] != expert_context.shape[1]:
        if delta.shape[1] > expert_context.shape[1]:
            delta = delta[:, : expert_context.shape[1]]
        else:
            repeat = (expert_context.shape[1] + delta.shape[1] - 1) // delta.shape[1]
            delta = delta.repeat(1, repeat, 1)[:, : expert_context.shape[1]]
    return expert_context + delta.to(device=expert_context.device, dtype=expert_context.dtype)


def _compatible_heads(d_model: int, requested: int) -> int:
    heads = max(1, min(int(requested), int(d_model)))
    while d_model % heads != 0 and heads > 1:
        heads -= 1
    return heads


def _valid_slot_mask(
    slot_mask: torch.Tensor | None,
    *,
    batch: int,
    slots: int,
    device: torch.device,
) -> torch.Tensor:
    if slot_mask is None:
        return torch.ones(batch, slots, device=device, dtype=torch.bool)
    if slot_mask.shape != (batch, slots):
        raise ValueError(f"slot_mask must be [{batch},{slots}], got {tuple(slot_mask.shape)}")
    mask = slot_mask.to(device=device).bool()
    empty = ~mask.any(dim=1)
    if empty.any():
        mask = mask.clone()
        mask[empty, 0] = True
    return mask


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(device=tokens.device, dtype=tokens.dtype)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (tokens * mask_f.unsqueeze(-1)).sum(dim=1) / denom


def _attention_entropy_norm(attn: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # attn: [B,H,Q,J] from nn.MultiheadAttention with average_attn_weights=False.
    eps = torch.finfo(attn.dtype).eps
    valid = mask[:, None, None, :].to(device=attn.device, dtype=attn.dtype)
    prob = attn * valid
    prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(prob.clamp_min(eps) * prob.clamp_min(eps).log()).sum(dim=-1)
    denom = mask.sum(dim=-1).clamp_min(2).to(device=attn.device, dtype=attn.dtype).log()
    return (entropy / denom[:, None, None].clamp_min(eps)).mean()
