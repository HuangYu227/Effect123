"""Global operator gate for V6.1.

The old fine-grained A_t/A_c router showed near-maximum entropy in experiments,
so V6.1 provides a simpler and more honest operator-level gate.  It predicts a
sample-level mixture over mechanism-aware velocity experts and expands it over
all time steps and channels.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GlobalOperatorGate(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        num_channels: int,
        num_operators: int,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        router_type: str = "mlp",
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_channels <= 0 or num_operators <= 0:
            raise ValueError("d_model, num_channels, and num_operators must be positive")
        self.d_model = int(d_model)
        self.num_channels = int(num_channels)
        self.num_operators = int(num_operators)
        self.hidden_dim = int(hidden_dim or d_model)
        self.temperature = float(temperature)
        self.router_type = str(router_type).lower()
        if self.router_type not in {"mlp", "attention"}:
            raise ValueError(f"router_type must be 'mlp' or 'attention', got {router_type!r}")
        self.state_encoder = nn.Sequential(
            nn.Conv1d(self.num_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
        )
        self.time_proj = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.head = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_operators),
        )
        heads = max(1, min(int(num_heads), self.hidden_dim))
        while self.hidden_dim % heads != 0 and heads > 1:
            heads -= 1
        self.operator_queries = nn.Parameter(torch.randn(self.num_operators, self.hidden_dim) * 0.02)
        self.text_token_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.hidden_dim),
            nn.GELU(),
        )
        self.gate_attn = nn.MultiheadAttention(
            self.hidden_dim,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        self.operator_prior = nn.Parameter(torch.zeros(self.num_operators))

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_context: torch.Tensor,
        slot_tokens: torch.Tensor | None = None,
        slot_mask: torch.Tensor | None = None,
        velocity_shape: tuple[int, int, int, int] | torch.Size | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        self._validate_inputs(x_t, t, text_context)
        dtype = text_context.dtype
        device = text_context.device
        x_t = x_t.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype).view(-1)
        h = self.state_encoder(x_t.transpose(1, 2).contiguous())
        if self.router_type == "attention":
            logits, attn_aux = self._attention_logits(
                h=h,
                t=t,
                text_context=text_context,
                slot_tokens=slot_tokens,
                slot_mask=slot_mask,
            )
        else:
            state = self.state_proj(torch.cat([h.mean(dim=-1), h.std(dim=-1, unbiased=False)], dim=-1))
            text = self.text_proj(text_context)
            time = self.time_proj(t[:, None])
            logits = self.head(torch.cat([state, text, time], dim=-1))
            attn_aux = {}
        gate = torch.softmax(logits / max(self.temperature, 1e-4), dim=-1)
        g = None
        if velocity_shape is not None:
            if len(velocity_shape) != 4:
                raise ValueError(f"velocity_shape must be [B,L,C,K], got {tuple(velocity_shape)}")
            b, length, channels, k = [int(v) for v in velocity_shape]
            if gate.shape != (b, k):
                raise ValueError(f"gate shape {tuple(gate.shape)} incompatible with velocity_shape {tuple(velocity_shape)}")
            g = gate[:, None, None, :].expand(b, length, channels, k)
        entropy = _entropy(gate, dim=-1).mean()
        aux = {
            "A_o": gate,
            "operator_gate_logits": logits.detach(),
            "operator_gate_entropy": entropy.detach(),
            "operator_gate_entropy_norm": (entropy / torch.log(torch.tensor(float(max(self.num_operators, 2)), device=device, dtype=dtype))).detach(),
            "operator_gate_max_prob": gate.max(dim=-1).values.mean().detach(),
            "operator_usage": gate.mean(dim=0).detach(),
            **attn_aux,
        }
        return gate, g, aux

    def _attention_logits(
        self,
        *,
        h: torch.Tensor,
        t: torch.Tensor,
        text_context: torch.Tensor,
        slot_tokens: torch.Tensor | None,
        slot_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = h.shape[0]
        dtype = text_context.dtype
        state_tokens = h.transpose(1, 2).contiguous()
        time_token = self.time_proj(t[:, None]).to(dtype=dtype)[:, None, :]
        if slot_tokens is None:
            text_tokens = self.text_token_proj(text_context)[:, None, :]
            text_mask = torch.ones(batch, 1, device=text_context.device, dtype=torch.bool)
        else:
            if slot_tokens.ndim != 3 or slot_tokens.shape[0] != batch or slot_tokens.shape[-1] != self.d_model:
                raise ValueError(
                    f"slot_tokens must be [B,J,{self.d_model}] when provided, got {getattr(slot_tokens, 'shape', None)}"
                )
            text_tokens = self.text_token_proj(slot_tokens.to(device=text_context.device, dtype=dtype))
            if slot_mask is None:
                text_mask = torch.ones(text_tokens.shape[:2], device=text_tokens.device, dtype=torch.bool)
            else:
                if slot_mask.shape != text_tokens.shape[:2]:
                    raise ValueError(f"slot_mask must have shape {tuple(text_tokens.shape[:2])}, got {tuple(slot_mask.shape)}")
                text_mask = slot_mask.to(device=text_tokens.device) > 0
                empty = ~text_mask.any(dim=1)
                if empty.any():
                    text_mask = text_mask.clone()
                    text_mask[empty, 0] = True
        memory = torch.cat([state_tokens, text_tokens, time_token], dim=1)
        state_mask = torch.ones(batch, state_tokens.shape[1], device=memory.device, dtype=torch.bool)
        time_mask = torch.ones(batch, 1, device=memory.device, dtype=torch.bool)
        memory_mask = torch.cat([state_mask, text_mask, time_mask], dim=1)
        query = self.operator_queries[None].expand(batch, -1, -1).to(dtype=dtype)
        attended, attn = self.gate_attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=~memory_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        logits = self.attn_gate_head(attended).squeeze(-1) + self.operator_prior.to(device=attended.device, dtype=dtype)
        state_end = state_tokens.shape[1]
        text_end = state_end + text_tokens.shape[1]
        attn_detached = attn.detach()
        text_mass = attn_detached[:, :, :, state_end:text_end].sum(dim=-1).mean()
        state_mass = attn_detached[:, :, :, :state_end].sum(dim=-1).mean()
        entropy = _masked_attention_entropy(attn_detached, memory_mask).mean()
        entropy_norm = entropy / torch.log(
            memory_mask.sum(dim=-1).float().clamp_min(2.0).to(device=attn.device, dtype=attn.dtype)
        ).mean()
        aux = {
            "operator_gate_attention_entropy": entropy.detach(),
            "operator_gate_attention_entropy_norm": entropy_norm.detach(),
            "operator_gate_attention_text_mass": text_mass.detach(),
            "operator_gate_attention_state_mass": state_mass.detach(),
        }
        return logits, aux

    @staticmethod
    def _validate_inputs(x_t: torch.Tensor, t: torch.Tensor, text_context: torch.Tensor) -> None:
        if not torch.is_tensor(x_t) or x_t.ndim != 3:
            raise ValueError(f"x_t must be [B,L,C], got {getattr(x_t, 'shape', None)}")
        if not torch.is_tensor(t) or t.ndim not in {1, 2}:
            raise ValueError(f"t must be [B] or [B,1], got {getattr(t, 'shape', None)}")
        if not torch.is_tensor(text_context) or text_context.ndim != 2:
            raise ValueError(f"text_context must be [B,D], got {getattr(text_context, 'shape', None)}")
        if x_t.shape[0] != t.shape[0] or x_t.shape[0] != text_context.shape[0]:
            raise ValueError("batch dimensions of x_t, t, and text_context must match")


def _entropy(prob: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def _masked_attention_entropy(attn: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # attn: [B,H,Q,M], mask: [B,M]
    mask_f = mask[:, None, None, :].to(device=attn.device, dtype=attn.dtype)
    p = (attn * mask_f).clamp_min(eps)
    return -(p * p.log()).sum(dim=-1)
