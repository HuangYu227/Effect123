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
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_channels <= 0 or num_operators <= 0:
            raise ValueError("d_model, num_channels, and num_operators must be positive")
        self.d_model = int(d_model)
        self.num_channels = int(num_channels)
        self.num_operators = int(num_operators)
        self.hidden_dim = int(hidden_dim or d_model)
        self.temperature = float(temperature)
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

    def forward(
        self,
        *,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_context: torch.Tensor,
        velocity_shape: tuple[int, int, int, int] | torch.Size | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        self._validate_inputs(x_t, t, text_context)
        dtype = text_context.dtype
        device = text_context.device
        x_t = x_t.to(device=device, dtype=dtype)
        t = t.to(device=device, dtype=dtype).view(-1)
        h = self.state_encoder(x_t.transpose(1, 2).contiguous())
        state = self.state_proj(torch.cat([h.mean(dim=-1), h.std(dim=-1, unbiased=False)], dim=-1))
        text = self.text_proj(text_context)
        time = self.time_proj(t[:, None])
        logits = self.head(torch.cat([state, text, time], dim=-1))
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
        }
        return gate, g, aux

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
