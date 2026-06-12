"""Routing regularization losses for text-time-channel-operator field G.

These losses are intentionally weak.  They never dominate the CFM velocity
MSE; they only prevent G, A_t, A_c, A_o from collapsing to trivial
solutions (uniform routing, one-hot collapse, zero field mass).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RoutingLossWeights:
    """Weights for making the factorized routing field identifiable.

    Defaults are intentionally small. They should regularize G without dominating
    Conditional Flow Matching's velocity loss.
    """
    op_confidence: float = 1.0e-2
    op_balance: float = 1.0e-2
    time_entropy: float = 5.0e-3
    channel_entropy: float = 2.5e-3
    time_tv: float = 1.0e-3
    field_mass: float = 1.0e-4


@dataclass(frozen=True)
class RoutingLossTargets:
    """Target normalized entropies and minimum usage thresholds."""
    op_entropy_norm: float = 0.65
    time_entropy_norm: float = 0.70
    channel_entropy_norm: float = 0.75
    min_operator_usage: float = 0.05
    field_mass: float | None = None


# ---------------------------------------------------------------------------
# Main loss computation
# ---------------------------------------------------------------------------

def compute_routing_regularization(
    aux: dict[str, torch.Tensor],
    *,
    weights: RoutingLossWeights | None = None,
    targets: RoutingLossTargets | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Regularize a text-time-channel-operator routing field.

    The current framework can minimize velocity MSE with a nearly uniform G.
    This function adds weak, interpretable pressures:
      1. per-sample operator confidence so experts specialize;
      2. batch-level operator balance so one expert does not collapse;
      3. target entropy for time/channel routing rather than maximum entropy;
      4. temporal smoothness of A_t;
      5. stable absolute field mass.

    It never assumes fixed P/C/K and safely skips terms whose aux keys are absent.
    """
    weights = weights or RoutingLossWeights()
    targets = targets or RoutingLossTargets()
    device, dtype = _aux_device_dtype(aux)
    total = torch.zeros((), device=device, dtype=dtype)
    parts: dict[str, torch.Tensor] = {}

    # --- Operator routing ---
    a_o = aux.get("A_o")
    if a_o is not None and a_o.numel() > 0:
        a_o = a_o.to(device=device, dtype=dtype)
        k = a_o.shape[-1]
        op_entropy = _entropy(a_o, dim=-1, eps=eps).mean()
        op_entropy_norm = op_entropy / _safe_log_dim(k, device, dtype)
        # Per-sample confidence: push normalized entropy below target floor
        op_conf = F.relu(op_entropy_norm - float(targets.op_entropy_norm)).mean()
        # Batch-level balance: KL(usage || uniform) to prevent expert collapse
        usage = a_o.mean(dim=tuple(range(a_o.ndim - 1))).clamp_min(eps)
        uniform = torch.full_like(usage, 1.0 / float(max(k, 2)))
        op_balance = (usage * (usage / uniform).clamp_min(eps).log()).sum()
        total = total + float(weights.op_confidence) * op_conf + float(weights.op_balance) * op_balance
        parts.update({
            "router_op_entropy": op_entropy.detach(),
            "router_op_entropy_norm": op_entropy_norm.detach(),
            "router_op_confidence": op_conf.detach(),
            "router_op_balance": op_balance.detach(),
            "router_op_usage_min": usage.min().detach(),
            "router_op_usage_max": usage.max().detach(),
        })

    # --- Time routing ---
    a_t = aux.get("A_t")
    if a_t is not None and a_t.numel() > 0:
        a_t = a_t.to(device=device, dtype=dtype)
        p = a_t.shape[-1]
        time_entropy = _entropy(a_t, dim=-1, eps=eps).mean()
        time_entropy_norm = time_entropy / _safe_log_dim(p, device, dtype)
        time_loss = F.relu(time_entropy_norm - float(targets.time_entropy_norm)).mean()
        total = total + float(weights.time_entropy) * time_loss
        parts.update({
            "router_time_entropy": time_entropy.detach(),
            "router_time_entropy_norm": time_entropy_norm.detach(),
            "router_time_confidence": time_loss.detach(),
        })
        if p > 1:
            time_tv = (a_t[..., 1:] - a_t[..., :-1]).abs().mean()
            total = total + float(weights.time_tv) * time_tv
            parts["router_time_tv"] = time_tv.detach()

    # --- Channel routing ---
    a_c = aux.get("A_c")
    if a_c is not None and a_c.numel() > 0:
        a_c = a_c.to(device=device, dtype=dtype)
        c = a_c.shape[-1]
        channel_entropy = _entropy(a_c, dim=-1, eps=eps).mean()
        channel_entropy_norm = channel_entropy / _safe_log_dim(c, device, dtype)
        channel_loss = F.relu(channel_entropy_norm - float(targets.channel_entropy_norm)).mean()
        total = total + float(weights.channel_entropy) * channel_loss
        parts.update({
            "router_channel_entropy": channel_entropy.detach(),
            "router_channel_entropy_norm": channel_entropy_norm.detach(),
            "router_channel_confidence": channel_loss.detach(),
        })

    # --- Field mass ---
    g = aux.get("G")
    if g is not None and g.numel() > 0:
        g = g.to(device=device, dtype=dtype)
        mass = g.abs().sum(dim=-1).mean()
        target_mass = targets.field_mass
        if target_mass is None:
            # Just track mass, no loss pressure
            parts["router_field_mass"] = mass.detach()
        else:
            target_tensor = torch.as_tensor(float(target_mass), device=device, dtype=dtype)
            field_mass_loss = (mass.clamp_min(eps).log() - target_tensor.clamp_min(eps).log()).square()
            total = total + float(weights.field_mass) * field_mass_loss
            parts["router_field_mass"] = mass.detach()
            parts["router_field_mass_loss"] = field_mass_loss.detach()

    parts["router_loss"] = total.detach()
    return total, parts


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_routing_weights(overrides: dict[str, Any] | None = None) -> RoutingLossWeights:
    weights = RoutingLossWeights()
    if not overrides:
        return weights
    for key, value in overrides.items():
        if not hasattr(weights, key):
            raise ValueError(f"Unknown routing loss weight {key!r}")
        weights = replace(weights, **{key: float(value)})
    return weights


def build_routing_targets(overrides: dict[str, Any] | None = None) -> RoutingLossTargets:
    targets = RoutingLossTargets()
    if not overrides:
        return targets
    for key, value in overrides.items():
        if not hasattr(targets, key):
            raise ValueError(f"Unknown routing loss target {key!r}")
        targets = replace(targets, **{key: float(value)})
    return targets


# ---------------------------------------------------------------------------
# Aux utilities
# ---------------------------------------------------------------------------

def detach_aux(aux: dict[str, Any] | Any) -> dict[str, Any] | Any:
    """Detach tensors recursively before returning them to loggers."""
    if not isinstance(aux, dict):
        return aux
    out: dict[str, Any] = {}
    for key, value in aux.items():
        if torch.is_tensor(value):
            out[key] = value.detach()
        elif isinstance(value, dict):
            out[key] = detach_aux(value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entropy(prob: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def _safe_log_dim(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(max(float(dim), 2.0), device=device, dtype=dtype).log().clamp_min(1e-8)


def _aux_device_dtype(aux: dict[str, Any]) -> tuple[torch.device, torch.dtype]:
    for value in aux.values():
        if torch.is_tensor(value):
            return value.device, value.dtype if value.is_floating_point() else torch.float32
        if isinstance(value, dict):
            try:
                return _aux_device_dtype(value)
            except ValueError:
                pass
    return torch.device("cpu"), torch.float32
