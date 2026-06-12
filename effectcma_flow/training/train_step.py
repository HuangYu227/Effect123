"""Single-objective Conditional Flow Matching training step.

Clean V6 deliberately uses a single training objective:

    L = E[||v_theta(x_t, t, c) - v_target||^2]

Router/operator quantities are returned only as diagnostics. They are not
optimized by auxiliary entropy, balance, sparsity, field-mass, or meta losses.
This keeps the framework closer to a publishable generative-model training
objective and avoids manual loss-weight tuning.
"""
from __future__ import annotations

from typing import Any
import warnings

import torch

from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def cfm_train_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    *,
    device: torch.device | None = None,
    grad_clip: float | None = 1.0,
    text_encoder_mode: str = "hash",
    cfm_loss_mode: str = "global",
    task_mode: str = "edit",
    noise_scale: float = 1.0,
    caption_slot_strategy: str = "single",
    max_caption_slots: int = 1,
    include_all_caption_candidates: bool = False,
    # Kept only for backward compatibility. Clean V6 must keep these disabled.
    routing_loss_weight: float = 0.0,
    routing_loss_weights: Any | None = None,
    routing_loss_targets: Any | None = None,
) -> dict[str, Any]:
    if float(routing_loss_weight) != 0.0:
        raise ValueError(
            "Clean V6 disables routing/meta auxiliary losses. "
            "Set train.routing_loss_weight: 0.0 or remove routing_loss_weight from the config."
        )
    if routing_loss_weights is not None or routing_loss_targets is not None:
        warnings.warn(
            "routing_loss_weights/routing_loss_targets are ignored in Clean V6; "
            "remove them from the config for reproducible single-loss training.",
            RuntimeWarning,
            stacklevel=2,
        )

    if device is not None:
        batch = batch_to_device(batch, device)
    task_mode = str(task_mode).lower()

    if task_mode == "edit":
        _require_keys(batch, ("B", "Y"), mode="edit")
        base = _expect_series(batch["B"], name="batch['B']")
        target = _expect_series(batch["Y"], name="batch['Y']")
        _require_same_shape(base, target, "B", "Y")
        source = base
        batch_size = base.shape[0]
        t = torch.rand(batch_size, device=base.device, dtype=base.dtype)
        x_t = (1.0 - t[:, None, None]) * base + t[:, None, None] * target
        target_v = target - base
        text_condition = text_condition_from_batch(batch, text_encoder_mode, condition_key="slots")
        pred_v, aux = model(base, x_t, t, text_condition)

    elif task_mode == "text2ts":
        _require_keys(batch, ("Y",), mode="text2ts")
        target = _expect_series(batch["Y"], name="batch['Y']")
        batch_size = target.shape[0]
        t = torch.rand(batch_size, device=target.device, dtype=target.dtype)
        source = torch.randn_like(target) * float(noise_scale)
        x_t = (1.0 - t[:, None, None]) * source + t[:, None, None] * target
        target_v = target - source
        text_condition = text_condition_from_batch(
            batch,
            text_encoder_mode,
            condition_key="caption",
            caption_slot_strategy=caption_slot_strategy,
            max_caption_slots=max_caption_slots,
            include_all_caption_candidates=include_all_caption_candidates,
        )
        if hasattr(model, "prepare_condition"):
            text_condition = model.prepare_condition(text_condition, device=target.device, dtype=target.dtype)
        pred_v, aux = model(x_t, t, text_condition)

    else:
        raise ValueError(f"Unknown task_mode {task_mode!r}; expected 'text2ts' or 'edit'")

    if pred_v.shape != target_v.shape:
        raise ValueError(f"model pred_v shape {tuple(pred_v.shape)} must match target_v {tuple(target_v.shape)}")
    sq_error = (pred_v - target_v) ** 2
    loss, loss_parts = _cfm_loss(sq_error, batch.get("mask"), mode=cfm_loss_mode)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    # Diagnostics only — no gradient, no auxiliary loss.
    with torch.no_grad():
        op_usage = _maybe_operator_usage(aux)
        gate_entropy = _maybe_entropy(aux, "A_o")
        time_entropy = _maybe_entropy(aux, "A_t")
        channel_entropy = _maybe_entropy(aux, "A_c")
        flow_stats = _flow_diagnostics(target=target, source=source, pred_v=pred_v, target_v=target_v)
        norm_stats = _router_diagnostics(aux)
        contribution = _operator_contribution(aux)

    return {
        "loss": loss.detach(),
        "loss_cfm": loss.detach(),
        **loss_parts,
        "pred_v": pred_v.detach(),
        "target_v": target_v.detach(),
        "aux": detach_aux(aux),
        "operator_usage": op_usage,
        "operator_contribution": contribution,
        "operator_gate_entropy": gate_entropy,
        "time_gate_entropy": time_entropy,
        "channel_gate_entropy": channel_entropy,
        **norm_stats,
        **flow_stats,
    }


# ---------------------------------------------------------------------------
# Aux utilities
# ---------------------------------------------------------------------------

def detach_aux(value: Any) -> Any:
    """Detach tensors recursively before returning them to loggers."""
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_aux(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_aux(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _expect_series(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(f"{name} must be [B, L, C], got {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        x = x.float()
    return x


def _require_keys(batch: dict[str, Any], keys: tuple[str, ...], *, mode: str) -> None:
    missing = [k for k in keys if k not in batch]
    if missing:
        raise ValueError(f"{mode} training batch missing required keys: {missing}")


def _require_same_shape(a: torch.Tensor, b: torch.Tensor, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have identical shapes, got {tuple(a.shape)} and {tuple(b.shape)}")


# ---------------------------------------------------------------------------
# Diagnostic helpers (no gradient)
# ---------------------------------------------------------------------------

def _entropy(prob: torch.Tensor, *, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def _maybe_entropy(aux: dict[str, Any], key: str) -> torch.Tensor:
    if key not in aux or not torch.is_tensor(aux[key]):
        return torch.tensor(float("nan"), device=_aux_device(aux))
    return _entropy(aux[key], dim=-1).mean().detach()


def _router_diagnostics(aux: dict[str, Any]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, label in (("A_o", "operator"), ("A_t", "time"), ("A_c", "channel")):
        p = aux.get(key)
        if torch.is_tensor(p) and p.numel() > 0:
            dim = p.shape[-1]
            entropy = _entropy(p, dim=-1).mean().detach()
            denom = torch.log(torch.tensor(float(max(dim, 2)), device=p.device, dtype=p.dtype)).clamp_min(1e-8)
            out[f"{label}_gate_entropy_norm"] = (entropy / denom).detach()
            out[f"{label}_gate_max_prob"] = p.max(dim=-1).values.mean().detach()
    if "G" in aux and torch.is_tensor(aux["G"]):
        g = aux["G"].detach()
        out["gate_abs_mass_mean"] = g.abs().sum(dim=-1).mean()
        out["gate_abs_mass_std"] = g.abs().sum(dim=-1).std(unbiased=False)
    return out


def _operator_contribution(aux: dict[str, Any]) -> torch.Tensor:
    if "G" not in aux or "V" not in aux or not torch.is_tensor(aux["G"]) or not torch.is_tensor(aux["V"]):
        a_o = aux.get("A_o")
        if torch.is_tensor(a_o):
            return torch.full((a_o.shape[-1],), float("nan"), device=a_o.device)
        return torch.full((1,), float("nan"), device=_aux_device(aux))
    return (aux["G"] * aux["V"]).abs().mean(dim=(0, 1, 2)).detach()


def _maybe_operator_usage(aux: dict[str, Any]) -> torch.Tensor:
    a_o = aux.get("A_o")
    if not torch.is_tensor(a_o):
        return torch.full((1,), float("nan"), device=_aux_device(aux))
    return a_o.mean(dim=tuple(range(a_o.ndim - 1))).detach()


def _aux_device(aux: dict[str, Any]) -> torch.device:
    for value in aux.values():
        if torch.is_tensor(value):
            return value.device
        if isinstance(value, dict):
            nested = _aux_device(value)
            if nested.type != "cpu":
                return nested
    return torch.device("cpu")


def _flow_diagnostics(
    *,
    target: torch.Tensor,
    source: torch.Tensor,
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    source = source.detach()
    target = target.detach()
    pred_v = pred_v.detach()
    target_v = target_v.detach()
    pred_v_rms = pred_v.square().mean().sqrt()
    target_v_rms = target_v.square().mean().sqrt()
    return {
        "source_mse": (source - target).square().mean(),
        "source_std": source.std(unbiased=False),
        "target_std": target.std(unbiased=False),
        "pred_v_rms": pred_v_rms,
        "target_v_rms": target_v_rms,
        "pred_target_rms_ratio": pred_v_rms / target_v_rms.clamp_min(1e-8),
        "velocity_cos": (pred_v * target_v).mean() / (pred_v_rms * target_v_rms).clamp_min(1e-8),
    }


# ---------------------------------------------------------------------------
# CFM loss
# ---------------------------------------------------------------------------

def _cfm_loss(
    sq_error: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    global_loss = sq_error.mean()
    parts = {"loss_global": global_loss.detach()}
    if mode == "global" or mask is None:
        return global_loss, parts
    if mode != "balanced":
        raise ValueError(f"Unknown cfm_loss_mode {mode!r}; expected 'global' or 'balanced'")
    mask = mask.to(sq_error.device, dtype=sq_error.dtype)
    if mask.shape == sq_error.shape[:2]:
        mask = mask.unsqueeze(-1).expand_as(sq_error)
    elif mask.shape == (*sq_error.shape[:2], 1):
        mask = mask.expand_as(sq_error)
    elif mask.shape != sq_error.shape:
        raise ValueError(f"balanced CFM requires mask broadcastable to {tuple(sq_error.shape)}, got {tuple(mask.shape)}")
    inv_mask = 1.0 - mask
    inside = (sq_error * mask).sum() / mask.sum().clamp_min(1.0)
    outside = (sq_error * inv_mask).sum() / inv_mask.sum().clamp_min(1.0)
    parts.update({"loss_inside": inside.detach(), "loss_outside": outside.detach()})
    return 0.5 * (inside + outside), parts
