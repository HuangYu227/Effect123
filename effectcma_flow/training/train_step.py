"""CFM training step for V6.1/V6.2.

Training objective:
    L = L_CFM + lambda_ortho * L_regime_ortho + lambda_bridge * L_bridge_align

No routing entropy, channel entropy, field mass, trend, frequency, volatility,
or artificial semantic-slot losses are used.
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
    condition_dropout_prob: float = 0.0,
    regime_ortho_weight: float = 0.0,
    bridge_alignment_weight: float = 0.0,
    # Backward-compatibility guard: old routing losses must stay disabled.
    routing_loss_weight: float = 0.0,
    **unused: Any,
) -> dict[str, Any]:
    if float(routing_loss_weight) != 0.0:
        raise ValueError("V6.1 forbids old routing/meta losses. Use regime_ortho_weight only.")
    if unused:
        warnings.warn(f"Unused train_step kwargs ignored: {sorted(unused.keys())}", RuntimeWarning, stacklevel=2)
    if device is not None:
        batch = batch_to_device(batch, device)
    task_mode = str(task_mode).lower()

    if task_mode == "edit":
        _require_keys(batch, ("B", "Y"), mode="edit")
        base = _expect_series(batch["B"], name="batch['B']")
        target = _expect_series(batch["Y"], name="batch['Y']")
        _require_same_shape(base, target, "B", "Y")
        source = base
        t = torch.rand(base.shape[0], device=base.device, dtype=base.dtype)
        x_t = (1.0 - t[:, None, None]) * base + t[:, None, None] * target
        target_v = target - base
        text_condition = text_condition_from_batch(batch, text_encoder_mode, condition_key="slots")
        pred_v, aux = model(base, x_t, t, text_condition)
    elif task_mode == "text2ts":
        _require_keys(batch, ("Y",), mode="text2ts")
        target = _expect_series(batch["Y"], name="batch['Y']")
        source = torch.randn_like(target) * float(noise_scale)
        t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype)
        x_t = (1.0 - t[:, None, None]) * source + t[:, None, None] * target
        target_v = target - source
        text_condition = text_condition_from_batch(
            _maybe_blank_caption_batch(batch, p=float(condition_dropout_prob)),
            text_encoder_mode,
            condition_key="caption",
            caption_slot_strategy=caption_slot_strategy,
            max_caption_slots=max_caption_slots,
            include_all_caption_candidates=include_all_caption_candidates,
        )
        # Note: prepare_condition is called inside model.forward(), not here.
        # This avoids redundant text-encoding and keeps dtype handling consistent.
        pred_v, aux = model(x_t, t, text_condition)
    else:
        raise ValueError(f"Unknown task_mode {task_mode!r}; expected 'text2ts' or 'edit'")

    if pred_v.shape != target_v.shape:
        raise ValueError(f"model pred_v shape {tuple(pred_v.shape)} must match target_v {tuple(target_v.shape)}")
    sq_error = (pred_v - target_v) ** 2
    cfm_loss, loss_parts = _cfm_loss(sq_error, batch.get("mask"), mode=cfm_loss_mode)
    regime_ortho = pred_v.new_zeros(())
    if float(regime_ortho_weight) > 0.0 and isinstance(aux, dict) and torch.is_tensor(aux.get("regime_ortho_loss")):
        regime_ortho = aux["regime_ortho_loss"].to(device=pred_v.device, dtype=pred_v.dtype)
    bridge_alignment = pred_v.new_zeros(())
    if float(bridge_alignment_weight) > 0.0 and isinstance(aux, dict) and torch.is_tensor(aux.get("bridge_alignment_loss")):
        bridge_alignment = aux["bridge_alignment_loss"].to(device=pred_v.device, dtype=pred_v.dtype)
    loss = cfm_loss + float(regime_ortho_weight) * regime_ortho + float(bridge_alignment_weight) * bridge_alignment

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    with torch.no_grad():
        stats = _flow_diagnostics(target=target, source=source, pred_v=pred_v, target_v=target_v)
        diag = _aux_diagnostics(aux)
    return {
        "loss": loss.detach(),
        "loss_cfm": cfm_loss.detach(),
        "loss_regime_ortho": regime_ortho.detach(),
        "loss_bridge_alignment": bridge_alignment.detach(),
        **loss_parts,
        "pred_v": pred_v.detach(),
        "target_v": target_v.detach(),
        "aux": detach_aux(aux),
        **diag,
        **stats,
    }


def detach_aux(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {k: detach_aux(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_aux(v) for v in value)
    return value


def _maybe_blank_caption_batch(batch: dict[str, Any], *, p: float) -> dict[str, Any]:
    if p <= 0.0 or "caption" not in batch:
        return batch
    captions = list(batch["caption"])
    if not captions:
        return batch
    # Sample-level dropout: each caption is independently blanked.
    mask = torch.rand(len(captions)) < p
    if not mask.any():
        return batch
    new_captions = ["" if m else c for m, c in zip(mask, captions)]
    new_batch = dict(batch)
    new_batch["caption"] = new_captions
    # Also blank caption_candidates if present, to prevent text leakage.
    if "caption_candidates" in batch and batch["caption_candidates"] is not None:
        new_batch["caption_candidates"] = [[] if m else c for m, c in zip(mask, batch["caption_candidates"])]
    return new_batch


def _expect_series(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 3:
        raise ValueError(f"{name} must be [B,L,C], got {tuple(x.shape)}")
    return x.float() if not torch.is_floating_point(x) else x


def _require_keys(batch: dict[str, Any], keys: tuple[str, ...], *, mode: str) -> None:
    missing = [k for k in keys if k not in batch]
    if missing:
        raise ValueError(f"{mode} batch missing keys: {missing}")


def _require_same_shape(a: torch.Tensor, b: torch.Tensor, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have same shape, got {tuple(a.shape)} vs {tuple(b.shape)}")


def _cfm_loss(sq_error: torch.Tensor, mask: torch.Tensor | None, *, mode: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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


def _flow_diagnostics(*, target: torch.Tensor, source: torch.Tensor, pred_v: torch.Tensor, target_v: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_v_rms = pred_v.detach().square().mean().sqrt()
    target_v_rms = target_v.detach().square().mean().sqrt()
    return {
        "source_mse": (source.detach() - target.detach()).square().mean(),
        "source_std": source.detach().std(unbiased=False),
        "target_std": target.detach().std(unbiased=False),
        "pred_v_rms": pred_v_rms,
        "target_v_rms": target_v_rms,
        "pred_target_rms_ratio": pred_v_rms / target_v_rms.clamp_min(1e-8),
        "velocity_cos": (pred_v.detach() * target_v.detach()).mean() / (pred_v_rms * target_v_rms).clamp_min(1e-8),
    }


def _aux_diagnostics(aux: dict[str, Any]) -> dict[str, torch.Tensor]:
    if not isinstance(aux, dict):
        return {}
    out: dict[str, torch.Tensor] = {}
    for key in [
        "regime_entropy",
        "regime_entropy_norm",
        "regime_max_prob",
        "regime_state_weight",
        "operator_gate_entropy",
        "operator_gate_entropy_norm",
        "operator_gate_max_prob",
        "bridge_alignment_loss",
        "text_slot_count",
        "text_slot_count_min",
        "text_slot_count_max",
        "bridge_text_to_state_entropy",
        "bridge_text_to_state_entropy_norm",
        "bridge_text_to_state_max_prob",
        "bridge_state_to_text_entropy",
        "bridge_state_to_text_entropy_norm",
        "bridge_state_to_text_max_prob",
        "bridge_context_norm",
        "bridge_text_context_norm",
        "bridge_state_context_norm",
        "bridge_expert_context_norm",
        "bridge_alignment_logit_pos",
        "bridge_alignment_logit_std",
    ]:
        val = aux.get(key)
        if torch.is_tensor(val):
            out[key] = val.detach()
    if torch.is_tensor(aux.get("regime_usage")):
        out["regime_usage"] = aux["regime_usage"].detach()
    if torch.is_tensor(aux.get("operator_usage")):
        out["operator_usage"] = aux["operator_usage"].detach()
    return out
