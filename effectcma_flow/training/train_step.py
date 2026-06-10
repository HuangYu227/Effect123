from __future__ import annotations

from typing import Any

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
) -> dict[str, Any]:
    if device is not None:
        batch = batch_to_device(batch, device)
    task_mode = str(task_mode).lower()
    if task_mode == "edit":
        base = batch["B"]
        target = batch["Y"]
        batch_size = base.shape[0]
        t = torch.rand(batch_size, device=base.device, dtype=base.dtype)
        x_t = (1.0 - t[:, None, None]) * base + t[:, None, None] * target
        target_v = target - base
        text_condition = text_condition_from_batch(batch, text_encoder_mode, condition_key="slots")
        pred_v, aux = model(base, x_t, t, text_condition)
    elif task_mode == "text2ts":
        target = batch["Y"]
        batch_size = target.shape[0]
        t = torch.rand(batch_size, device=target.device, dtype=target.dtype)
        source = torch.randn_like(target) * float(noise_scale)
        x_t = (1.0 - t[:, None, None]) * source + t[:, None, None] * target
        target_v = target - source
        text_condition = text_condition_from_batch(batch, text_encoder_mode, condition_key="caption")
        pred_v, aux = model(x_t, t, text_condition)
    else:
        raise ValueError(f"Unknown task_mode {task_mode!r}; expected 'text2ts' or 'edit'")
    sq_error = (pred_v - target_v) ** 2
    loss, loss_parts = _cfm_loss(sq_error, batch.get("mask"), mode=cfm_loss_mode)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    with torch.no_grad():
        op_usage = aux["A_o"].mean(dim=(0, 1)).detach()
        gate_entropy = _entropy(aux["A_o"], dim=-1).mean().detach()
        contribution = _operator_contribution(aux)
        time_entropy = _maybe_entropy(aux, "A_t")
        channel_entropy = _maybe_entropy(aux, "A_c")
    return {
        "loss": loss.detach(),
        **loss_parts,
        "pred_v": pred_v.detach(),
        "target_v": target_v.detach(),
        "aux": aux,
        "operator_usage": op_usage,
        "operator_contribution": contribution,
        "operator_gate_entropy": gate_entropy,
        "time_gate_entropy": time_entropy,
        "channel_gate_entropy": channel_entropy,
    }


def _entropy(prob: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = prob.clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


def _maybe_entropy(aux: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in aux:
        return torch.tensor(float("nan"), device=aux["A_o"].device)
    return _entropy(aux[key], dim=-1).mean().detach()


def _operator_contribution(aux: dict[str, torch.Tensor]) -> torch.Tensor:
    if "G" not in aux or "V" not in aux:
        return torch.full((aux["A_o"].shape[-1],), float("nan"), device=aux["A_o"].device)
    return (aux["G"] * aux["V"]).abs().mean(dim=(0, 1, 2)).detach()


def _cfm_loss(sq_error: torch.Tensor, mask: torch.Tensor | None, *, mode: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    global_loss = sq_error.mean()
    parts = {"loss_global": global_loss.detach()}
    if mode == "global" or mask is None:
        return global_loss, parts
    if mode != "balanced":
        raise ValueError(f"Unknown cfm_loss_mode {mode!r}; expected 'global' or 'balanced'")
    if mask.shape != sq_error.shape:
        raise ValueError(f"balanced CFM requires mask shape {tuple(sq_error.shape)}, got {tuple(mask.shape)}")
    mask = mask.to(sq_error.device, dtype=sq_error.dtype)
    inv_mask = 1.0 - mask
    inside = (sq_error * mask).sum() / mask.sum().clamp_min(1.0)
    outside = (sq_error * inv_mask).sum() / inv_mask.sum().clamp_min(1.0)
    parts.update({"loss_inside": inside.detach(), "loss_outside": outside.detach()})
    return 0.5 * (inside + outside), parts
