from __future__ import annotations

import torch


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, base: torch.Tensor, mask: torch.Tensor, *, threshold: float = 1e-4) -> dict[str, float]:
    _validate_metric_inputs(pred, target, base, mask)
    eps = 1e-8
    pred_res = pred - base
    target_res = target - base
    mask = mask.to(pred.device, dtype=pred.dtype)
    inv_mask = 1.0 - mask
    mse = torch.mean((pred - target) ** 2)
    residual_mse = torch.mean((pred_res - target_res) ** 2)
    inside_count = mask.sum().clamp_min(1.0)
    outside_count = inv_mask.sum().clamp_min(1.0)
    inside_residual_mse = (((pred_res - target_res) ** 2) * mask).sum() / inside_count
    outside_residual_mse = (((pred_res - target_res) ** 2) * inv_mask).sum() / outside_count
    pred_energy = pred_res.abs().sum().clamp_min(eps)
    inside_energy = (pred_res.abs() * mask).sum()
    outside_energy = (pred_res.abs() * inv_mask).sum()
    scope_precision = inside_energy / pred_energy
    leakage = outside_energy / pred_energy

    pred_support = pred_res.abs() > threshold
    true_support = mask > 0.5
    tp = (pred_support & true_support).sum().float()
    fp = (pred_support & ~true_support).sum().float()
    fn = (~pred_support & true_support).sum().float()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "mse": float(mse.detach().cpu()),
        "residual_mse": float(residual_mse.detach().cpu()),
        "inside_residual_mse": float(inside_residual_mse.detach().cpu()),
        "outside_residual_mse": float(outside_residual_mse.detach().cpu()),
        "outside_energy_ratio": float(leakage.detach().cpu()),
        "leakage": float(leakage.detach().cpu()),
        "scope_precision": float(scope_precision.detach().cpu()),
        "mask_precision": float(precision.detach().cpu()),
        "mask_recall": float(recall.detach().cpu()),
        "mask_iou": float(iou.detach().cpu()),
    }


def _validate_metric_inputs(pred: torch.Tensor, target: torch.Tensor, base: torch.Tensor, mask: torch.Tensor) -> None:
    if pred.shape != target.shape or pred.shape != base.shape or pred.shape != mask.shape:
        raise ValueError(
            "pred, target, base, and mask must have identical shapes; "
            f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}, base={tuple(base.shape)}, mask={tuple(mask.shape)}"
        )
    if pred.device != target.device or pred.device != base.device or pred.device != mask.device:
        raise ValueError("pred, target, base, and mask must be on the same device")
    if mask.dtype == torch.bool:
        return
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if not torch.is_floating_point(mask) and mask.dtype not in integer_dtypes:
        raise ValueError(f"mask must be bool or numeric 0/1 tensor, got {mask.dtype}")
    valid = (mask == 0) | (mask == 1)
    if not bool(valid.all().detach().cpu()):
        raise ValueError("mask must be binary with values 0/1")


def average_metric_dicts(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted(items[0].keys())
    return {key: float(sum(item[key] for item in items) / len(items)) for key in keys}


def compute_field_metrics(aux: dict[str, torch.Tensor], mask: torch.Tensor, specs: list[dict], *, threshold: float = 1e-8) -> dict[str, float]:
    """Measure whether the learned effect field is localized like the synthetic spec."""
    if "G" not in aux:
        return {}
    g = aux["G"].detach()
    if g.ndim != 4:
        raise ValueError(f"aux['G'] must be [B, L, C, K], got {tuple(g.shape)}")
    if mask.shape != g.shape[:3]:
        raise ValueError(f"mask must have shape {tuple(g.shape[:3])}, got {tuple(mask.shape)}")
    mask = mask.to(g.device, dtype=g.dtype)
    activation = g.abs().sum(dim=-1)
    total = activation.sum().clamp_min(threshold)
    inside = (activation * mask).sum()
    outside = (activation * (1.0 - mask)).sum()
    metrics = {
        "field_scope_precision": float((inside / total).detach().cpu()),
        "field_leakage": float((outside / total).detach().cpu()),
    }
    if "A_c" in aux:
        a_c = aux["A_c"].detach().mean(dim=1)
        top_channels = a_c.argmax(dim=-1).detach().cpu().tolist()
        hits = [float(int(top in set(spec["channels"]))) for top, spec in zip(top_channels, specs)]
        metrics["channel_top1_acc"] = float(sum(hits) / max(len(hits), 1))
    if "A_o" in aux:
        p = aux["A_o"].detach().clamp_min(1e-8)
        metrics["operator_entropy"] = float((-(p * p.log()).sum(dim=-1).mean()).detach().cpu())
    time_scores = activation.sum(dim=-1)
    positions = torch.arange(g.shape[1], device=g.device, dtype=g.dtype)[None, :]
    centers = (time_scores * positions).sum(dim=-1) / time_scores.sum(dim=-1).clamp_min(threshold)
    target_centers = torch.tensor(
        [(float(spec["start"]) + float(spec["end"]) - 1.0) / 2.0 for spec in specs],
        device=g.device,
        dtype=g.dtype,
    )
    metrics["time_center_mae"] = float((centers - target_centers).abs().mean().detach().cpu())
    return metrics
