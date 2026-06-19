from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.nn import functional as F


def normalized_multi_resolution_fft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    fft_sizes: Iterable[int] = (16, 32, 64, 128),
    log_magnitude: bool = True,
    distance: str = "l1",
    mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    reduction: str = "mean",
) -> torch.Tensor:
    """Normalized multi-resolution rFFT magnitude loss for [B,L,C] signals.

    Each sample/channel spectrum is RMS-normalized before comparison. This keeps
    the loss focused on spectral shape and periodic structure instead of simply
    duplicating time-domain amplitude MSE.
    """

    if pred.shape != target.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} must match target {tuple(target.shape)}")
    if pred.ndim != 3:
        raise ValueError(f"normalized MR-FFT loss expects [B,L,C], got {tuple(pred.shape)}")
    if reduction not in {"mean", "none"}:
        raise ValueError(f"Unknown reduction {reduction!r}; expected 'mean' or 'none'")
    sizes = _parse_fft_sizes(fft_sizes)
    if not sizes or pred.shape[1] < 2:
        return pred.new_zeros(pred.shape[0]) if reduction == "none" else pred.new_zeros(())

    pred_f, target_f = _apply_mask(pred, target, mask)
    fft_dtype = torch.float32 if pred_f.dtype in {torch.float16, torch.bfloat16} else pred_f.dtype
    pred_f = pred_f.to(fft_dtype).transpose(1, 2).contiguous()
    target_f = target_f.to(fft_dtype).transpose(1, 2).contiguous()

    losses: list[torch.Tensor] = []
    for n_fft in sizes:
        p_mag = torch.fft.rfft(pred_f, n=n_fft, dim=-1).abs()
        t_mag = torch.fft.rfft(target_f, n=n_fft, dim=-1).abs()
        p_mag = _normalize_spectrum(p_mag, eps=eps)
        t_mag = _normalize_spectrum(t_mag, eps=eps)
        if log_magnitude:
            p_mag = torch.log1p(p_mag)
            t_mag = torch.log1p(t_mag)
        diff = _distance(p_mag, t_mag, distance)
        losses.append(diff.mean(dim=(1, 2)))
    per_sample = torch.stack(losses, dim=0).mean(dim=0).to(dtype=pred.dtype)
    if reduction == "none":
        return per_sample
    return per_sample.mean()


def _parse_fft_sizes(fft_sizes: Iterable[int]) -> tuple[int, ...]:
    out = tuple(sorted({int(size) for size in fft_sizes if int(size) > 1}))
    return out


def _apply_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mask is None:
        return pred, target
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    if mask.shape == pred.shape[:2]:
        mask = mask.unsqueeze(-1)
    if mask.shape == (*pred.shape[:2], 1):
        mask = mask.expand_as(pred)
    elif mask.shape != pred.shape:
        raise ValueError(f"spectral mask must broadcast to {tuple(pred.shape)}, got {tuple(mask.shape)}")
    return pred * mask, target * mask


def _normalize_spectrum(mag: torch.Tensor, *, eps: float) -> torch.Tensor:
    scale = mag.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(float(eps))
    return mag / scale


def _distance(pred: torch.Tensor, target: torch.Tensor, distance: str) -> torch.Tensor:
    mode = str(distance).lower()
    if mode in {"l1", "mae"}:
        return (pred - target).abs()
    if mode in {"l2", "mse"}:
        return (pred - target).square()
    if mode in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError("spectral_distance must be one of: 'l1', 'mse', 'smooth_l1'")
