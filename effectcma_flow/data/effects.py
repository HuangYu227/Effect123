from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch


SUPPORTED_EFFECTS = (
    "level_shift",
    "trend_up",
    "trend_down",
    "volatility_up",
    "volatility_down",
    "spike",
    "drop",
)


@dataclass(frozen=True)
class EffectSpec:
    effect_type: str
    start: int
    end: int
    channels: tuple[int, ...]
    strength: float
    delay: int = 0

    def to_dict(self) -> dict:
        out = asdict(self)
        out["channels"] = list(self.channels)
        return out


def make_mask(length: int, channels: int, spec: EffectSpec, device=None) -> torch.Tensor:
    validate_spec(spec, length, channels)
    mask = torch.zeros(length, channels, dtype=torch.float32, device=device)
    mask[spec.start : spec.end, list(spec.channels)] = 1.0
    return mask


def validate_spec(spec: EffectSpec, length: int, channels: int) -> None:
    if spec.effect_type not in SUPPORTED_EFFECTS:
        raise ValueError(f"Unsupported effect_type {spec.effect_type!r}")
    if not 0 <= spec.start < spec.end <= length:
        raise ValueError(f"Invalid effect span [{spec.start}, {spec.end}) for length {length}")
    if not spec.channels:
        raise ValueError("EffectSpec.channels must not be empty")
    bad = [c for c in spec.channels if c < 0 or c >= channels]
    if bad:
        raise ValueError(f"Channel indices out of range for C={channels}: {bad}")
    if not torch.isfinite(torch.tensor(float(spec.strength))):
        raise ValueError("EffectSpec.strength must be finite")


def normalize_channels(channels: Iterable[int]) -> tuple[int, ...]:
    unique = tuple(sorted({int(c) for c in channels}))
    if not unique:
        raise ValueError("At least one channel is required")
    return unique


def apply_effect(base: torch.Tensor, spec: EffectSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a synthetic effect to one `[L, C]` trajectory.

    Synthetic effects are data-generation operators only. The model never receives
    `spec` or this operator implementation as input.
    """
    if base.ndim != 2:
        raise ValueError(f"base must have shape [L, C], got {tuple(base.shape)}")
    length, channels = base.shape
    validate_spec(spec, length, channels)
    target = base.clone()
    mask = make_mask(length, channels, spec, device=base.device)
    idx = list(spec.channels)
    s, e = spec.start, spec.end
    seg = target[s:e, idx]

    if spec.effect_type == "level_shift":
        target[s:e, idx] = seg + float(spec.strength)
    elif spec.effect_type == "trend_up":
        target[s:e, idx] = seg + _ramp(e - s, base.device, base.dtype, abs(float(spec.strength)))[:, None]
    elif spec.effect_type == "trend_down":
        target[s:e, idx] = seg - _ramp(e - s, base.device, base.dtype, abs(float(spec.strength)))[:, None]
    elif spec.effect_type == "volatility_up":
        factor = max(float(spec.strength), 1.01)
        target[s:e, idx] = _scale_volatility(seg, factor)
    elif spec.effect_type == "volatility_down":
        factor = min(max(float(spec.strength), 0.01), 0.99)
        target[s:e, idx] = _scale_volatility(seg, factor)
    elif spec.effect_type == "spike":
        pulse = _local_pulse(e - s, base.device, base.dtype, abs(float(spec.strength)))
        target[s:e, idx] = seg + pulse[:, None]
    elif spec.effect_type == "drop":
        pulse = _local_pulse(e - s, base.device, base.dtype, abs(float(spec.strength)))
        target[s:e, idx] = seg - pulse[:, None]
    else:  # pragma: no cover - validate_spec catches this.
        raise ValueError(f"Unsupported effect_type {spec.effect_type!r}")

    return target, mask


def _ramp(length: int, device, dtype, amplitude: float) -> torch.Tensor:
    if length == 1:
        return torch.full((1,), amplitude, device=device, dtype=dtype)
    return torch.linspace(0.0, amplitude, length, device=device, dtype=dtype)


def _local_pulse(length: int, device, dtype, amplitude: float) -> torch.Tensor:
    center = (length - 1) / 2.0
    t = torch.arange(length, device=device, dtype=dtype)
    sigma = max(length / 6.0, 1.0)
    return amplitude * torch.exp(-((t - center) ** 2) / (2.0 * sigma**2))


def _scale_volatility(segment: torch.Tensor, factor: float) -> torch.Tensor:
    mu = segment.mean(dim=0, keepdim=True)
    return mu + factor * (segment - mu)

