from __future__ import annotations

import torch


@torch.no_grad()
def euler_sample(model, base: torch.Tensor, text_condition, *, steps: int = 16) -> tuple[torch.Tensor, dict]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    x = base.clone()
    aux = {}
    batch_size = base.shape[0]
    dt = 1.0 / float(steps)
    for n in range(steps):
        t = torch.full((batch_size,), n / float(steps), device=base.device, dtype=base.dtype)
        v, aux = model(base, x, t, text_condition)
        x = x + dt * v
    return x, aux


@torch.no_grad()
def euler_sample_text2ts(
    model,
    shape_like: torch.Tensor,
    text_condition,
    *,
    steps: int = 16,
    noise_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if shape_like.ndim != 3:
        raise ValueError(f"shape_like must be [B, L, C], got {tuple(shape_like.shape)}")
    if hasattr(model, "prepare_condition") and hasattr(model, "initial_state"):
        prepared = model.prepare_condition(text_condition, device=shape_like.device, dtype=shape_like.dtype)
        source_shape = shape_like.new_zeros(shape_like.shape)
        x = model.initial_state(source_shape, prepared, noise_scale=float(noise_scale), noise=noise, generator=generator)
        text_condition = prepared
    else:
        if noise is None:
            x = torch.randn(shape_like.shape, device=shape_like.device, dtype=shape_like.dtype, generator=generator) * float(noise_scale)
        else:
            x = noise.to(shape_like.device, dtype=shape_like.dtype)
    if x.shape != shape_like.shape:
        raise ValueError(f"noise must have shape {tuple(shape_like.shape)}, got {tuple(x.shape)}")
    aux = {}
    batch_size = shape_like.shape[0]
    dt = 1.0 / float(steps)
    for n in range(steps):
        t = torch.full((batch_size,), n / float(steps), device=shape_like.device, dtype=shape_like.dtype)
        v, aux = model(x, t, text_condition)
        x = x + dt * v
    return x, aux
