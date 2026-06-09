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

