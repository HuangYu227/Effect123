from __future__ import annotations

import torch


@torch.no_grad()
def euler_sample(model, base: torch.Tensor, text_condition, *, steps: int = 16) -> tuple[torch.Tensor, dict]:
    """Euler ODE sampling for edit mode (preserved for backward compatibility)."""
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
    steps: int = 64,
    noise_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    cfg_scale: float = 1.0,
    uncond_condition=None,
) -> tuple[torch.Tensor, dict]:
    """Euler sampling for text2ts. Thin wrapper around sample_text2ts."""
    return sample_text2ts(
        model,
        shape_like,
        text_condition,
        solver="euler",
        steps=steps,
        noise_scale=noise_scale,
        noise=noise,
        generator=generator,
        cfg_scale=cfg_scale,
        uncond_condition=uncond_condition,
    )


@torch.no_grad()
def rk4_sample_text2ts(
    model,
    shape_like: torch.Tensor,
    text_condition,
    *,
    steps: int = 64,
    noise_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    cfg_scale: float = 1.0,
    uncond_condition=None,
) -> tuple[torch.Tensor, dict]:
    """RK4 sampling for text2ts. Thin wrapper around sample_text2ts."""
    return sample_text2ts(
        model,
        shape_like,
        text_condition,
        solver="rk4",
        steps=steps,
        noise_scale=noise_scale,
        noise=noise,
        generator=generator,
        cfg_scale=cfg_scale,
        uncond_condition=uncond_condition,
    )


@torch.no_grad()
def sample_text2ts(
    model,
    shape_like: torch.Tensor,
    text_condition,
    *,
    solver: str = "rk4",
    steps: int = 64,
    noise_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    return_trajectory: bool = False,
    cfg_scale: float = 1.0,
    uncond_condition=None,
    guidance_t_lo: float = 0.0,
    guidance_t_hi: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Unified text2ts sampling with configurable ODE solver.

    Args:
        model: TextToTSFlow or compatible model.
        shape_like: Tensor of shape [B, L, C] defining output shape.
        text_condition: Text encoder output (slots, dict, etc.).
        solver: One of "euler", "midpoint", "rk4".
        steps: Number of integration steps (default 64).
        noise_scale: Scale factor for initial noise.
        noise: Optional pre-sampled noise tensor.
        generator: Optional RNG for reproducibility.
        return_trajectory: If True, aux includes "trajectory" list.
        cfg_scale: Classifier-free guidance scale. ``<= 1.0`` disables guidance
            (single conditional pass, identical to legacy behavior). When
            ``> 1.0`` the velocity is extrapolated as
            ``v_uncond + cfg_scale * (v_cond - v_uncond)``, sharpening how
            strongly the generated series follows the caption.
        uncond_condition: Optional explicit unconditional condition for CFG.
            Defaults to blank captions (one empty string per sample), the
            standard CFG null condition matched to training-time caption
            dropout.
        guidance_t_lo: Lower bound (inclusive) of the flow-time interval on
            which CFG extrapolation is applied. Limited-Interval Guidance:
            outside ``[guidance_t_lo, guidance_t_hi]`` the plain conditional
            velocity is used and the unconditional forward is skipped. The
            default ``0.0`` keeps guidance active for all ``t`` (no-op).
        guidance_t_hi: Upper bound (inclusive) of the guidance interval. The
            default ``1.0`` keeps guidance active for all ``t`` (no-op).

    Returns:
        (x_final, aux) tuple.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if shape_like.ndim != 3:
        raise ValueError(f"shape_like must be [B, L, C], got {tuple(shape_like.shape)}")
    solver = str(solver).lower()
    if solver not in {"euler", "midpoint", "rk4"}:
        raise ValueError(f"Unknown solver {solver!r}; expected 'euler', 'midpoint', or 'rk4'")

    x = _initial_noise(shape_like, noise_scale=noise_scale, noise=noise, generator=generator)
    batch_size = shape_like.shape[0]
    device = shape_like.device
    dtype = shape_like.dtype
    dt = 1.0 / float(steps)
    cfg_scale = float(cfg_scale)
    guidance_t_lo = float(guidance_t_lo)
    guidance_t_hi = float(guidance_t_hi)
    use_cfg = cfg_scale > 1.0
    if hasattr(model, "prepare_condition"):
        text_condition = model.prepare_condition(text_condition, device=device, dtype=dtype)
        if use_cfg:
            null_condition = uncond_condition if uncond_condition is not None else _blank_condition(batch_size)
            uncond_condition = model.prepare_condition(null_condition, device=device, dtype=dtype)
    elif use_cfg and uncond_condition is None:
        uncond_condition = _blank_condition(batch_size)

    def velocity(x_cur: torch.Tensor, t_val: float) -> tuple[torch.Tensor, dict]:
        t = torch.full((batch_size,), t_val, device=device, dtype=dtype)
        v_cond, aux_cond = model(x_cur, t, text_condition)
        apply_cfg = use_cfg and (guidance_t_lo <= t_val <= guidance_t_hi)
        if not apply_cfg:
            return v_cond, aux_cond
        v_uncond, _ = model(x_cur, t, uncond_condition)
        v_guided = v_uncond + cfg_scale * (v_cond - v_uncond)
        return v_guided, aux_cond

    aux = {}
    trajectory = []

    for n in range(steps):
        t0 = n * dt
        if return_trajectory:
            trajectory.append(x.detach().clone())

        if solver == "euler":
            v, aux = velocity(x, t0)
            x = x + dt * v

        elif solver == "midpoint":
            v1, _ = velocity(x, t0)
            x_mid = x + 0.5 * dt * v1
            v2, aux = velocity(x_mid, min(1.0, t0 + 0.5 * dt))
            x = x + dt * v2

        elif solver == "rk4":
            k1, _ = velocity(x, t0)
            k2, _ = velocity(x + 0.5 * dt * k1, min(1.0, t0 + 0.5 * dt))
            k3, _ = velocity(x + 0.5 * dt * k2, min(1.0, t0 + 0.5 * dt))
            k4, aux = velocity(x + dt * k3, min(1.0, t0 + dt))
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    aux = {**aux, "sampler_solver": solver, "sampler_steps": steps, "cfg_scale": cfg_scale}
    if return_trajectory:
        aux["trajectory"] = trajectory
    return x, aux


def _blank_condition(batch_size: int) -> list[list[str]]:
    """Standard CFG null condition: one empty caption slot per sample."""
    return [[""] for _ in range(batch_size)]


def _initial_noise(
    shape_like: torch.Tensor,
    *,
    noise_scale: float = 1.0,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if noise is not None:
        x = noise.to(shape_like.device, dtype=shape_like.dtype)
    else:
        x = torch.randn(
            shape_like.shape,
            device=shape_like.device,
            dtype=shape_like.dtype,
            generator=generator,
        )
    if x.shape != shape_like.shape:
        raise ValueError(f"noise must have shape {tuple(shape_like.shape)}, got {tuple(x.shape)}")
    return x * float(noise_scale)
