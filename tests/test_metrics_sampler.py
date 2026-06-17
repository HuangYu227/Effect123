from __future__ import annotations

import torch
import pytest

from effectcma_flow.evaluation.metrics import compute_field_metrics, compute_metrics, compute_text2ts_metrics
from effectcma_flow.evaluation.sampler import euler_sample, euler_sample_text2ts, sample_text2ts
from effectcma_flow.models.build import build_model


def _text2ts_smoke_model():
    cfg = {
        "task": {"mode": "text2ts"},
        "model": {
            "d_model": 16,
            "patch_len": 3,
            "num_operators": 3,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "operator_hidden": 8,
            "operator_t_dim": 4,
        },
        "text_encoder": {"mode": "hash", "hash_dim": 32},
    }
    return build_model(cfg, sequence_length=12, num_channels=4)


def test_metrics_catch_global_leakage():
    base = torch.zeros(1, 6, 3)
    target = torch.zeros_like(base)
    target[:, 2:4, 1] = 1.0
    pred = torch.ones_like(base) * 0.2
    mask = torch.zeros_like(base)
    mask[:, 2:4, 1] = 1.0
    metrics = compute_metrics(pred, target, base, mask, threshold=0.05)
    assert metrics["leakage"] > 0.5
    assert metrics["mask_precision"] < 0.5


def test_metrics_reject_bad_mask_shape_and_values():
    base = torch.zeros(1, 6, 3)
    with pytest.raises(ValueError, match="identical shapes"):
        compute_metrics(base, base, base, torch.zeros(6, 3))
    bad_mask = torch.zeros_like(base)
    bad_mask[:, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        compute_metrics(base, base, base, bad_mask)


def test_field_metrics_report_localization():
    g = torch.zeros(1, 6, 3, 2)
    g[:, 2:4, 1, 0] = 1.0
    mask = torch.zeros(1, 6, 3)
    mask[:, 2:4, 1] = 1.0
    aux = {
        "G": g,
        "A_c": torch.tensor([[[0.1, 0.8, 0.1]]]),
        "A_o": torch.tensor([[[0.9, 0.1]]]),
    }
    metrics = compute_field_metrics(aux, mask, [{"start": 2, "end": 4, "channels": [1]}])
    assert metrics["field_scope_precision"] == pytest.approx(1.0)
    assert metrics["channel_top1_acc"] == pytest.approx(1.0)
    assert metrics["time_center_mae"] == pytest.approx(0.0)


def test_euler_sampler_smoke():
    cfg = {
        "task": {"mode": "edit"},
        "model": {
            "d_model": 16,
            "patch_len": 3,
            "num_operators": 3,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "operator_hidden": 8,
            "operator_t_dim": 4,
        },
        "text_encoder": {"mode": "hash", "hash_dim": 32},
    }
    model = build_model(cfg, sequence_length=12, num_channels=4)
    base = torch.randn(2, 12, 4)
    pred1, _ = euler_sample(model, base, [["slot"], ["slot"]], steps=1)
    pred4, _ = euler_sample(model, base, [["slot"], ["slot"]], steps=4)
    assert pred1.shape == base.shape
    assert pred4.shape == base.shape
    assert torch.isfinite(pred4).all()


def test_text2ts_sampler_smoke_and_metrics():
    cfg = {
        "task": {"mode": "text2ts"},
        "model": {
            "d_model": 16,
            "patch_len": 3,
            "num_operators": 3,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "operator_hidden": 8,
            "operator_t_dim": 4,
        },
        "text_encoder": {"mode": "hash", "hash_dim": 32},
    }
    model = build_model(cfg, sequence_length=12, num_channels=4)
    target = torch.randn(2, 12, 4)
    pred1, _ = euler_sample_text2ts(model, target, [["caption"], ["caption"]], steps=1)
    pred4, _ = euler_sample_text2ts(model, target, [["caption"], ["caption"]], steps=4)
    assert pred1.shape == target.shape
    assert pred4.shape == target.shape
    assert torch.isfinite(pred4).all()
    metrics = compute_text2ts_metrics(pred4, target)
    assert set(metrics) == {"mae", "mse", "pred_std", "target_std"}


class _DeterministicCFGModel(torch.nn.Module):
    """Minimal text2ts model with an exactly reproducible velocity field.

    The velocity is a fixed per-condition constant so the CFG arithmetic
    ``v_uncond + s * (v_cond - v_uncond)`` can be verified bit-exactly without
    fighting transformer/nested-tensor nondeterminism. Each distinct caption
    string maps to a deterministic constant velocity; the blank caption "" used
    by the default CFG null condition maps to zero.
    """

    def prepare_condition(self, text_condition, *, device, dtype):
        if isinstance(text_condition, dict) and text_condition.get("_prepared", False):
            return text_condition
        captions = [slots[0] for slots in text_condition]
        return {"_prepared": True, "captions": captions}

    @staticmethod
    def _value(caption: str) -> float:
        return 0.0 if caption == "" else (len(caption) % 7) + 1.0

    def forward(self, x_t, t, text_condition):
        captions = text_condition["captions"]
        values = torch.tensor([self._value(c) for c in captions], device=x_t.device, dtype=x_t.dtype)
        v = values[:, None, None].expand_as(x_t)
        return v, {"text_context": v.mean(dim=(1, 2))}


def test_cfg_scale_one_is_exact_noop():
    """cfg_scale <= 1.0 must take exactly the single conditional pass."""
    model = _DeterministicCFGModel()
    shape = torch.zeros(2, 8, 3)
    noise = torch.zeros(2, 8, 3)  # zero noise → output equals integral of velocity
    captions = [["sunny"], ["rainy day ahead"]]
    base, aux = sample_text2ts(model, shape, captions, solver="euler", steps=4, noise=noise, cfg_scale=1.0)
    # Euler from x0=0 with constant v over t in [0,1): x = v (since sum(dt)=1).
    expected = torch.tensor([model._value("sunny"), model._value("rainy day ahead")])
    for b in range(2):
        assert torch.allclose(base[b], torch.full((8, 3), float(expected[b])), atol=1e-5)
    assert aux["cfg_scale"] == 1.0


def test_cfg_guidance_arithmetic_is_exact():
    """Guided velocity must equal v_uncond + s*(v_cond - v_uncond) exactly.

    Default null condition is the blank caption (value 0), so with s and a
    conditional constant v_cond the guided constant is s * v_cond, integrated
    over t in [0,1) to give s * v_cond at the endpoint.
    """
    model = _DeterministicCFGModel()
    shape = torch.zeros(2, 8, 3)
    noise = torch.zeros(2, 8, 3)
    captions = [["sunny"], ["rainy day ahead"]]
    s = 3.0
    guided, aux = sample_text2ts(model, shape, captions, solver="euler", steps=4, noise=noise, cfg_scale=s)
    v_cond = torch.tensor([model._value("sunny"), model._value("rainy day ahead")])
    expected = s * v_cond  # v_uncond = 0 for blank caption
    for b in range(2):
        assert torch.allclose(guided[b], torch.full((8, 3), float(expected[b])), atol=1e-5)
    assert aux["cfg_scale"] == s


def test_cfg_explicit_uncond_condition_is_used():
    """An explicit uncond_condition must override the default blank null."""
    model = _DeterministicCFGModel()
    shape = torch.zeros(1, 8, 3)
    noise = torch.zeros(1, 8, 3)
    s = 2.0
    guided, _ = sample_text2ts(
        model,
        shape,
        [["sunny"]],
        solver="euler",
        steps=4,
        noise=noise,
        cfg_scale=s,
        uncond_condition=[["rainy day ahead"]],
    )
    v_cond = model._value("sunny")
    v_uncond = model._value("rainy day ahead")
    expected = v_uncond + s * (v_cond - v_uncond)
    assert torch.allclose(guided[0], torch.full((8, 3), float(expected)), atol=1e-5)


def test_lig_default_interval_is_exact_noop():
    """guidance_t_lo=0.0, guidance_t_hi=1.0 must be bit-identical to omitting them."""
    model = _DeterministicCFGModel()
    shape = torch.zeros(2, 8, 3)
    noise = torch.zeros(2, 8, 3)
    captions = [["sunny"], ["rainy day ahead"]]
    s = 3.0
    baseline, _ = sample_text2ts(model, shape, captions, solver="euler", steps=4, noise=noise.clone(), cfg_scale=s)
    with_interval, _ = sample_text2ts(
        model,
        shape,
        captions,
        solver="euler",
        steps=4,
        noise=noise.clone(),
        cfg_scale=s,
        guidance_t_lo=0.0,
        guidance_t_hi=1.0,
    )
    assert torch.equal(baseline, with_interval)


def test_lig_narrow_interval_gates_per_step():
    """A narrow interval extrapolates only at t inside [lo, hi]; outside falls back to v_cond.

    Euler with steps=4 gives per-step times t0 in {0, 0.25, 0.5, 0.75}. With the
    interval [0.5, 1.0]: t in {0, 0.25} fall back to the plain conditional velocity
    and t in {0.5, 0.75} are extrapolated to v_uncond + s*(v_cond - v_uncond). The
    velocity field is a per-condition constant, so the Euler integral is
    dt * sum over steps of the (gated) velocity.
    """
    model = _DeterministicCFGModel()
    shape = torch.zeros(1, 8, 3)
    noise = torch.zeros(1, 8, 3)
    s = 3.0
    dt = 0.25
    guided, _ = sample_text2ts(
        model,
        shape,
        [["sunny"]],
        solver="euler",
        steps=4,
        noise=noise,
        cfg_scale=s,
        guidance_t_lo=0.5,
        guidance_t_hi=1.0,
    )
    v_cond = model._value("sunny")  # blank null → v_uncond = 0
    v_guided = 0.0 + s * (v_cond - 0.0)
    # t0 in {0, 0.25} → fallback v_cond; t0 in {0.5, 0.75} → extrapolated v_guided.
    expected = dt * (v_cond + v_cond + v_guided + v_guided)
    assert torch.allclose(guided[0], torch.full((8, 3), float(expected)), atol=1e-5)

    # Sanity: guiding all t (the no-op interval) integrates purely to s * v_cond.
    guided_all, _ = sample_text2ts(
        model,
        shape,
        [["sunny"]],
        solver="euler",
        steps=4,
        noise=noise,
        cfg_scale=s,
        guidance_t_lo=0.0,
        guidance_t_hi=1.0,
    )
    assert torch.allclose(guided_all[0], torch.full((8, 3), float(s * v_cond)), atol=1e-5)


def test_cfg_real_model_runs_for_all_solvers():
    """Smoke: CFG path is wired through every solver on the real model."""
    model = _text2ts_smoke_model()
    target = torch.zeros(2, 12, 4)
    noise = torch.randn(2, 12, 4)
    captions = [["rising temperature"], ["calm dry spell"]]
    for solver in ("euler", "midpoint", "rk4"):
        pred, aux = sample_text2ts(model, target, captions, solver=solver, steps=2, noise=noise.clone(), cfg_scale=2.0)
        assert pred.shape == target.shape
        assert torch.isfinite(pred).all()
        assert aux["cfg_scale"] == 2.0
