from __future__ import annotations

import torch
import pytest

from effectcma_flow.evaluation.metrics import compute_field_metrics, compute_metrics, compute_text2ts_metrics
from effectcma_flow.evaluation.sampler import euler_sample, euler_sample_text2ts
from effectcma_flow.models.build import build_model


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
