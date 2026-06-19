from __future__ import annotations

import numpy as np
import pytest

from effectcma_flow.evaluation import contsg_metrics as cm


def _rng():
    return np.random.default_rng(42)


def test_statistical_metrics_zero_on_identical_data():
    rng = _rng()
    x = rng.normal(size=(128, 24, 3))
    out = cm.compute_statistical_metrics(x, x, train_reference=x)
    for key in ("MDD", "ACD", "SD", "KD"):
        assert out[key] == pytest.approx(0.0, abs=1e-9)


def test_statistical_metrics_increase_with_perturbation():
    rng = _rng()
    real = rng.normal(size=(256, 24, 3))
    close = real + 0.05 * rng.normal(size=real.shape)
    far = real + 1.0 * rng.normal(size=real.shape)
    near = cm.compute_statistical_metrics(real, close, train_reference=real)
    away = cm.compute_statistical_metrics(real, far, train_reference=real)
    # A larger perturbation should not look more similar on the marginal metrics.
    assert away["MDD"] >= near["MDD"]
    assert away["ACD"] >= near["ACD"]


def test_acd_detects_autocorrelation_change():
    rng = _rng()
    t = np.linspace(0, 8 * np.pi, 48)
    periodic = np.stack([np.sin(t)[None] + 0.01 * rng.normal(size=(64, 48)) for _ in range(2)], axis=-1)
    noise = rng.normal(size=(64, 48, 2))
    assert cm.acd(periodic, noise) > cm.acd(periodic, periodic + 0.01 * rng.normal(size=periodic.shape))


def test_frechet_distance_zero_for_same_distribution():
    rng = _rng()
    feat = rng.normal(size=(512, 32))
    assert cm.frechet_distance(feat, feat) == pytest.approx(0.0, abs=1e-4)


def test_frechet_distance_positive_for_shifted_mean():
    rng = _rng()
    a = rng.normal(size=(512, 16))
    b = a + 3.0
    assert cm.frechet_distance(a, b) > 5.0


def test_precision_recall_perfect_overlap():
    rng = _rng()
    feat = rng.normal(size=(256, 8))
    p, r = cm.precision_recall(feat, feat, k=3)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(1.0)


def test_precision_recall_disjoint_manifolds():
    rng = _rng()
    real = rng.normal(size=(256, 8))
    gen = rng.normal(size=(256, 8)) + 50.0
    p, r = cm.precision_recall(real, gen, k=3)
    assert p < 0.05
    assert r < 0.05


def test_precision_recall_matches_bruteforce():
    rng = _rng()
    real = rng.normal(size=(120, 6))
    gen = rng.normal(size=(90, 6)) + 0.3

    def brute(ref, query, k=3):
        # ConTSG-Bench uses squared kNN radii and compares each query to its nearest ref point.
        dref = np.sum((ref[:, None] - ref[None, :]) ** 2, axis=-1)
        np.fill_diagonal(dref, np.inf)
        radius = np.sort(dref, axis=1)[:, k - 1]
        dq = np.sum((query[:, None] - ref[None, :]) ** 2, axis=-1)
        nn = np.argmin(dq, axis=1)
        return float((dq[np.arange(query.shape[0]), nn] <= radius[nn]).mean())

    rng_sample = np.random.RandomState(0)
    real_idx = rng_sample.choice(real.shape[0], size=90, replace=False)
    real_sampled = real[real_idx]

    p, r = cm.precision_recall(real, gen, k=3, seed=0)
    assert p == pytest.approx(brute(real_sampled, gen), abs=1e-9)
    assert r == pytest.approx(brute(gen, real_sampled), abs=1e-9)


def test_precision_recall_can_disable_conbench_sampling():
    rng = _rng()
    real = rng.normal(size=(120, 6))
    gen = rng.normal(size=(90, 6)) + 0.3
    sampled = cm.precision_recall(real, gen, k=3, seed=0)
    full = cm.precision_recall(real, gen, k=3, sample_equal=False)
    assert sampled != full


def test_cttp_score_bounds():
    rng = _rng()
    feat = rng.normal(size=(64, 16))
    assert cm.cttp_score(feat, feat) == pytest.approx(1.0, abs=1e-6)
    assert cm.cttp_score(feat, -feat) == pytest.approx(-1.0, abs=1e-6)


def test_j_ftsd_zero_for_identical_pairs():
    rng = _rng()
    ts = rng.normal(size=(256, 16))
    cond = rng.normal(size=(256, 16))
    assert cm.j_ftsd(ts, cond, ts, cond) == pytest.approx(0.0, abs=1e-3)


def test_embedding_panel_has_seven_keys():
    rng = _rng()
    real_ts = rng.normal(size=(200, 16))
    gen_ts = real_ts + 0.1 * rng.normal(size=real_ts.shape)
    text = rng.normal(size=(200, 16))
    out = cm.compute_embedding_metrics(
        real_ts_feat=real_ts,
        gen_ts_feat=gen_ts,
        real_cond_feat=text,
        gen_cond_feat=text,
    )
    assert set(out) == {"FID", "Precision", "Recall", "CTTPScore", "J-FTSD", "JointPrecision", "JointRecall"}


def test_shape_validation_errors():
    rng = _rng()
    with pytest.raises(ValueError):
        cm.mdd(rng.normal(size=(4, 4)), rng.normal(size=(4, 4)))  # not 3-D
    with pytest.raises(ValueError):
        cm.frechet_distance(rng.normal(size=(4, 4, 4)), rng.normal(size=(4, 4, 4)))  # not 2-D
