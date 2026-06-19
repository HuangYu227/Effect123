"""ConTSG-Bench evaluation metrics.

Implements the 11 metrics from ConTSG-Bench Appendix C:

  Statistical fidelity (raw time series, no encoder):
    MDD, ACD, SD, KD
  Embedding fidelity (time-series embedding):
    FID, Precision, Recall
  Condition adherence (joint time-series + text embedding):
    CTTPScore, J-FTSD, JointPrecision, JointRecall

Statistical metrics consume raw ``(N, T, C)`` time series. Embedding metrics
consume pre-computed ``(N, D)`` features (e.g. from the CTTP encoder). The kNN
manifold metrics (Precision/Recall and their joint variants) are implemented
with chunked ``torch.cdist`` so no scikit-learn dependency is needed and the
work can optionally run on GPU.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
import torch

EPS = 1e-8

# Direction table so callers can normalise / report consistently.
LOWER_IS_BETTER = ("MDD", "ACD", "SD", "KD", "FID", "J-FTSD")
HIGHER_IS_BETTER = ("Precision", "Recall", "JointPrecision", "JointRecall", "CTTPScore")


def _as_ntc(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"expected (N, T, C) time series, got {arr.shape}")
    return arr


def _as_feat(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (num_samples, dim), got {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Statistical fidelity (raw time series)
# ---------------------------------------------------------------------------


def mdd(real: np.ndarray, generated: np.ndarray, *, train_reference: np.ndarray | None = None, bins: int = 32, eps: float = EPS) -> float:
    """Marginal Distribution Difference. Lower is better.

    Histogram range is taken from ``train_reference`` (the training set when
    available) so generated outliers cannot shift the bins. Generated/real
    values are clipped into that range before binning.
    """
    real = _as_ntc(real)
    generated = _as_ntc(generated)
    ref = _as_ntc(train_reference) if train_reference is not None else real
    if real.shape[1:] != generated.shape[1:]:
        raise ValueError(f"real and generated must share (T, C), got {real.shape} vs {generated.shape}")

    _, T, C = real.shape
    total = 0.0
    count = 0
    for t in range(T):
        for c in range(C):
            lo = float(np.min(ref[:, t, c]))
            hi = float(np.max(ref[:, t, c]))
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if hi - lo < eps:
                lo -= 0.5
                hi += 0.5
            edges = np.linspace(lo, hi, bins + 1)
            r_vals = np.clip(real[:, t, c], lo, hi)
            g_vals = np.clip(generated[:, t, c], lo, hi)
            p_r, _ = np.histogram(r_vals, bins=edges)
            p_g, _ = np.histogram(g_vals, bins=edges)
            p_r = p_r.astype(np.float64) / max(p_r.sum(), 1)
            p_g = p_g.astype(np.float64) / max(p_g.sum(), 1)
            total += float(np.mean(np.abs(p_r - p_g)))
            count += 1
    return float(total / max(count, 1))


def _autocorr_profile(x: np.ndarray, max_lag: int | None, eps: float = EPS) -> np.ndarray:
    x = _as_ntc(x)
    _, T, C = x.shape
    if max_lag is None:
        max_lag = 50
    max_lag = int(max(0, min(max_lag, T - 1)))
    centered = x - x.mean(axis=1, keepdims=True)
    denom = np.sum(centered ** 2, axis=1) + eps  # (N, C)
    prof = np.zeros((C, max_lag + 1), dtype=np.float64)
    for lag in range(0, max_lag + 1):
        if lag == 0:
            num = np.sum(centered ** 2, axis=1)  # (N, C)
        else:
            num = np.sum(centered[:, :-lag, :] * centered[:, lag:, :], axis=1)  # (N, C)
        prof[:, lag] = np.mean(num / denom, axis=0)
    return prof


def acd(real: np.ndarray, generated: np.ndarray, *, max_lag: int | None = None) -> float:
    """Auto-Correlation Difference. Lower is better."""
    return float(np.mean(np.abs(_autocorr_profile(real, max_lag) - _autocorr_profile(generated, max_lag))))


def _flatten_channels(x: np.ndarray) -> np.ndarray:
    arr = _as_ntc(x)
    return arr.reshape(-1, arr.shape[-1])


def _standardized_moment(x: np.ndarray, power: int, eps: float = EPS) -> np.ndarray:
    z = _flatten_channels(x)
    mu = z.mean(axis=0)
    std = z.std(axis=0) + eps
    return np.mean(((z - mu) / std) ** power, axis=0)


def sd(real: np.ndarray, generated: np.ndarray) -> float:
    """Skewness Difference. Lower is better."""
    return float(np.mean(np.abs(_standardized_moment(real, 3) - _standardized_moment(generated, 3))))


def kd(real: np.ndarray, generated: np.ndarray) -> float:
    """Kurtosis Difference (Pearson; normal ~= 3). Lower is better."""
    return float(np.mean(np.abs(_standardized_moment(real, 4) - _standardized_moment(generated, 4))))


# ---------------------------------------------------------------------------
# Embedding fidelity
# ---------------------------------------------------------------------------


def _covariance(feat: np.ndarray) -> np.ndarray:
    feat = _as_feat(feat, "feat")
    centered = feat - feat.mean(axis=0, keepdims=True)
    cov = np.zeros((feat.shape[1], feat.shape[1]), dtype=np.float64)
    for start in range(0, feat.shape[0], 4096):
        block = centered[start : start + 4096]
        cov += np.einsum("ni,nj->ij", block, block, optimize=False)
    return cov / float(feat.shape[0] - 1)


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ik,kj->ij", a, b, optimize=False)


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    sym = 0.5 * (matrix + matrix.T)
    tensor = torch.as_tensor(sym, dtype=torch.float64, device="cpu")
    eigvals, eigvecs = torch.linalg.eigh(tensor)
    eigvals = torch.clamp(eigvals, min=0.0)
    sqrt_tensor = (eigvecs * torch.sqrt(eigvals).unsqueeze(0)) @ eigvecs.T
    return sqrt_tensor.numpy()


def frechet_distance(real_feat: np.ndarray, gen_feat: np.ndarray, *, eps: float = 1e-6, reg: float = 1e-4) -> float:
    """Frechet distance between two feature distributions. Lower is better."""
    real_feat = _as_feat(real_feat, "real_feat")
    gen_feat = _as_feat(gen_feat, "gen_feat")
    if real_feat.shape[1] != gen_feat.shape[1]:
        raise ValueError(f"feature dims mismatch: {real_feat.shape} vs {gen_feat.shape}")
    if real_feat.shape[0] < 2 or gen_feat.shape[0] < 2:
        raise ValueError("need at least two samples per set to estimate covariance")

    mu_r = real_feat.mean(axis=0)
    mu_g = gen_feat.mean(axis=0)
    cov_r = _covariance(real_feat)
    cov_g = _covariance(gen_feat)
    if reg > 0:
        cov_r = cov_r + np.eye(cov_r.shape[0]) * reg
        cov_g = cov_g + np.eye(cov_g.shape[0]) * reg
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(cov_r.dot(cov_g), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_r.shape[0]) * eps
        covmean = linalg.sqrtm((cov_r + offset).dot(cov_g + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(np.sum(diff * diff) + np.trace(cov_r) + np.trace(cov_g) - 2.0 * np.trace(covmean))
    return float(max(value, 0.0))


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=device)


def _knn_kth_distance(feats: torch.Tensor, k: int, chunk: int = 1024) -> torch.Tensor:
    """Squared distance to each point's k-th nearest neighbour, excluding itself."""
    n = feats.shape[0]
    if n <= k:
        raise ValueError(f"need more than k samples, got n={n}, k={k}")
    out = torch.empty(n, dtype=feats.dtype, device=feats.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(feats[start:end], feats).pow(2)  # (b, n)
        rows = torch.arange(end - start, device=feats.device)
        cols = torch.arange(start, end, device=feats.device)
        d[rows, cols] = float("inf")
        vals, _ = torch.topk(d, k, dim=1, largest=False)
        out[start:end] = vals[:, -1]
    return out


def _cross_1nn(query: torch.Tensor, reference: torch.Tensor, chunk: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    n = query.shape[0]
    min_dist = torch.empty(n, dtype=query.dtype, device=query.device)
    min_idx = torch.empty(n, dtype=torch.long, device=query.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(query[start:end], reference).pow(2)  # (b, m)
        vals, idx = torch.min(d, dim=1)
        min_dist[start:end] = vals
        min_idx[start:end] = idx
    return min_dist, min_idx


def precision_recall(
    real_feat: np.ndarray,
    gen_feat: np.ndarray,
    *,
    k: int = 5,
    max_samples: int | None = None,
    seed: int = 0,
    sample_equal: bool = True,
    device: torch.device | str = "cpu",
    chunk: int = 1024,
) -> tuple[float, float]:
    """ConTSG-Bench kNN Precision and Recall in feature space. Higher is better."""
    real_feat, gen_feat = _prepare_prdc_features(
        real_feat,
        gen_feat,
        max_samples=max_samples,
        seed=seed,
        sample_equal=sample_equal,
    )
    dev = torch.device(device)
    real_t = _to_tensor(real_feat, dev)
    gen_t = _to_tensor(gen_feat, dev)
    k_eff = min(int(k), real_t.shape[0] - 1, gen_t.shape[0] - 1)
    if k_eff < 1:
        raise ValueError(f"need at least two samples per set, got real={real_t.shape[0]}, gen={gen_t.shape[0]}")
    r_radius = _knn_kth_distance(real_t, k_eff, chunk)
    g_radius = _knn_kth_distance(gen_t, k_eff, chunk)
    d_g2r, nn_g2r = _cross_1nn(gen_t, real_t, chunk)
    d_r2g, nn_r2g = _cross_1nn(real_t, gen_t, chunk)
    precision = float((d_g2r <= r_radius[nn_g2r]).double().mean().item())
    recall = float((d_r2g <= g_radius[nn_r2g]).double().mean().item())
    return precision, recall


# ---------------------------------------------------------------------------
# Condition adherence
# ---------------------------------------------------------------------------


def _l2_normalize(z: np.ndarray, eps: float = EPS) -> np.ndarray:
    z = _as_feat(z, "features")
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + eps)


def cttp_score(gen_ts_feat: np.ndarray, cond_text_feat: np.ndarray, *, normalize: bool = True) -> float:
    """Mean paired cosine similarity between TS and text embeddings. Higher is better."""
    gen_ts_feat = _as_feat(gen_ts_feat, "gen_ts_feat")
    cond_text_feat = _as_feat(cond_text_feat, "cond_text_feat")
    if gen_ts_feat.shape != cond_text_feat.shape:
        raise ValueError(f"shape mismatch: {gen_ts_feat.shape} vs {cond_text_feat.shape}")
    if normalize:
        gen_ts_feat = _l2_normalize(gen_ts_feat)
        cond_text_feat = _l2_normalize(cond_text_feat)
    return float(np.mean(np.sum(gen_ts_feat * cond_text_feat, axis=1)))


def make_joint_features(ts_feat: np.ndarray, cond_feat: np.ndarray, *, normalize_parts: bool = False) -> np.ndarray:
    """Concatenate TS and condition embeddings for J-FTSD / Joint P&R."""
    ts_feat = _as_feat(ts_feat, "ts_feat")
    cond_feat = _as_feat(cond_feat, "cond_feat")
    if ts_feat.shape[0] != cond_feat.shape[0]:
        raise ValueError(f"sample count mismatch: {ts_feat.shape} vs {cond_feat.shape}")
    if normalize_parts:
        ts_feat = _l2_normalize(ts_feat)
        cond_feat = _l2_normalize(cond_feat)
    return np.concatenate([ts_feat, cond_feat], axis=1)


def make_conbench_joint_features(
    real_ts_feat: np.ndarray,
    real_cond_feat: np.ndarray,
    gen_ts_feat: np.ndarray,
    gen_cond_feat: np.ndarray,
    *,
    ts_weight: float = 1.0,
    text_weight: float = 1.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ConTSG-Bench joint features for JointPrecision/JointRecall."""
    real_ts_feat = _as_feat(real_ts_feat, "real_ts_feat")
    real_cond_feat = _as_feat(real_cond_feat, "real_cond_feat")
    gen_ts_feat = _as_feat(gen_ts_feat, "gen_ts_feat")
    gen_cond_feat = _as_feat(gen_cond_feat, "gen_cond_feat")
    if real_ts_feat.shape[0] != real_cond_feat.shape[0]:
        raise ValueError(f"real sample count mismatch: {real_ts_feat.shape} vs {real_cond_feat.shape}")
    if gen_ts_feat.shape[0] != gen_cond_feat.shape[0]:
        raise ValueError(f"generated sample count mismatch: {gen_ts_feat.shape} vs {gen_cond_feat.shape}")
    if real_ts_feat.shape[1] != gen_ts_feat.shape[1]:
        raise ValueError(f"TS feature dims mismatch: {real_ts_feat.shape} vs {gen_ts_feat.shape}")
    if real_cond_feat.shape[1] != gen_cond_feat.shape[1]:
        raise ValueError(f"text feature dims mismatch: {real_cond_feat.shape} vs {gen_cond_feat.shape}")

    all_ts = np.concatenate([gen_ts_feat, real_ts_feat], axis=0)
    all_text = np.concatenate([gen_cond_feat, real_cond_feat], axis=0)
    mu_ts = all_ts.mean(axis=0)
    std_ts = all_ts.std(axis=0)
    mu_text = all_text.mean(axis=0)
    std_text = all_text.std(axis=0)

    real_ts_n = ((real_ts_feat - mu_ts) / (std_ts + eps)) * ts_weight
    gen_ts_n = ((gen_ts_feat - mu_ts) / (std_ts + eps)) * ts_weight
    real_text_n = ((real_cond_feat - mu_text) / (std_text + eps)) * text_weight
    gen_text_n = ((gen_cond_feat - mu_text) / (std_text + eps)) * text_weight
    return np.concatenate([real_ts_n, real_text_n], axis=1), np.concatenate([gen_ts_n, gen_text_n], axis=1)


def j_ftsd(real_ts_feat: np.ndarray, real_cond_feat: np.ndarray, gen_ts_feat: np.ndarray, gen_cond_feat: np.ndarray, *, normalize_parts: bool = False) -> float:
    """Joint Frechet Time Series Distance. Lower is better."""
    real_joint = make_joint_features(real_ts_feat, real_cond_feat, normalize_parts=normalize_parts)
    gen_joint = make_joint_features(gen_ts_feat, gen_cond_feat, normalize_parts=normalize_parts)
    return frechet_distance(real_joint, gen_joint)


def joint_precision_recall(
    real_ts_feat: np.ndarray,
    real_cond_feat: np.ndarray,
    gen_ts_feat: np.ndarray,
    gen_cond_feat: np.ndarray,
    *,
    k: int = 5,
    max_samples: int | None = None,
    seed: int = 0,
    sample_equal: bool = True,
    device: torch.device | str = "cpu",
    chunk: int = 1024,
) -> tuple[float, float]:
    """ConTSG-Bench Joint Precision/Recall with per-modality standardization."""
    real_joint, gen_joint = make_conbench_joint_features(real_ts_feat, real_cond_feat, gen_ts_feat, gen_cond_feat)
    return precision_recall(
        real_joint,
        gen_joint,
        k=k,
        max_samples=max_samples,
        seed=seed,
        sample_equal=sample_equal,
        device=device,
        chunk=chunk,
    )


def _prepare_prdc_features(
    real_feat: np.ndarray,
    gen_feat: np.ndarray,
    *,
    max_samples: int | None,
    seed: int,
    sample_equal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    real = _filter_finite_rows(_as_feat(real_feat, "real_feat"))
    gen = _filter_finite_rows(_as_feat(gen_feat, "gen_feat"))
    if real.shape[1] != gen.shape[1]:
        raise ValueError(f"feature dims mismatch: {real.shape} vs {gen.shape}")
    if real.shape[0] < 2 or gen.shape[0] < 2:
        raise ValueError(f"need at least two samples per set, got real={real.shape[0]}, gen={gen.shape[0]}")
    if not sample_equal:
        return real, gen
    n_cap = int(max_samples) if max_samples is not None else min(real.shape[0], gen.shape[0])
    n = min(real.shape[0], gen.shape[0], n_cap)
    if n < 2:
        raise ValueError(f"need at least two sampled points, got n={n}")
    return real[_sample_indices(real.shape[0], n, seed)], gen[_sample_indices(gen.shape[0], n, seed)]


def _filter_finite_rows(x: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x).all(axis=1)
    return x[mask]


def _sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if k >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(int(seed))
    return rng.choice(n, size=int(k), replace=False)


# ---------------------------------------------------------------------------
# Convenience aggregators
# ---------------------------------------------------------------------------


def compute_statistical_metrics(real_ts: np.ndarray, gen_ts: np.ndarray, *, train_reference: np.ndarray | None = None, bins: int = 32, max_lag: int | None = None) -> dict[str, float]:
    """The four encoder-free statistical metrics (cheap enough for training)."""
    return {
        "MDD": mdd(real_ts, gen_ts, train_reference=train_reference, bins=bins),
        "ACD": acd(real_ts, gen_ts, max_lag=max_lag),
        "SD": sd(real_ts, gen_ts),
        "KD": kd(real_ts, gen_ts),
    }


def compute_embedding_metrics(
    *,
    real_ts_feat: np.ndarray,
    gen_ts_feat: np.ndarray,
    real_cond_feat: np.ndarray,
    gen_cond_feat: np.ndarray,
    k: int = 5,
    device: torch.device | str = "cpu",
    chunk: int = 1024,
) -> dict[str, float]:
    """The seven encoder-based metrics (FID/P/R, CTTP, J-FTSD/JointP/JointR)."""
    out: dict[str, float] = {}
    out["FID"] = frechet_distance(real_ts_feat, gen_ts_feat)
    p, r = precision_recall(real_ts_feat, gen_ts_feat, k=k, device=device, chunk=chunk)
    out["Precision"] = p
    out["Recall"] = r
    out["CTTPScore"] = cttp_score(gen_ts_feat, gen_cond_feat)
    out["J-FTSD"] = j_ftsd(real_ts_feat, real_cond_feat, gen_ts_feat, gen_cond_feat)
    jp, jr = joint_precision_recall(real_ts_feat, real_cond_feat, gen_ts_feat, gen_cond_feat, k=k, device=device, chunk=chunk)
    out["JointPrecision"] = jp
    out["JointRecall"] = jr
    return out
