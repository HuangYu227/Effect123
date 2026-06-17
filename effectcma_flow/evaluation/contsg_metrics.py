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
import torch
from scipy import linalg

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


def mdd(real: np.ndarray, generated: np.ndarray, *, train_reference: np.ndarray | None = None, bins: int = 50, eps: float = EPS) -> float:
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
    _, T, _ = x.shape
    if max_lag is None:
        max_lag = min(T // 2, 50)
    max_lag = int(max(1, min(max_lag, T - 1)))
    centered = x - x.mean(axis=1, keepdims=True)
    denom = np.sum(centered ** 2, axis=1) + eps  # (N, C)
    prof = []
    for lag in range(1, max_lag + 1):
        num = np.sum(centered[:, :-lag, :] * centered[:, lag:, :], axis=1)  # (N, C)
        prof.append(float(np.mean(num / denom)))
    return np.asarray(prof, dtype=np.float64)


def acd(real: np.ndarray, generated: np.ndarray, *, max_lag: int | None = None) -> float:
    """Auto-Correlation Difference. Lower is better."""
    return float(np.linalg.norm(_autocorr_profile(real, max_lag) - _autocorr_profile(generated, max_lag), ord=2))


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


def frechet_distance(real_feat: np.ndarray, gen_feat: np.ndarray, *, eps: float = 1e-6) -> float:
    """Frechet distance between two feature distributions. Lower is better."""
    real_feat = _as_feat(real_feat, "real_feat")
    gen_feat = _as_feat(gen_feat, "gen_feat")
    if real_feat.shape[1] != gen_feat.shape[1]:
        raise ValueError(f"feature dims mismatch: {real_feat.shape} vs {gen_feat.shape}")
    if real_feat.shape[0] < 2 or gen_feat.shape[0] < 2:
        raise ValueError("need at least two samples per set to estimate covariance")

    mu_r = real_feat.mean(axis=0)
    mu_g = gen_feat.mean(axis=0)
    cov_r = np.cov(real_feat, rowvar=False)
    cov_g = np.cov(gen_feat, rowvar=False)
    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(cov_r.dot(cov_g), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_r.shape[0]) * eps
        covmean = linalg.sqrtm((cov_r + offset).dot(cov_g + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = float(diff.dot(diff) + np.trace(cov_r) + np.trace(cov_g) - 2.0 * np.trace(covmean))
    return float(max(value, 0.0))


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64, device=device)


def _knn_kth_distance(feats: torch.Tensor, k: int, chunk: int = 1024) -> torch.Tensor:
    """Distance from each point to its k-th nearest neighbour, excluding itself."""
    n = feats.shape[0]
    if n <= k:
        raise ValueError(f"need more than k samples, got n={n}, k={k}")
    out = torch.empty(n, dtype=feats.dtype, device=feats.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(feats[start:end], feats)  # (b, n)
        # k-th neighbour excluding self: take k+1 smallest, drop the zero self-distance.
        vals, _ = torch.topk(d, k + 1, dim=1, largest=False)
        out[start:end] = vals[:, k]
    return out


def _fraction_in_manifold(query: torch.Tensor, reference: torch.Tensor, radius: torch.Tensor, chunk: int = 1024) -> float:
    n = query.shape[0]
    inside = torch.zeros(n, dtype=torch.bool, device=query.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        d = torch.cdist(query[start:end], reference)  # (b, m)
        inside[start:end] = (d <= radius[None, :]).any(dim=1)
    return float(inside.double().mean().item())


def precision_recall(real_feat: np.ndarray, gen_feat: np.ndarray, *, k: int = 3, device: torch.device | str = "cpu", chunk: int = 1024) -> tuple[float, float]:
    """Improved Precision and Recall in feature space. Higher is better."""
    dev = torch.device(device)
    real_t = _to_tensor(_as_feat(real_feat, "real_feat"), dev)
    gen_t = _to_tensor(_as_feat(gen_feat, "gen_feat"), dev)
    r_radius = _knn_kth_distance(real_t, k, chunk)
    g_radius = _knn_kth_distance(gen_t, k, chunk)
    precision = _fraction_in_manifold(gen_t, real_t, r_radius, chunk)
    recall = _fraction_in_manifold(real_t, gen_t, g_radius, chunk)
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


def make_joint_features(ts_feat: np.ndarray, cond_feat: np.ndarray, *, normalize_parts: bool = True) -> np.ndarray:
    """Concatenate TS and condition embeddings for J-FTSD / Joint P&R."""
    ts_feat = _as_feat(ts_feat, "ts_feat")
    cond_feat = _as_feat(cond_feat, "cond_feat")
    if ts_feat.shape[0] != cond_feat.shape[0]:
        raise ValueError(f"sample count mismatch: {ts_feat.shape} vs {cond_feat.shape}")
    if normalize_parts:
        ts_feat = _l2_normalize(ts_feat)
        cond_feat = _l2_normalize(cond_feat)
    return np.concatenate([ts_feat, cond_feat], axis=1)


def j_ftsd(real_ts_feat: np.ndarray, real_cond_feat: np.ndarray, gen_ts_feat: np.ndarray, gen_cond_feat: np.ndarray, *, normalize_parts: bool = True) -> float:
    """Joint Frechet Time Series Distance. Lower is better."""
    real_joint = make_joint_features(real_ts_feat, real_cond_feat, normalize_parts=normalize_parts)
    gen_joint = make_joint_features(gen_ts_feat, gen_cond_feat, normalize_parts=normalize_parts)
    return frechet_distance(real_joint, gen_joint)


def joint_precision_recall(real_ts_feat: np.ndarray, real_cond_feat: np.ndarray, gen_ts_feat: np.ndarray, gen_cond_feat: np.ndarray, *, k: int = 3, normalize_parts: bool = True, device: torch.device | str = "cpu", chunk: int = 1024) -> tuple[float, float]:
    """Precision/Recall in the joint embedding space. Higher is better."""
    real_joint = make_joint_features(real_ts_feat, real_cond_feat, normalize_parts=normalize_parts)
    gen_joint = make_joint_features(gen_ts_feat, gen_cond_feat, normalize_parts=normalize_parts)
    return precision_recall(real_joint, gen_joint, k=k, device=device, chunk=chunk)


# ---------------------------------------------------------------------------
# Convenience aggregators
# ---------------------------------------------------------------------------


def compute_statistical_metrics(real_ts: np.ndarray, gen_ts: np.ndarray, *, train_reference: np.ndarray | None = None, bins: int = 50, max_lag: int | None = None) -> dict[str, float]:
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
    k: int = 3,
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
