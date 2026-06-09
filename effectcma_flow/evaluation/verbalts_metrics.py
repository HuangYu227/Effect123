from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy import linalg
from tqdm import tqdm

from effectcma_flow.evaluation.sampler import euler_sample
from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
    """Frechet distance used by VerbalTS for FID/JFTSD."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape:
        raise ValueError(f"mean vectors have different shapes: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise ValueError(f"covariances have different shapes: {sigma1.shape} vs {sigma2.shape}")

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))


class VerbalTSMetricComputer:
    """Optional VerbalTS evaluator metrics for EffectCMA generated trajectories.

    The metric definitions follow VerbalTS:
    - FID: Frechet distance between CTTP time-series embeddings.
    - JFTSD: Frechet distance between joint [time-series, text] embeddings.
    - CTTP: average paired similarity between generated time-series and text embeddings.
    """

    def __init__(
        self,
        *,
        verbalts_root: str | Path,
        clip_config_path: str | Path,
        clip_model_path: str | Path,
        device: torch.device,
        stats: dict[str, torch.Tensor] | None,
    ) -> None:
        self.device = device
        self.stats = stats
        self.clip = _load_verbalts_cttp(
            verbalts_root=verbalts_root,
            clip_config_path=clip_config_path,
            clip_model_path=clip_model_path,
            device=device,
        )

    @torch.no_grad()
    def compute(
        self,
        *,
        model: torch.nn.Module,
        reference_loader,
        generated_loader,
        text_encoder_mode: str,
        steps: int,
        reference_max_batches: int | None = None,
        generated_max_batches: int | None = None,
        cache_dir: str | Path | None = None,
    ) -> dict[str, float]:
        reference_cache_dir = cache_dir if reference_max_batches is None else None
        ref_stats = self._load_reference_stats(reference_cache_dir)
        if ref_stats is None:
            ref_ts, ref_joint = self._collect_reference_embeddings(reference_loader, reference_max_batches)
            ref_stats = {
                "ts_mean": _mean_cov(ref_ts)[0],
                "ts_cov": _mean_cov(ref_ts)[1],
                "joint_mean": _mean_cov(ref_joint)[0],
                "joint_cov": _mean_cov(ref_joint)[1],
                "reference_count": float(ref_ts.shape[0]),
            }
            self._save_reference_stats(reference_cache_dir, ref_stats)

        gen_ts, gen_joint, cttp = self._collect_generated_embeddings(
            model=model,
            loader=generated_loader,
            text_encoder_mode=text_encoder_mode,
            steps=steps,
            max_batches=generated_max_batches,
        )
        gen_ts_mean, gen_ts_cov = _mean_cov(gen_ts)
        gen_joint_mean, gen_joint_cov = _mean_cov(gen_joint)
        return {
            "verbalts_fid": calculate_frechet_distance(ref_stats["ts_mean"], ref_stats["ts_cov"], gen_ts_mean, gen_ts_cov),
            "verbalts_jftsd": calculate_frechet_distance(ref_stats["joint_mean"], ref_stats["joint_cov"], gen_joint_mean, gen_joint_cov),
            "verbalts_cttp": float(cttp),
            "verbalts_reference_count": float(ref_stats["reference_count"]),
            "verbalts_generated_count": float(gen_ts.shape[0]),
        }

    @torch.no_grad()
    def _collect_reference_embeddings(self, loader, max_batches: int | None) -> tuple[np.ndarray, np.ndarray]:
        ts_embeddings = []
        joint_embeddings = []
        for batch_no, batch in enumerate(tqdm(loader, desc="verbalts-ref", dynamic_ncols=True)):
            if max_batches is not None and batch_no >= max_batches:
                break
            target = self._denormalize(batch["Y"].to(self.device).float())
            text = [str(x) for x in batch["full_text"]]
            ts_emb, text_emb = self._embed(target, text)
            ts_embeddings.append(ts_emb.cpu())
            joint_embeddings.append(torch.cat([ts_emb, text_emb], dim=-1).cpu())
        return _cat_numpy(ts_embeddings, "reference time-series embeddings"), _cat_numpy(joint_embeddings, "reference joint embeddings")

    @torch.no_grad()
    def _collect_generated_embeddings(
        self,
        *,
        model: torch.nn.Module,
        loader,
        text_encoder_mode: str,
        steps: int,
        max_batches: int | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        model.eval()
        ts_embeddings = []
        joint_embeddings = []
        cttp_sum = 0.0
        count = 0
        for batch_no, batch in enumerate(tqdm(loader, desc="verbalts-gen", dynamic_ncols=True)):
            if max_batches is not None and batch_no >= max_batches:
                break
            batch = batch_to_device(batch, self.device)
            pred, _ = euler_sample(model, batch["B"], text_condition_from_batch(batch, text_encoder_mode), steps=steps)
            pred = self._denormalize(pred.float())
            text = [str(x) for x in batch["full_text"]]
            ts_emb, text_emb = self._embed(pred, text)
            ts_embeddings.append(ts_emb.cpu())
            joint_embeddings.append(torch.cat([ts_emb, text_emb], dim=-1).cpu())
            cttp_sum += (ts_emb * text_emb).sum(dim=-1).sum().item()
            count += int(ts_emb.shape[0])
        if count == 0:
            raise ValueError("No generated batches were evaluated for VerbalTS metrics")
        return _cat_numpy(ts_embeddings, "generated time-series embeddings"), _cat_numpy(joint_embeddings, "generated joint embeddings"), cttp_sum / count

    def _embed(self, ts: torch.Tensor, text: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ts_len = torch.full((ts.shape[0],), ts.shape[1], device=self.device, dtype=torch.int32)
        ts_emb = self.clip.get_ts_coemb(ts, ts_len)
        text_emb = self.clip.get_text_coemb(text, None)
        return ts_emb, text_emb

    def _denormalize(self, ts: torch.Tensor) -> torch.Tensor:
        if self.stats is None:
            return ts
        mean = self.stats["mean"].to(ts.device, dtype=ts.dtype)
        std = self.stats["std"].to(ts.device, dtype=ts.dtype)
        return ts * std + mean

    def _load_reference_stats(self, cache_dir: str | Path | None) -> dict[str, Any] | None:
        if cache_dir is None:
            return None
        cache = Path(cache_dir)
        files = {
            "ts_mean": cache / "verbalts_ref_ts_mean.npy",
            "ts_cov": cache / "verbalts_ref_ts_cov.npy",
            "joint_mean": cache / "verbalts_ref_joint_mean.npy",
            "joint_cov": cache / "verbalts_ref_joint_cov.npy",
        }
        count_path = cache / "verbalts_ref_count.npy"
        if not all(path.exists() for path in files.values()) or not count_path.exists():
            return None
        out = {key: np.load(path, allow_pickle=False) for key, path in files.items()}
        out["reference_count"] = float(np.load(count_path, allow_pickle=False).reshape(-1)[0])
        return out

    def _save_reference_stats(self, cache_dir: str | Path | None, stats: dict[str, Any]) -> None:
        if cache_dir is None:
            return
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        np.save(cache / "verbalts_ref_ts_mean.npy", stats["ts_mean"])
        np.save(cache / "verbalts_ref_ts_cov.npy", stats["ts_cov"])
        np.save(cache / "verbalts_ref_joint_mean.npy", stats["joint_mean"])
        np.save(cache / "verbalts_ref_joint_cov.npy", stats["joint_cov"])
        np.save(cache / "verbalts_ref_count.npy", np.array([stats["reference_count"]], dtype=np.float64))


def _load_verbalts_cttp(
    *,
    verbalts_root: str | Path,
    clip_config_path: str | Path,
    clip_model_path: str | Path,
    device: torch.device,
):
    root = Path(verbalts_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"VerbalTS root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models.cttp.cttp_model import CTTP

    with Path(clip_config_path).open("r", encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    configs["device"] = str(device)
    clip = CTTP(configs)
    state = torch.load(clip_model_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    clip.load_state_dict(state)
    clip = clip.to(device)
    clip.eval()
    for param in clip.parameters():
        param.requires_grad_(False)
    return clip


def _mean_cov(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got {values.shape}")
    if values.shape[0] < 2:
        raise ValueError("At least two samples are required to compute covariance")
    return np.mean(values, axis=0), np.cov(values, rowvar=False)


def _cat_numpy(items: list[torch.Tensor], name: str) -> np.ndarray:
    if not items:
        raise ValueError(f"No {name} collected")
    return torch.cat(items, dim=0).numpy()
