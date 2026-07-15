from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FACTOR_NAMES = ("trend", "periodicity", "channel")
FACTOR_LABELS = ("Trend", "Periodicity", "Channel")
COLORS = ("#0072B2", "#D55E00", "#009E73")
MARKERS = ("o", "s", "^")
HATCHES = ("///", "\\\\", "...")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path


@dataclass
class CachedBatch:
    batch: dict[str, Any]
    noise: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether a text-conditioned flow can recover a selectively "
            "perturbed property of an intermediate flow state."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Runtime/base YAML config.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="LABEL=CHECKPOINT",
        help="Repeat for each compatible checkpoint; order controls the legend.",
    )
    parser.add_argument("--text-encoder-model", default=None)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--flow-times", type=float, nargs="+", default=(0.25, 0.50, 0.75))
    parser.add_argument("--reference-time", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "midpoint", "rk4"), default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--guidance-t-lo", type=float, default=None)
    parser.add_argument("--guidance-t-hi", type=float, default=None)
    parser.add_argument("--trend-window", type=int, default=17)
    parser.add_argument("--period-topk", type=int, default=3)
    parser.add_argument("--period-retain", type=float, default=0.05)
    parser.add_argument("--channel-shift", type=int, default=0, help="0 uses sequence_length//4.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--figure-stem", default="counterfactual_state_recovery_single_column")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def parse_model_spec(value: str) -> ModelSpec:
    if "=" not in value:
        raise ValueError(f"--model must use LABEL=CHECKPOINT, got {value!r}")
    label, checkpoint = value.split("=", 1)
    label = label.strip()
    checkpoint = checkpoint.strip()
    if not label or not checkpoint:
        raise ValueError(f"--model must use non-empty LABEL=CHECKPOINT, got {value!r}")
    return ModelSpec(label=label, checkpoint=Path(checkpoint))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def cache_batch(batch: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu()
        elif isinstance(value, list):
            result[key] = list(value)
        elif isinstance(value, tuple):
            result[key] = tuple(value)
        else:
            result[key] = value
    return result


def ensure_odd_window(window: int, length: int) -> int:
    window = max(3, min(int(window), int(length) - (1 - int(length) % 2)))
    if window % 2 == 0:
        window -= 1
    return max(3, window)


def smooth_time(x: torch.Tensor, window: int) -> torch.Tensor:
    window = ensure_odd_window(window, x.shape[1])
    pad = window // 2
    channels_first = x.transpose(1, 2)
    padded = F.pad(channels_first, (pad, pad), mode="reflect")
    return F.avg_pool1d(padded, kernel_size=window, stride=1).transpose(1, 2)


def corrupt_trend(x: torch.Tensor, target: torch.Tensor, *, window: int, **_: Any) -> torch.Tensor:
    del target
    low_frequency = smooth_time(x, window)
    centered_trend = low_frequency - low_frequency.mean(dim=1, keepdim=True)
    return x - centered_trend


def corrupt_periodicity(
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    topk: int,
    retain: float,
    **_: Any,
) -> torch.Tensor:
    spectrum = torch.fft.rfft(x, dim=1)
    target_magnitude = torch.fft.rfft(target, dim=1).abs()
    if target_magnitude.shape[1] <= 1:
        return x.clone()
    scores = target_magnitude.clone()
    scores[:, 0, :] = -torch.inf
    k = max(1, min(int(topk), int(scores.shape[1] - 1)))
    indices = scores.topk(k=k, dim=1).indices
    mask = torch.ones_like(scores)
    mask.scatter_(1, indices, float(retain))
    return torch.fft.irfft(spectrum * mask, n=x.shape[1], dim=1)


def corrupt_channel(
    x: torch.Tensor,
    target: torch.Tensor,
    *,
    shift: int,
    **_: Any,
) -> torch.Tensor:
    del target
    if x.shape[2] < 2:
        raise ValueError("Channel-relation intervention requires at least two channels.")
    output = x.clone()
    shift = int(shift) if int(shift) > 0 else max(1, int(x.shape[1] // 4))
    for channel in range(1, x.shape[2]):
        output[:, :, channel] = torch.roll(x[:, :, channel], shifts=shift * channel, dims=1)
    return output


def trend_error(x: torch.Tensor, target: torch.Tensor, *, window: int) -> torch.Tensor:
    x_trend = smooth_time(x, window)
    target_trend = smooth_time(target, window)
    numerator = (x_trend - target_trend).square().mean(dim=(1, 2)).sqrt()
    centered = target_trend - target_trend.mean(dim=1, keepdim=True)
    denominator = centered.square().mean(dim=(1, 2)).sqrt().clamp_min(0.10)
    return numerator / denominator


def periodicity_error(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x_mag = torch.log1p(torch.fft.rfft(x, dim=1).abs())
    y_mag = torch.log1p(torch.fft.rfft(target, dim=1).abs())
    if x_mag.shape[1] > 1:
        x_mag = x_mag[:, 1:, :]
        y_mag = y_mag[:, 1:, :]
    x_flat = x_mag.flatten(1)
    y_flat = y_mag.flatten(1)
    similarity = F.cosine_similarity(x_flat, y_flat, dim=1, eps=1e-8)
    return 1.0 - similarity


def cross_correlation_profile(x: torch.Tensor, max_lag: int) -> torch.Tensor:
    x = (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True).clamp_min(1e-6)
    profiles: list[torch.Tensor] = []
    for other_channel in range(1, x.shape[2]):
        correlations: list[torch.Tensor] = []
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                first = x[:, :lag, 0]
                second = x[:, -lag:, other_channel]
            elif lag > 0:
                first = x[:, lag:, 0]
                second = x[:, :-lag, other_channel]
            else:
                first = x[:, :, 0]
                second = x[:, :, other_channel]
            correlations.append((first * second).mean(dim=1))
        profiles.append(torch.stack(correlations, dim=1))
    return torch.stack(profiles, dim=1).mean(dim=1)


def channel_error(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if x.shape[2] < 2:
        raise ValueError("Channel-relation metric requires at least two channels.")
    max_lag = max(1, int(x.shape[1] // 4))
    x_profile = cross_correlation_profile(x, max_lag)
    y_profile = cross_correlation_profile(target, max_lag)
    similarity = F.cosine_similarity(x_profile, y_profile, dim=1, eps=1e-8)
    return 1.0 - similarity


def prepare_condition_pair(
    model: torch.nn.Module,
    raw_condition: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg_scale: float,
) -> tuple[Any, Any | None]:
    condition = model.prepare_condition(raw_condition, device=device, dtype=dtype)
    if float(cfg_scale) <= 1.0:
        return condition, None
    if isinstance(raw_condition, dict) and torch.is_tensor(raw_condition.get("embeddings")):
        null_raw: Any = {"embeddings": torch.zeros_like(raw_condition["embeddings"])}
        if torch.is_tensor(raw_condition.get("mask")):
            null_raw["mask"] = raw_condition["mask"]
    else:
        null_raw = [[""] for _ in range(batch_size)]
    return condition, model.prepare_condition(null_raw, device=device, dtype=dtype)


@torch.no_grad()
def resume_ode(
    model: torch.nn.Module,
    x_start: torch.Tensor,
    condition: Any,
    null_condition: Any | None,
    *,
    start_step: int,
    total_steps: int,
    solver: str,
    cfg_scale: float,
    guidance_t_lo: float,
    guidance_t_hi: float,
) -> torch.Tensor:
    x = x_start.clone()
    batch_size = x.shape[0]
    dt = 1.0 / float(total_steps)

    def velocity(x_cur: torch.Tensor, t_value: float) -> torch.Tensor:
        time = torch.full((batch_size,), t_value, device=x.device, dtype=x.dtype)
        conditional, _ = model(x_cur, time, condition)
        apply_cfg = null_condition is not None and guidance_t_lo <= t_value <= guidance_t_hi
        if not apply_cfg:
            return conditional
        unconditional, _ = model(x_cur, time, null_condition)
        return unconditional + float(cfg_scale) * (conditional - unconditional)

    for step in range(int(start_step), int(total_steps)):
        t0 = step * dt
        if solver == "euler":
            x = x + dt * velocity(x, t0)
        elif solver == "midpoint":
            k1 = velocity(x, t0)
            k2 = velocity(x + 0.5 * dt * k1, min(1.0, t0 + 0.5 * dt))
            x = x + dt * k2
        elif solver == "rk4":
            k1 = velocity(x, t0)
            k2 = velocity(x + 0.5 * dt * k1, min(1.0, t0 + 0.5 * dt))
            k3 = velocity(x + 0.5 * dt * k2, min(1.0, t0 + 0.5 * dt))
            k4 = velocity(x + dt * k3, min(1.0, t0 + dt))
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unsupported solver: {solver}")
    return x


def actual_flow_steps(flow_times: list[float], steps: int) -> list[tuple[float, int]]:
    output: list[tuple[float, int]] = []
    for requested in flow_times:
        if not 0.0 < float(requested) < 1.0:
            raise ValueError(f"flow times must lie strictly inside (0,1), got {requested}")
        start_step = int(round(float(requested) * steps))
        actual = start_step / float(steps)
        if abs(actual - float(requested)) > 1e-7:
            raise ValueError(
                f"flow time {requested} is not aligned to the {steps}-step ODE grid; "
                f"nearest value is {actual:.6f}"
            )
        output.append((actual, start_step))
    return output


def recovery_stat(
    match_before: np.ndarray,
    conflict_before: np.ndarray,
    match_after: np.ndarray,
    conflict_after: np.ndarray,
) -> tuple[float, float, float]:
    gap_before = float(np.mean(conflict_before - match_before))
    gap_after = float(np.mean(conflict_after - match_after))
    if gap_before <= 1e-8:
        return float("nan"), gap_before, gap_after
    return 1.0 - gap_after / gap_before, gap_before, gap_after


def bootstrap_recovery(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    if draws <= 0:
        return float("nan"), float("nan")
    count = int(arrays[0].shape[0])
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(draws)):
        index = rng.integers(0, count, size=count)
        value, _, _ = recovery_stat(*(array[index] for array in arrays))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(np.asarray(values), [2.5, 97.5]))


def macro_bootstrap(
    factor_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    point_values = [recovery_stat(*arrays)[0] for arrays in factor_arrays]
    point = float(np.nanmean(point_values))
    if draws <= 0:
        return point, float("nan"), float("nan")
    count = int(factor_arrays[0][0].shape[0])
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(int(draws)):
        index = rng.integers(0, count, size=count)
        factor_values = [recovery_stat(*(array[index] for array in arrays))[0] for arrays in factor_arrays]
        values.append(float(np.nanmean(factor_values)))
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return point, float(low), float(high)


def safe_key(*parts: str) -> str:
    return "__".join(re.sub(r"[^A-Za-z0-9]+", "_", part).strip("_") for part in parts)


def plot_single_column(
    rows: list[dict[str, Any]],
    model_labels: list[str],
    flow_times: list[float],
    reference_time: float,
    output_dir: Path,
    stem: str,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.3,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 3.65), sharey=True)
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.12, top=0.88, hspace=0.53)

    factor_x = np.arange(len(FACTOR_NAMES), dtype=np.float32)
    width = 0.23
    for model_index, label in enumerate(model_labels):
        selected = [
            next(
                row
                for row in rows
                if row["model"] == label
                and row["factor"] == factor
                and abs(float(row["flow_time"]) - reference_time) < 1e-8
            )
            for factor in FACTOR_NAMES
        ]
        values = np.asarray([row["recovery"] for row in selected]) * 100.0
        low = np.asarray([row["ci_low"] for row in selected]) * 100.0
        high = np.asarray([row["ci_high"] for row in selected]) * 100.0
        error = np.vstack([values - low, high - values])
        axes[0].bar(
            factor_x + (model_index - (len(model_labels) - 1) / 2.0) * width,
            values,
            width=width,
            color=COLORS[model_index % len(COLORS)],
            edgecolor="black",
            linewidth=0.45,
            hatch=HATCHES[model_index % len(HATCHES)],
            yerr=error,
            error_kw={"elinewidth": 0.7, "capsize": 1.7, "capthick": 0.7},
            label=label,
            zorder=3,
        )
    axes[0].set_xticks(factor_x, FACTOR_LABELS)
    axes[0].set_title(f"(a) Recovery by perturbed property ($t={reference_time:.2f}$)", pad=4)

    for model_index, label in enumerate(model_labels):
        selected = [
            next(
                row
                for row in rows
                if row["model"] == label
                and row["factor"] == "macro"
                and abs(float(row["flow_time"]) - flow_time) < 1e-8
            )
            for flow_time in flow_times
        ]
        values = np.asarray([row["recovery"] for row in selected]) * 100.0
        low = np.asarray([row["ci_low"] for row in selected]) * 100.0
        high = np.asarray([row["ci_high"] for row in selected]) * 100.0
        axes[1].errorbar(
            flow_times,
            values,
            yerr=np.vstack([values - low, high - values]),
            color=COLORS[model_index % len(COLORS)],
            marker=MARKERS[model_index % len(MARKERS)],
            linewidth=1.35,
            markersize=3.6,
            capsize=2.0,
            elinewidth=0.75,
            label=label,
            zorder=3,
        )
    axes[1].set_title("(b) Macro recovery across flow states", pad=4)
    axes[1].set_xlabel("Flow time $t$")
    axes[1].set_xticks(flow_times, [f"{value:.2f}" for value in flow_times])

    finite_values = [float(row["ci_low"]) * 100.0 for row in rows if np.isfinite(row["ci_low"])]
    finite_values += [float(row["ci_high"]) * 100.0 for row in rows if np.isfinite(row["ci_high"])]
    lower = min(0.0, min(finite_values) if finite_values else 0.0)
    upper = max(100.0, max(finite_values) if finite_values else 100.0)
    margin = max(4.0, 0.06 * (upper - lower))
    for axis in axes:
        axis.axhline(0.0, color="#777777", linewidth=0.65, zorder=1)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_ylabel("Conflict recovery (%)")
        axis.set_ylim(lower - margin, upper + margin)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=len(model_labels), frameon=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[saved] figure: {pdf_path}")
    print(f"[saved] figure: {png_path}")


def main() -> None:
    args = parse_args()
    model_specs = [parse_model_spec(value) for value in args.model]
    if len(model_specs) < 2:
        raise ValueError("Provide at least two --model entries.")
    if len(model_specs) > len(COLORS):
        raise ValueError(f"This single-column design supports at most {len(COLORS)} models.")
    if len({spec.label for spec in model_specs}) != len(model_specs):
        raise ValueError("Model labels must be unique.")
    if args.num_samples <= 0 or args.batch_size <= 0 or args.bootstrap < 0:
        raise ValueError("num-samples and batch-size must be positive; bootstrap must be non-negative.")
    for path in (args.config, args.data_root, *(spec.checkpoint for spec in model_specs)):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    set_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from effectcma_flow.config import load_config, resolve_data_root
    from effectcma_flow.data import WeatherRawCaptionDataset, collate_raw_caption_batch
    from effectcma_flow.models import build_model
    from effectcma_flow.training.checkpoint import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint
    from effectcma_flow.training.utils import resolve_device, text_condition_from_batch

    runtime_cfg = load_config(args.config)
    runtime_cfg.setdefault("task", {})["mode"] = "text2ts"
    runtime_cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    runtime_cfg["train"]["device"] = "auto"
    resolve_data_root(runtime_cfg, str(args.data_root))
    if args.text_encoder_model is not None:
        text_cfg = runtime_cfg.setdefault("text_encoder", {})
        mode = str(text_cfg.get("mode", "longclip")).lower()
        text_cfg["longclip_model_name" if mode == "longclip" else "hf_model_name"] = str(args.text_encoder_model)

    device = resolve_device("auto")
    reference_payload = load_training_checkpoint(model_specs[0].checkpoint, device)
    reference_cfg = checkpoint_eval_config(runtime_cfg, reference_payload, task_mode_override="text2ts")
    reference_stats = checkpoint_stats_or_none(reference_payload)
    sample_cfg = reference_cfg.get("sample", {})
    steps = int(args.steps if args.steps is not None else sample_cfg.get("steps", 48))
    solver = str(args.solver if args.solver is not None else sample_cfg.get("solver", "rk4")).lower()
    noise_scale = float(args.noise_scale if args.noise_scale is not None else sample_cfg.get("noise_scale", 1.0))
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else sample_cfg.get("cfg_scale", 1.0))
    guidance_t_lo = float(args.guidance_t_lo if args.guidance_t_lo is not None else sample_cfg.get("guidance_t_lo", 0.0))
    guidance_t_hi = float(args.guidance_t_hi if args.guidance_t_hi is not None else sample_cfg.get("guidance_t_hi", 1.0))
    flow_steps = actual_flow_steps([float(x) for x in args.flow_times], steps)
    flow_times = [item[0] for item in flow_steps]
    reference_time = min(flow_times, key=lambda value: abs(value - float(args.reference_time)))

    text_mode = str(reference_cfg["text_encoder"].get("mode", "hash"))
    include_embeddings = bool(reference_cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    dataset = WeatherRawCaptionDataset(
        reference_cfg["data"]["root"],
        args.split,
        window_length=reference_cfg["data"].get("window_length"),
        normalize=bool(reference_cfg["data"].get("normalize", True)),
        stats=reference_stats,
        seed=int(reference_cfg["data"].get("seed", 0)) + 20_000,
        caption_policy="random",
        include_precomputed_embeddings=include_embeddings,
        precomputed_dim=int(reference_cfg["text_encoder"].get("precomputed_dim", 128)),
    )
    if dataset.num_channels < 2:
        raise ValueError("The requested three-factor experiment requires a multivariate dataset.")
    count = min(int(args.num_samples), len(dataset))
    selection_rng = np.random.default_rng(int(args.seed))
    selected_indices = np.sort(selection_rng.choice(len(dataset), size=count, replace=False))
    np.save(args.output_dir / "sample_indices.npy", selected_indices)
    loader = DataLoader(
        Subset(dataset, selected_indices.tolist()),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_raw_caption_batch,
    )
    noise_generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 17)
    cached_batches: list[CachedBatch] = []
    for batch in loader:
        batch = cache_batch(batch)
        noise = torch.randn(batch["Y"].shape, generator=noise_generator, dtype=batch["Y"].dtype)
        cached_batches.append(CachedBatch(batch=batch, noise=noise))

    error_functions: dict[str, Callable[..., torch.Tensor]] = {
        "trend": lambda x, y: trend_error(x, y, window=int(args.trend_window)),
        "periodicity": periodicity_error,
        "channel": channel_error,
    }
    corruptions: dict[str, Callable[..., torch.Tensor]] = {
        "trend": lambda x, y: corrupt_trend(x, y, window=int(args.trend_window)),
        "periodicity": lambda x, y: corrupt_periodicity(
            x, y, topk=int(args.period_topk), retain=float(args.period_retain)
        ),
        "channel": lambda x, y: corrupt_channel(x, y, shift=int(args.channel_shift)),
    }

    raw_chunks: dict[str, list[np.ndarray]] = {}
    model_metadata: list[dict[str, Any]] = []
    for model_index, spec in enumerate(model_specs):
        payload = reference_payload if model_index == 0 else load_training_checkpoint(spec.checkpoint, device)
        cfg = checkpoint_eval_config(runtime_cfg, payload, task_mode_override="text2ts")
        stats = checkpoint_stats_or_none(payload)
        for key in reference_stats:
            if key not in stats or not torch.allclose(reference_stats[key], stats[key], rtol=1e-5, atol=1e-6):
                raise ValueError(f"Checkpoint {spec.checkpoint} uses incompatible normalization statistic {key!r}.")
        variant_text_mode = str(cfg["text_encoder"].get("mode", "hash"))
        if variant_text_mode != text_mode:
            raise ValueError(
                f"All compared checkpoints must use the same text interface; {spec.label} uses {variant_text_mode}, "
                f"reference uses {text_mode}."
            )
        model = build_model(cfg, sequence_length=dataset.sequence_length, num_channels=dataset.num_channels).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        model_cfg = cfg.get("model", cfg)
        model_metadata.append(
            {
                "label": spec.label,
                "checkpoint": str(spec.checkpoint),
                "step": int(payload.get("step", -1)),
                "use_cross_modal_bridge": bool(model_cfg.get("use_cross_modal_bridge", False)),
                "operator_architecture": str(model_cfg.get("operator_architecture", "homogeneous")),
                "temporal_pyramid_patch_lens": model_cfg.get("temporal_pyramid_patch_lens"),
            }
        )
        caption_slot_strategy = str(cfg.get("train", {}).get("caption_slot_strategy", "single"))
        max_caption_slots = int(cfg.get("train", {}).get("max_caption_slots", 8))
        include_all_candidates = bool(cfg.get("train", {}).get("include_all_caption_candidates", False))

        progress = tqdm(cached_batches, desc=f"state-recovery [{spec.label}]", dynamic_ncols=True)
        for cached in progress:
            batch = move_batch(cached.batch, device)
            target = batch["Y"].float()
            noise = cached.noise.to(device=device, dtype=target.dtype) * noise_scale
            raw_condition = text_condition_from_batch(
                batch,
                variant_text_mode,
                condition_key="caption",
                caption_slot_strategy=caption_slot_strategy,
                max_caption_slots=max_caption_slots,
                include_all_caption_candidates=include_all_candidates,
            )
            condition, null_condition = prepare_condition_pair(
                model,
                raw_condition,
                batch_size=target.shape[0],
                device=device,
                dtype=target.dtype,
                cfg_scale=cfg_scale,
            )
            for flow_time, start_step in flow_steps:
                matched_state = (1.0 - flow_time) * noise + flow_time * target
                matched_output = resume_ode(
                    model,
                    matched_state,
                    condition,
                    null_condition,
                    start_step=start_step,
                    total_steps=steps,
                    solver=solver,
                    cfg_scale=cfg_scale,
                    guidance_t_lo=guidance_t_lo,
                    guidance_t_hi=guidance_t_hi,
                )
                for factor in FACTOR_NAMES:
                    conflict_state = corruptions[factor](matched_state, target)
                    conflict_output = resume_ode(
                        model,
                        conflict_state,
                        condition,
                        null_condition,
                        start_step=start_step,
                        total_steps=steps,
                        solver=solver,
                        cfg_scale=cfg_scale,
                        guidance_t_lo=guidance_t_lo,
                        guidance_t_hi=guidance_t_hi,
                    )
                    metric = error_functions[factor]
                    values = {
                        "match_before": metric(matched_state, target),
                        "conflict_before": metric(conflict_state, target),
                        "match_after": metric(matched_output, target),
                        "conflict_after": metric(conflict_output, target),
                    }
                    for value_name, value in values.items():
                        key = safe_key(spec.label, f"t{flow_time:.4f}", factor, value_name)
                        raw_chunks.setdefault(key, []).append(value.detach().float().cpu().numpy())

        del model
        if model_index > 0:
            del payload
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    raw_arrays = {key: np.concatenate(chunks, axis=0).astype(np.float32, copy=False) for key, chunks in raw_chunks.items()}
    np.savez_compressed(args.output_dir / "raw_recovery_errors.npz", **raw_arrays)

    rows: list[dict[str, Any]] = []
    for model_index, spec in enumerate(model_specs):
        for time_index, flow_time in enumerate(flow_times):
            factor_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
            for factor_index, factor in enumerate(FACTOR_NAMES):
                arrays = tuple(
                    raw_arrays[safe_key(spec.label, f"t{flow_time:.4f}", factor, value_name)]
                    for value_name in ("match_before", "conflict_before", "match_after", "conflict_after")
                )
                factor_arrays.append(arrays)  # type: ignore[arg-type]
                recovery, gap_before, gap_after = recovery_stat(*arrays)
                if not np.isfinite(recovery):
                    raise RuntimeError(
                        f"Intervention {factor!r} at t={flow_time:.4f} did not create a positive "
                        f"measurable conflict for model {spec.label!r} (gap_before={gap_before:.6g})."
                    )
                ci_low, ci_high = bootstrap_recovery(
                    arrays, draws=int(args.bootstrap), seed=int(args.seed) + 1000 * model_index + 100 * time_index + factor_index
                )
                if not np.isfinite(ci_low) or not np.isfinite(ci_high):
                    raise RuntimeError(f"Bootstrap failed for {spec.label!r}, {factor!r}, t={flow_time:.4f}.")
                rows.append(
                    {
                        "model": spec.label,
                        "flow_time": flow_time,
                        "factor": factor,
                        "recovery": recovery,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "gap_before": gap_before,
                        "gap_after": gap_after,
                        "samples": count,
                    }
                )
            macro, macro_low, macro_high = macro_bootstrap(
                factor_arrays,
                draws=int(args.bootstrap),
                seed=int(args.seed) + 10_000 + 1000 * model_index + time_index,
            )
            rows.append(
                {
                    "model": spec.label,
                    "flow_time": flow_time,
                    "factor": "macro",
                    "recovery": macro,
                    "ci_low": macro_low,
                    "ci_high": macro_high,
                    "gap_before": None,
                    "gap_after": None,
                    "samples": count,
                }
            )

    csv_path = args.output_dir / "recovery_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "experiment": "counterfactual_text_state_conflict_recovery",
        "definition": (
            "Recovery = 1 - mean(E_conflict_after - E_match_after) / "
            "mean(E_conflict_before - E_match_before). The matched and conflict states share caption, "
            "clean target, noise, and flow time; only one state property is perturbed."
        ),
        "models": model_metadata,
        "data_root": str(args.data_root),
        "split": args.split,
        "samples": count,
        "sample_indices": str(args.output_dir / "sample_indices.npy"),
        "flow_times": flow_times,
        "reference_time": reference_time,
        "solver": solver,
        "steps": steps,
        "noise_scale": noise_scale,
        "cfg_scale": cfg_scale,
        "guidance_interval": [guidance_t_lo, guidance_t_hi],
        "interventions": {
            "trend": {"operation": "remove centered moving-average component", "window": int(args.trend_window)},
            "periodicity": {
                "operation": "attenuate strongest non-DC target-frequency bins in the current state",
                "topk": int(args.period_topk),
                "retain": float(args.period_retain),
            },
            "channel": {
                "operation": "circularly shift non-reference channels",
                "shift": int(args.channel_shift) if int(args.channel_shift) > 0 else dataset.sequence_length // 4,
            },
        },
        "bootstrap_draws": int(args.bootstrap),
        "seed": int(args.seed),
        "rows": rows,
    }
    json_path = args.output_dir / "recovery_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_single_column(
        rows,
        [spec.label for spec in model_specs],
        flow_times,
        reference_time,
        args.output_dir,
        str(args.figure_stem),
        int(args.dpi),
    )
    print(f"[saved] raw errors: {args.output_dir / 'raw_recovery_errors.npz'}")
    print(f"[saved] table: {csv_path}")
    print(f"[saved] summary: {json_path}")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
