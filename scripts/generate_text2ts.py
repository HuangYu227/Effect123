from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from effectcma_flow.config import load_config
from effectcma_flow.evaluation.sampler import sample_text2ts
from effectcma_flow.models import build_model
from effectcma_flow.training import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multivariate time series from free-form text captions using a trained "
            "TextToTS flow checkpoint."
        )
    )
    parser.add_argument("--config", required=True, help="Training/evaluation YAML config.")
    parser.add_argument("--checkpoint", required=True, help="TextToTS checkpoint path.")
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Caption to condition on. Can be passed multiple times.",
    )
    parser.add_argument(
        "--text-file",
        default=None,
        help=(
            "Optional caption file. Plain text uses one caption per non-empty line. "
            "JSON may be a list of strings or a list of string lists."
        ),
    )
    parser.add_argument(
        "--slot-separator",
        default=None,
        help="Optional separator for multiple text slots in each plain-text caption line.",
    )
    parser.add_argument("--output-dir", default="results/text2ts_generation")
    parser.add_argument("--data-root", default=None, help="Dataset root used only for shape inference if needed.")
    parser.add_argument("--sequence-length", type=int, default=None, help="Override output length.")
    parser.add_argument("--num-channels", type=int, default=None, help="Override output channel count.")
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per caption.")
    parser.add_argument("--batch-size", type=int, default=16, help="Generation batch size over caption-sample pairs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Override cfg.train.device, e.g. cuda:0 or cpu.")
    parser.add_argument("--solver", default=None, choices=["euler", "midpoint", "rk4"])
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--guidance-t-lo", type=float, default=None)
    parser.add_argument("--guidance-t-hi", type=float, default=None)
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf", "longclip"])
    parser.add_argument("--text-encoder-model", default=None, help="Override hf_model_name or longclip_model_name.")
    parser.add_argument(
        "--normalized-output",
        action="store_true",
        help="Save model-normalized values instead of denormalizing with checkpoint train stats.",
    )
    parser.add_argument("--save-normalized", action="store_true", help="Also save generated_normalized.npy.")
    parser.add_argument("--save-csv", action="store_true", help="Save one CSV per generated sample.")
    parser.add_argument("--plot", action="store_true", help="Save one PNG plot per generated sample.")
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    captions = _load_captions(args.text, args.text_file, slot_separator=args.slot_separator)
    if not captions:
        raise ValueError("Provide at least one --text or --text-file caption")

    cfg = load_config(args.config)
    _apply_runtime_overrides(cfg, args)

    device = resolve_device(args.device or str(cfg.get("train", {}).get("device", "auto")))
    payload = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint_eval_config(
        cfg,
        payload,
        text_encoder_override=args.text_encoder_mode,
        task_mode_override="text2ts",
    )
    if args.device is not None:
        cfg.setdefault("train", {})["device"] = args.device

    stats = checkpoint_stats_or_none(payload)
    sequence_length, num_channels = _resolve_output_shape(cfg, payload, args)

    model = build_model(cfg, sequence_length=sequence_length, num_channels=num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    sample_cfg = cfg.get("sample", {})
    solver = str(args.solver or sample_cfg.get("solver", "rk4")).lower()
    steps = int(args.steps if args.steps is not None else sample_cfg.get("steps", 64))
    noise_scale = float(args.noise_scale if args.noise_scale is not None else sample_cfg.get("noise_scale", 1.0))
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else sample_cfg.get("cfg_scale", 1.0))
    guidance_t_lo = float(
        args.guidance_t_lo if args.guidance_t_lo is not None else sample_cfg.get("guidance_t_lo", 0.0)
    )
    guidance_t_hi = float(
        args.guidance_t_hi if args.guidance_t_hi is not None else sample_cfg.get("guidance_t_hi", 1.0)
    )

    normalized = _generate(
        model=model,
        captions=captions,
        sequence_length=sequence_length,
        num_channels=num_channels,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        solver=solver,
        steps=steps,
        noise_scale=noise_scale,
        cfg_scale=cfg_scale,
        guidance_t_lo=guidance_t_lo,
        guidance_t_hi=guidance_t_hi,
        seed=int(args.seed),
    )

    output = normalized if args.normalized_output else _denormalize_if_needed(normalized, cfg, stats)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "generated.npy", output)
    if args.save_normalized or args.normalized_output:
        np.save(output_dir / "generated_normalized.npy", normalized)

    _write_captions(output_dir / "captions.json", captions)
    _write_metadata(
        output_dir / "metadata.json",
        args=args,
        cfg=cfg,
        captions=captions,
        sequence_length=sequence_length,
        num_channels=num_channels,
        solver=solver,
        steps=steps,
        noise_scale=noise_scale,
        cfg_scale=cfg_scale,
        guidance_t_lo=guidance_t_lo,
        guidance_t_hi=guidance_t_hi,
        normalized_output=bool(args.normalized_output),
        checkpoint_step=payload.get("step"),
        output_shape=list(output.shape),
    )
    if args.save_csv:
        _write_csv_samples(output_dir / "csv", output)
    if args.plot:
        _plot_samples(output_dir / "plots", output, captions)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "generated": str((output_dir / "generated.npy").resolve()),
                "shape": list(output.shape),
                "num_captions": len(captions),
                "num_samples": int(args.num_samples),
                "sequence_length": int(sequence_length),
                "num_channels": int(num_channels),
                "solver": solver,
                "steps": int(steps),
                "cfg_scale": float(cfg_scale),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _apply_runtime_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.data_root is not None:
        cfg.setdefault("data", {})["root"] = args.data_root
    if args.text_encoder_mode is not None:
        cfg.setdefault("text_encoder", {})["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        text_cfg = cfg.setdefault("text_encoder", {})
        mode = str(text_cfg.get("mode", "")).lower()
        key = "longclip_model_name" if mode == "longclip" else "hf_model_name"
        text_cfg[key] = args.text_encoder_model
    cfg.setdefault("task", {})["mode"] = "text2ts"


def _load_captions(
    direct_texts: list[str],
    text_file: str | None,
    *,
    slot_separator: str | None,
) -> list[list[str]]:
    captions: list[list[str]] = []
    for text in direct_texts:
        captions.append(_split_slots(text, slot_separator))
    if text_file is None:
        return captions

    path = Path(text_file)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("--text-file JSON must contain a list")
        for item in data:
            if isinstance(item, str):
                captions.append(_split_slots(item, slot_separator))
            elif isinstance(item, list) and all(isinstance(x, str) for x in item):
                captions.append([x for x in item if x.strip()])
            else:
                raise ValueError("JSON captions must be strings or lists of strings")
        return captions

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            captions.append(_split_slots(line, slot_separator))
    return captions


def _split_slots(text: str, slot_separator: str | None) -> list[str]:
    if slot_separator is None:
        return [text]
    slots = [part.strip() for part in text.split(slot_separator)]
    slots = [part for part in slots if part]
    return slots or [""]


def _resolve_output_shape(
    cfg: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[int, int]:
    seq = args.sequence_length
    channels = args.num_channels
    if seq is not None and channels is not None:
        return int(seq), int(channels)

    root = _candidate_data_root(cfg, payload, args)
    if root is not None:
        train_ts = root / "train_ts.npy"
        if train_ts.exists():
            arr = np.load(train_ts, mmap_mode="r", allow_pickle=False)
            if arr.ndim != 3:
                raise ValueError(f"{train_ts} must have shape [N, L, C], got {arr.shape}")
            raw_len = int(arr.shape[1])
            raw_channels = int(arr.shape[2])
            window_length = cfg.get("data", {}).get("window_length")
            inferred_seq = int(window_length) if window_length is not None else raw_len
            return int(seq or inferred_seq), int(channels or raw_channels)

    missing = []
    if seq is None:
        missing.append("--sequence-length")
    if channels is None:
        missing.append("--num-channels")
    raise ValueError(
        "Could not infer output shape from data root. Pass "
        + " and ".join(missing)
        + ", or provide --data-root containing train_ts.npy."
    )


def _candidate_data_root(
    cfg: dict[str, Any],
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> Path | None:
    values = [
        args.data_root,
        cfg.get("data", {}).get("root"),
        payload.get("config", {}).get("data", {}).get("root"),
    ]
    for value in values:
        if value:
            return Path(str(value)).expanduser()
    return None


def _generate(
    *,
    model,
    captions: list[list[str]],
    sequence_length: int,
    num_channels: int,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    solver: str,
    steps: int,
    noise_scale: float,
    cfg_scale: float,
    guidance_t_lo: float,
    guidance_t_hi: float,
    seed: int,
) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    pairs: list[tuple[int, int, list[str]]] = []
    for caption_idx, slots in enumerate(captions):
        for sample_idx in range(num_samples):
            pairs.append((caption_idx, sample_idx, slots))

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    output = np.empty((len(captions), num_samples, sequence_length, num_channels), dtype=np.float32)

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        text_condition = [slots for _, _, slots in chunk]
        shape_like = torch.empty(
            (len(chunk), sequence_length, num_channels),
            device=device,
            dtype=dtype,
        )
        generated, _ = sample_text2ts(
            model,
            shape_like,
            text_condition,
            solver=solver,
            steps=steps,
            noise_scale=noise_scale,
            generator=generator,
            cfg_scale=cfg_scale,
            guidance_t_lo=guidance_t_lo,
            guidance_t_hi=guidance_t_hi,
        )
        generated_np = generated.detach().cpu().float().numpy()
        for local_idx, (caption_idx, sample_idx, _) in enumerate(chunk):
            output[caption_idx, sample_idx] = generated_np[local_idx]
    return output


def _denormalize_if_needed(
    values: np.ndarray,
    cfg: dict[str, Any],
    stats: dict[str, torch.Tensor],
) -> np.ndarray:
    if not bool(cfg.get("data", {}).get("normalize", True)):
        return values
    mean = stats["mean"].detach().cpu().float().numpy().reshape(1, 1, 1, -1)
    std = stats["std"].detach().cpu().float().numpy().reshape(1, 1, 1, -1)
    return values * std + mean


def _write_captions(path: Path, captions: list[list[str]]) -> None:
    path.write_text(json.dumps(captions, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_metadata(
    path: Path,
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    captions: list[list[str]],
    sequence_length: int,
    num_channels: int,
    solver: str,
    steps: int,
    noise_scale: float,
    cfg_scale: float,
    guidance_t_lo: float,
    guidance_t_hi: float,
    normalized_output: bool,
    checkpoint_step: Any,
    output_shape: list[int],
) -> None:
    metadata = {
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint_step,
        "data_root": cfg.get("data", {}).get("root"),
        "text_encoder": cfg.get("text_encoder", {}),
        "sequence_length": int(sequence_length),
        "num_channels": int(num_channels),
        "num_captions": len(captions),
        "num_samples": int(args.num_samples),
        "seed": int(args.seed),
        "solver": solver,
        "steps": int(steps),
        "noise_scale": float(noise_scale),
        "cfg_scale": float(cfg_scale),
        "guidance_t_lo": float(guidance_t_lo),
        "guidance_t_hi": float(guidance_t_hi),
        "normalized_output": bool(normalized_output),
        "output_shape": output_shape,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv_samples(output_dir: Path, values: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_names = [f"channel_{idx}" for idx in range(values.shape[-1])]
    for caption_idx in range(values.shape[0]):
        for sample_idx in range(values.shape[1]):
            path = output_dir / f"caption_{caption_idx:03d}_sample_{sample_idx:03d}.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time"] + channel_names)
                for t, row in enumerate(values[caption_idx, sample_idx]):
                    writer.writerow([t] + [float(v) for v in row])


def _plot_samples(output_dir: Path, values: np.ndarray, captions: list[list[str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    for caption_idx in range(values.shape[0]):
        title = " | ".join(captions[caption_idx])
        for sample_idx in range(values.shape[1]):
            fig, ax = plt.subplots(figsize=(10, 4), dpi=160)
            series = values[caption_idx, sample_idx]
            for channel_idx in range(series.shape[-1]):
                ax.plot(series[:, channel_idx], linewidth=1.2, label=f"ch{channel_idx}")
            ax.set_title(title[:140])
            ax.set_xlabel("time")
            ax.set_ylabel("value")
            if series.shape[-1] <= 12:
                ax.legend(loc="best", fontsize=7, ncol=min(series.shape[-1], 4))
            fig.tight_layout()
            fig.savefig(output_dir / f"caption_{caption_idx:03d}_sample_{sample_idx:03d}.png")
            plt.close(fig)


if __name__ == "__main__":
    main()
