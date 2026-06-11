from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from effectcma_flow.config import load_config, resolve_data_root
from effectcma_flow.data import (
    WeatherRawCaptionDataset,
    WeatherSemiSyntheticDataset,
    collate_effect_batch,
    collate_raw_caption_batch,
    compute_train_stats,
)
from effectcma_flow.evaluation.metrics import compute_field_metrics, compute_metrics, compute_text2ts_metrics
from effectcma_flow.evaluation.sampler import euler_sample, sample_text2ts
from effectcma_flow.evaluation.visualize import dump_generated_plot, dump_sample_plot, dump_text2ts_plot
from effectcma_flow.models import build_model
from effectcma_flow.training import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint, resolve_device
from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/weather_core.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="artifacts/sample_weather.png")
    parser.add_argument("--npy-output", default=None)
    parser.add_argument("--caption", default=None, help="Prompt-driven text-only generation without reading a target sample")
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf", "longclip"])
    parser.add_argument("--text-encoder-model", default=None, help="Override hf_model_name or longclip_model_name")
    parser.add_argument("--task-mode", default=None, choices=["text2ts", "edit"])
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.text_encoder_mode is not None:
        cfg["text_encoder"]["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        key = "longclip_model_name" if str(cfg["text_encoder"].get("mode", "")).lower() == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
    if args.task_mode is not None:
        cfg.setdefault("task", {})["mode"] = args.task_mode
    resolve_data_root(cfg, args.data_root)
    result = run_sample(
        cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        index=args.index,
        output=args.output,
        npy_output=args.npy_output,
        caption=args.caption,
        length=args.length,
        channels=args.channels,
        allow_random_init=args.allow_random_init,
        text_encoder_override=args.text_encoder_mode,
        task_mode_override=args.task_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_sample(
    cfg: dict,
    *,
    checkpoint: str | None,
    split: str,
    index: int,
    output: str,
    npy_output: str | None = None,
    caption: str | None = None,
    length: int | None = None,
    channels: int | None = None,
    allow_random_init: bool = False,
    text_encoder_override: str | None = None,
    task_mode_override: str | None = None,
) -> dict:
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    payload = None
    if checkpoint is None:
        if not allow_random_init:
            raise ValueError("Sampling requires --checkpoint. Use --allow-random-init only for smoke tests.")
    else:
        payload = load_training_checkpoint(checkpoint, device)
        cfg = checkpoint_eval_config(cfg, payload, text_encoder_override=text_encoder_override, task_mode_override=task_mode_override)
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    task_mode = str(cfg.get("task", {}).get("mode", "text2ts")).lower()
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    stats = checkpoint_stats_or_none(payload) if payload is not None else compute_train_stats(cfg["data"]["root"])
    if caption is not None:
        return run_prompt_sample(
            cfg,
            stats=stats,
            checkpoint_payload=payload,
            caption=caption,
            output=output,
            npy_output=npy_output,
            length=length,
            channels=channels,
            device=device,
        )
    if task_mode == "text2ts":
        ds = WeatherRawCaptionDataset(
            cfg["data"]["root"],
            split,
            window_length=cfg["data"].get("window_length"),
            normalize=bool(cfg["data"].get("normalize", True)),
            stats=stats,
            seed=int(cfg["data"].get("seed", 0)) + 30_000,
            caption_policy="cyclic",
            include_precomputed_embeddings=include_embeddings,
            precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
        )
        batch = collate_raw_caption_batch([ds[index]])
    elif task_mode == "edit":
        ds = WeatherSemiSyntheticDataset(
            cfg["data"]["root"],
            split,
            window_length=cfg["data"].get("window_length"),
            normalize=bool(cfg["data"].get("normalize", True)),
            stats=stats,
            effect_types=cfg["data"].get("effect_types"),
            seed=int(cfg["data"].get("seed", 0)) + 30_000,
            include_precomputed_embeddings=include_embeddings,
            precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
        )
        batch = collate_effect_batch([ds[index]])
    else:
        raise ValueError(f"Unknown task.mode {task_mode!r}; expected 'text2ts' or 'edit'")
    model = build_model(cfg, sequence_length=ds.sequence_length, num_channels=ds.num_channels).to(device)
    if payload is not None:
        model.load_state_dict(payload["model"])
    batch = batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        if task_mode == "text2ts":
            pred, aux = sample_text2ts(
                model,
                batch["Y"],
                text_condition_from_batch(batch, text_mode, condition_key="caption"),
                solver=str(cfg.get("sample", {}).get("solver", "euler")),
                steps=int(cfg.get("sample", {}).get("steps", 16)),
                noise_scale=float(cfg.get("sample", {}).get("noise_scale", 1.0)),
            )
        else:
            pred, aux = euler_sample(model, batch["B"], text_condition_from_batch(batch, text_mode, condition_key="slots"), steps=int(cfg.get("sample", {}).get("steps", 16)))
    if task_mode == "text2ts":
        metrics = compute_text2ts_metrics(pred, batch["Y"])
        channel = 0
        dump_text2ts_plot(output, batch["Y"][0], pred[0], channel=channel)
        return {"caption": batch["caption"][0], "plot": output, "metrics": metrics}
    metrics = compute_metrics(pred, batch["Y"], batch["B"], batch["mask"])
    metrics.update(compute_field_metrics(aux, batch["mask"], batch["spec"]))
    channel = int(batch["spec"][0]["channels"][0])
    dump_sample_plot(output, batch["B"][0], batch["Y"][0], pred[0], batch["mask"][0], channel=channel)
    return {"slot": batch["slots"][0], "effect_type": batch["effect_type"][0], "plot": output, "metrics": metrics}


def run_prompt_sample(
    cfg: dict,
    *,
    stats: dict[str, torch.Tensor],
    checkpoint_payload: dict | None,
    caption: str,
    output: str,
    npy_output: str | None,
    length: int | None,
    channels: int | None,
    device: torch.device,
) -> dict:
    task_mode = str(cfg.get("task", {}).get("mode", "text2ts")).lower()
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    if task_mode != "text2ts":
        raise ValueError("--caption prompt generation requires task.mode='text2ts'")
    if text_mode == "precomputed":
        raise ValueError("--caption prompt generation cannot use text_encoder.mode='precomputed'; use hash/hf/longclip")
    inferred_length, inferred_channels = _infer_shape(cfg, stats)
    length = int(length or inferred_length)
    channels = int(channels or inferred_channels)
    shape_like = torch.zeros(1, length, channels, device=device, dtype=torch.float32)
    model = build_model(cfg, sequence_length=length, num_channels=channels).to(device)
    if checkpoint_payload is not None:
        model.load_state_dict(checkpoint_payload["model"])
    model.eval()
    with torch.no_grad():
        pred, _ = sample_text2ts(
            model,
            shape_like,
            [[caption]],
            solver=str(cfg.get("sample", {}).get("solver", "euler")),
            steps=int(cfg.get("sample", {}).get("steps", 16)),
            noise_scale=float(cfg.get("sample", {}).get("noise_scale", 1.0)),
        )
    pred_raw = _denormalize(pred, stats)[0]
    dump_generated_plot(output, pred_raw, channel=0)
    if npy_output is None:
        npy_output = str(Path(output).with_suffix(".npy"))
    Path(npy_output).parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_output, pred_raw.detach().cpu().numpy().astype(np.float32))
    return {"caption": caption, "plot": output, "npy": npy_output, "shape": [length, channels]}


def _infer_shape(cfg: dict, stats: dict[str, torch.Tensor]) -> tuple[int, int]:
    root = cfg.get("data", {}).get("root")
    length = cfg.get("data", {}).get("window_length")
    channels = int(stats["mean"].shape[-1])
    if root:
        try:
            ds = WeatherRawCaptionDataset(root, "train", stats=stats, window_length=length, caption_policy="first")
            return ds.sequence_length, ds.num_channels
        except FileNotFoundError:
            pass
    if length is None:
        raise ValueError("Prompt generation needs --length when data root is unavailable and config data.window_length is null")
    return int(length), channels


def _denormalize(ts: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    mean = stats["mean"].to(ts.device, dtype=ts.dtype)
    std = stats["std"].to(ts.device, dtype=ts.dtype)
    return ts * std + mean

if __name__ == "__main__":
    main()
