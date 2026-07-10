"""Export per-step global-operator gate traces for a trained TextToTSFlow model.

The script follows the configured ODE solver, but records the conditional gate
at each main integration state.  It is an analysis utility: it never changes
model weights or the general sampler implementation.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


OPERATOR_NAMES = ("temporal", "frequency", "channel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export conditional temporal/frequency/channel gate traces along ODE sampling."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "midpoint", "rk4"), default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument(
        "--caption-shuffle",
        action="store_true",
        help="Cyclically replace each sample's caption with another sample's caption in the same batch.",
    )
    parser.add_argument("--text-encoder-mode", choices=("hash", "precomputed", "hf", "longclip"), default=None)
    parser.add_argument("--text-encoder-model", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def take_rows(batch: dict[str, Any], count: int) -> dict[str, Any]:
    """Keep the first ``count`` samples while preserving text/list fields."""
    if count <= 0:
        raise ValueError("count must be positive")
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0:
            result[key] = value[:count]
        elif isinstance(value, list):
            result[key] = value[:count]
        elif isinstance(value, tuple):
            result[key] = value[:count]
        else:
            result[key] = value
    return result


def zero_precomputed_condition(condition: Any) -> dict[str, torch.Tensor] | None:
    if not isinstance(condition, dict):
        return None
    embeddings = condition.get("embeddings")
    if not torch.is_tensor(embeddings):
        return None
    result = {"embeddings": torch.zeros_like(embeddings)}
    mask = condition.get("mask")
    if torch.is_tensor(mask):
        result["mask"] = mask
    return result


def prepare_condition_pair(
    model: torch.nn.Module,
    raw_condition: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    use_cfg: bool,
) -> tuple[Any, Any | None]:
    condition = model.prepare_condition(raw_condition, device=device, dtype=dtype)
    if not use_cfg:
        return condition, None
    null_raw = zero_precomputed_condition(raw_condition)
    if null_raw is None:
        null_raw = [[""] for _ in range(batch_size)]
    null_condition = model.prepare_condition(null_raw, device=device, dtype=dtype)
    return condition, null_condition


def conditional_velocity(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_value: float,
    condition: Any,
    null_condition: Any | None,
    cfg_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
    velocity, aux = model(x, t, condition)
    if null_condition is None:
        return velocity, aux
    null_velocity, _ = model(x, t, null_condition)
    return null_velocity + cfg_scale * (velocity - null_velocity), aux


@torch.no_grad()
def trace_batch(
    model: torch.nn.Module,
    shape_like: torch.Tensor,
    raw_condition: Any,
    *,
    steps: int,
    solver: str,
    cfg_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow the configured solver and keep gate weights at each main state."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    x = torch.randn_like(shape_like)
    condition, null_condition = prepare_condition_pair(
        model,
        raw_condition,
        batch_size=x.shape[0],
        device=x.device,
        dtype=x.dtype,
        use_cfg=cfg_scale > 1.0,
    )
    dt = 1.0 / float(steps)
    gate_history: list[torch.Tensor] = []

    for step in range(steps):
        t0 = step * dt
        k1, aux = conditional_velocity(model, x, t0, condition, null_condition, cfg_scale)
        gate = aux.get("A_o")
        if not torch.is_tensor(gate) or gate.ndim != 2:
            raise RuntimeError(
                "Expected global-operator gate aux['A_o'] with shape [B,K]. "
                "Use a checkpoint trained with router_mode=global_operator."
            )
        gate_history.append(gate.detach().float().cpu())

        if solver == "euler":
            x = x + dt * k1
        elif solver == "midpoint":
            k2, _ = conditional_velocity(model, x + 0.5 * dt * k1, t0 + 0.5 * dt, condition, null_condition, cfg_scale)
            x = x + dt * k2
        elif solver == "rk4":
            k2, _ = conditional_velocity(model, x + 0.5 * dt * k1, t0 + 0.5 * dt, condition, null_condition, cfg_scale)
            k3, _ = conditional_velocity(model, x + 0.5 * dt * k2, t0 + 0.5 * dt, condition, null_condition, cfg_scale)
            k4, _ = conditional_velocity(model, x + dt * k3, min(1.0, t0 + dt), condition, null_condition, cfg_scale)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unsupported solver: {solver}")

    return torch.stack(gate_history, dim=1), x.detach().float().cpu()


def summarize(gates: np.ndarray) -> dict[str, Any]:
    if gates.ndim != 3:
        raise ValueError(f"gates must be [N,S,K], got {gates.shape}")
    means = gates.mean(axis=(0, 1))
    per_sample = gates.mean(axis=1)
    per_step = gates.mean(axis=0)
    entropy = -(gates * np.log(np.clip(gates, 1e-8, 1.0))).sum(axis=-1)
    dominant = per_sample.argmax(axis=-1)
    return {
        "operator_names": list(OPERATOR_NAMES[: gates.shape[-1]]),
        "mean_gate": means.tolist(),
        "gate_std_over_samples_and_steps": gates.std(axis=(0, 1)).tolist(),
        "mean_within_sample_gate_range": (gates.max(axis=1) - gates.min(axis=1)).mean(axis=0).tolist(),
        "mean_stepwise_gate_change_l1": float(np.abs(np.diff(gates, axis=1)).sum(axis=-1).mean()) if gates.shape[1] > 1 else 0.0,
        "mean_max_probability": float(gates.max(axis=-1).mean()),
        "mean_entropy": float(entropy.mean()),
        "normalized_entropy": float(entropy.mean() / np.log(float(gates.shape[-1]))),
        "dominant_operator_fraction": np.bincount(dominant, minlength=gates.shape[-1]).astype(float).tolist(),
        "per_step_mean_gate": per_step.tolist(),
    }


def main() -> None:
    args = parse_args()
    from effectcma_flow.config import load_config, resolve_data_root
    from effectcma_flow.data import WeatherRawCaptionDataset, collate_raw_caption_batch
    from effectcma_flow.models import build_model
    from effectcma_flow.training.checkpoint import (
        checkpoint_eval_config,
        checkpoint_stats_or_none,
        load_training_checkpoint,
    )
    from effectcma_flow.training.utils import text_condition_from_batch, resolve_device

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.text_encoder_mode is not None:
        cfg.setdefault("text_encoder", {})["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        mode = str(cfg.setdefault("text_encoder", {}).get("mode", "")).lower()
        key = "longclip_model_name" if mode == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
    cfg.setdefault("task", {})["mode"] = "text2ts"
    resolve_data_root(cfg, str(args.data_root))

    device = resolve_device(str(cfg["train"].get("device", "auto")))
    payload = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint_eval_config(cfg, payload, text_encoder_override=args.text_encoder_mode, task_mode_override="text2ts")
    stats = checkpoint_stats_or_none(payload)
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"

    dataset = WeatherRawCaptionDataset(
        cfg["data"]["root"],
        args.split,
        window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)),
        stats=stats,
        seed=int(cfg["data"].get("seed", 0)) + 20_000,
        caption_policy="random",
        include_precomputed_embeddings=include_embeddings,
        precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_raw_caption_batch)
    model = build_model(cfg, sequence_length=dataset.sequence_length, num_channels=dataset.num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    sample_cfg = cfg.get("sample", {})
    steps = int(args.steps or sample_cfg.get("steps", 48))
    solver = str(args.solver or sample_cfg.get("solver", "rk4")).lower()
    cfg_scale = float(args.cfg_scale if args.cfg_scale is not None else sample_cfg.get("cfg_scale", 1.0))
    max_samples = min(int(args.max_samples), len(dataset)) if int(args.max_samples) > 0 else len(dataset)
    caption_slot_strategy = str(cfg.get("train", {}).get("caption_slot_strategy", "single"))
    max_caption_slots = int(cfg.get("train", {}).get("max_caption_slots", 8))
    include_all_caption_candidates = bool(cfg.get("train", {}).get("include_all_caption_candidates", False))

    traces: list[torch.Tensor] = []
    finals: list[torch.Tensor] = []
    source_captions: list[str] = []
    condition_captions: list[str] = []
    exported = 0
    for batch in tqdm(loader, desc="gate-trace", dynamic_ncols=True):
        if exported >= max_samples:
            break
        remaining = max_samples - exported
        batch_size = len(batch["caption"])
        if batch_size > remaining:
            batch = take_rows(batch, remaining)
        batch = move_batch(batch, device)
        original_captions = [str(item) for item in batch["caption"]]
        conditioned_batch = batch
        conditioned_caption_values = original_captions
        if args.caption_shuffle:
            if len(original_captions) < 2:
                raise ValueError("caption shuffle requires at least two samples per batch")
            conditioned_caption_values = original_captions[1:] + original_captions[:1]
            conditioned_batch = {**batch, "caption": conditioned_caption_values}
        raw_condition = text_condition_from_batch(
            conditioned_batch,
            text_mode,
            condition_key="caption",
            caption_slot_strategy=caption_slot_strategy,
            max_caption_slots=max_caption_slots,
            include_all_caption_candidates=include_all_caption_candidates,
        )
        gate_trace, final_state = trace_batch(
            model,
            batch["Y"],
            raw_condition,
            steps=steps,
            solver=solver,
            cfg_scale=cfg_scale,
        )
        traces.append(gate_trace)
        finals.append(final_state)
        source_captions.extend(original_captions)
        condition_captions.extend(conditioned_caption_values)
        exported += int(gate_trace.shape[0])

    gates = torch.cat(traces, dim=0).numpy()
    final_series = torch.cat(finals, dim=0).numpy()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        gates=gates.astype(np.float32, copy=False),
        flow_time=np.arange(steps, dtype=np.float32) / float(steps),
        final_state=final_series.astype(np.float32, copy=False),
    )
    output.with_suffix(".captions.json").write_text(json.dumps(source_captions, ensure_ascii=False, indent=2), encoding="utf-8")
    output.with_suffix(".condition_captions.json").write_text(
        json.dumps(condition_captions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = summarize(gates)
    summary.update(
        {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "split": args.split,
            "samples": int(gates.shape[0]),
            "steps": steps,
            "solver": solver,
            "cfg_scale": cfg_scale,
            "seed": int(args.seed),
            "caption_shuffle": bool(args.caption_shuffle),
        }
    )
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] trace: {output} gates={gates.shape} final_state={final_series.shape}")
    print(f"[saved] captions: {output.with_suffix('.captions.json')}")
    if args.caption_shuffle:
        print(f"[saved] condition captions: {output.with_suffix('.condition_captions.json')}")
    print(f"[saved] summary: {output.with_suffix('.summary.json')}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
