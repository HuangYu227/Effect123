"""Export a traceable operator-wise vector-field decomposition during ODE sampling.

Unlike ``export_operator_gate_trace.py``, this utility records the actual
conditional state, candidate operator velocities, gate weights, their weighted
contributions, and the fused velocity at selected main ODE states.  It is for
mechanism analysis only and does not alter model weights or the sampler.
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
        description="Export state and operator-vector-field snapshots along an ODE trajectory."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--solver", choices=("euler", "midpoint", "rk4"), default="rk4")
    parser.add_argument(
        "--record-times",
        default="0.02,0.50,0.96",
        help="Comma-separated flow times in [0, 1) to record, e.g. 0.02,0.50,0.96.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Keep 1.0 for an exact sum of operator contributions and fused velocity.",
    )
    parser.add_argument("--seed", type=int, default=20260710)
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
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0:
            result[key] = value[:count]
        elif isinstance(value, (list, tuple)):
            result[key] = value[:count]
        else:
            result[key] = value
    return result


def parse_record_indices(record_times: str, steps: int) -> list[int]:
    values = [float(part.strip()) for part in record_times.split(",") if part.strip()]
    if not values:
        raise ValueError("--record-times must contain at least one value")
    indices: list[int] = []
    for value in values:
        if not 0.0 <= value < 1.0:
            raise ValueError(f"record time must be in [0,1), got {value}")
        index = min(steps - 1, max(0, int(round(value * steps))))
        if index not in indices:
            indices.append(index)
    return sorted(indices)


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
    if cfg_scale == 1.0:
        return condition, None
    if isinstance(raw_condition, dict) and torch.is_tensor(raw_condition.get("embeddings")):
        null_raw: Any = {"embeddings": torch.zeros_like(raw_condition["embeddings"])}
        if torch.is_tensor(raw_condition.get("mask")):
            null_raw["mask"] = raw_condition["mask"]
    else:
        null_raw = [[""] for _ in range(batch_size)]
    null_condition = model.prepare_condition(null_raw, device=device, dtype=dtype)
    return condition, null_condition


def velocity_at_state(
    model: torch.nn.Module,
    x: torch.Tensor,
    t_value: float,
    condition: Any,
    null_condition: Any | None,
    cfg_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
    conditional_velocity, aux = model(x, t, condition)
    if null_condition is None:
        return conditional_velocity, aux
    null_velocity, _ = model(x, t, null_condition)
    return null_velocity + cfg_scale * (conditional_velocity - null_velocity), aux


@torch.no_grad()
def trace_batch(
    model: torch.nn.Module,
    shape_like: torch.Tensor,
    raw_condition: Any,
    *,
    steps: int,
    solver: str,
    cfg_scale: float,
    record_indices: list[int],
) -> dict[str, torch.Tensor]:
    x = torch.randn_like(shape_like)
    condition, null_condition = prepare_condition_pair(
        model,
        raw_condition,
        batch_size=x.shape[0],
        device=x.device,
        dtype=x.dtype,
        cfg_scale=cfg_scale,
    )
    dt = 1.0 / float(steps)
    records: dict[str, list[torch.Tensor]] = {
        "state": [],
        "candidate_velocity": [],
        "gate": [],
        "weighted_velocity": [],
        "conditional_fused_velocity": [],
        "ode_velocity": [],
        "next_state": [],
    }

    for step in range(steps):
        t0 = step * dt
        k1, aux = velocity_at_state(model, x, t0, condition, null_condition, cfg_scale)
        record = step in record_indices
        if record:
            candidate = aux.get("V")
            gate = aux.get("A_o")
            if not torch.is_tensor(candidate) or candidate.ndim != 4:
                raise RuntimeError("Expected aux['V'] with shape [B,L,C,K] from the operator bank.")
            if not torch.is_tensor(gate) or gate.ndim != 2:
                raise RuntimeError("Expected aux['A_o'] with shape [B,K] from the global operator gate.")
            expanded_gate = gate[:, None, None, :]
            weighted = expanded_gate * candidate
            records["state"].append(x.detach().float().cpu())
            records["candidate_velocity"].append(candidate.detach().float().cpu())
            records["gate"].append(gate.detach().float().cpu())
            records["weighted_velocity"].append(weighted.detach().float().cpu())
            records["conditional_fused_velocity"].append(weighted.sum(dim=-1).detach().float().cpu())
            records["ode_velocity"].append(k1.detach().float().cpu())

        if solver == "euler":
            x = x + dt * k1
        elif solver == "midpoint":
            k2, _ = velocity_at_state(model, x + 0.5 * dt * k1, t0 + 0.5 * dt, condition, null_condition, cfg_scale)
            x = x + dt * k2
        elif solver == "rk4":
            k2, _ = velocity_at_state(model, x + 0.5 * dt * k1, t0 + 0.5 * dt, condition, null_condition, cfg_scale)
            k3, _ = velocity_at_state(model, x + 0.5 * dt * k2, t0 + 0.5 * dt, condition, null_condition, cfg_scale)
            k4, _ = velocity_at_state(model, x + dt * k3, min(1.0, t0 + dt), condition, null_condition, cfg_scale)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unsupported solver: {solver}")

        if record:
            records["next_state"].append(x.detach().float().cpu())

    return {key: torch.stack(value, dim=1) for key, value in records.items()}


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    record_indices = parse_record_indices(args.record_times, args.steps)
    set_seed(args.seed)

    from effectcma_flow.config import load_config, resolve_data_root
    from effectcma_flow.data import WeatherRawCaptionDataset, collate_raw_caption_batch
    from effectcma_flow.models import build_model
    from effectcma_flow.training.checkpoint import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint
    from effectcma_flow.training.utils import resolve_device, text_condition_from_batch

    cfg = load_config(args.config)
    cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.text_encoder_mode is not None:
        cfg.setdefault("text_encoder", {})["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        mode = str(cfg.setdefault("text_encoder", {}).get("mode", "")).lower()
        cfg["text_encoder"]["longclip_model_name" if mode == "longclip" else "hf_model_name"] = args.text_encoder_model
    cfg.setdefault("task", {})["mode"] = "text2ts"
    resolve_data_root(cfg, str(args.data_root))

    device = resolve_device(str(cfg["train"].get("device", "auto")))
    payload = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint_eval_config(cfg, payload, text_encoder_override=args.text_encoder_mode, task_mode_override="text2ts")
    stats = checkpoint_stats_or_none(payload)
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    dataset = WeatherRawCaptionDataset(
        cfg["data"]["root"], args.split, window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)), stats=stats,
        seed=int(cfg["data"].get("seed", 0)) + 20_000, caption_policy="random",
        include_precomputed_embeddings=include_embeddings,
        precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_raw_caption_batch)
    model = build_model(cfg, sequence_length=dataset.sequence_length, num_channels=dataset.num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    max_samples = min(int(args.max_samples), len(dataset)) if int(args.max_samples) > 0 else len(dataset)
    caption_slot_strategy = str(cfg.get("train", {}).get("caption_slot_strategy", "single"))
    max_caption_slots = int(cfg.get("train", {}).get("max_caption_slots", 8))
    include_all_candidates = bool(cfg.get("train", {}).get("include_all_caption_candidates", False))
    chunks: dict[str, list[torch.Tensor]] = {}
    captions: list[str] = []
    exported = 0
    for batch in tqdm(loader, desc="vectorfield-trace", dynamic_ncols=True):
        if exported >= max_samples:
            break
        remaining = max_samples - exported
        if len(batch["caption"]) > remaining:
            batch = take_rows(batch, remaining)
        batch = move_batch(batch, device)
        raw_condition = text_condition_from_batch(
            batch, text_mode, condition_key="caption", caption_slot_strategy=caption_slot_strategy,
            max_caption_slots=max_caption_slots, include_all_caption_candidates=include_all_candidates,
        )
        traced = trace_batch(
            model, batch["Y"], raw_condition, steps=int(args.steps), solver=args.solver,
            cfg_scale=float(args.cfg_scale), record_indices=record_indices,
        )
        for key, value in traced.items():
            chunks.setdefault(key, []).append(value)
        captions.extend(str(value) for value in batch["caption"])
        exported += int(traced["state"].shape[0])

    traced = {key: torch.cat(value, dim=0).numpy().astype(np.float32, copy=False) for key, value in chunks.items()}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **traced,
        flow_time=np.asarray(record_indices, dtype=np.float32) / float(args.steps),
        record_indices=np.asarray(record_indices, dtype=np.int32),
    )
    output.with_suffix(".captions.json").write_text(json.dumps(captions, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "operator_names": list(OPERATOR_NAMES), "config": str(args.config), "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root), "split": args.split, "samples": int(traced["state"].shape[0]),
        "steps": int(args.steps), "solver": args.solver, "cfg_scale": float(args.cfg_scale), "seed": int(args.seed),
        "flow_time": (np.asarray(record_indices, dtype=np.float32) / float(args.steps)).tolist(),
        "state_shape": list(traced["state"].shape), "candidate_velocity_shape": list(traced["candidate_velocity"].shape),
        "max_decomposition_error": float(np.abs(traced["conditional_fused_velocity"] - traced["weighted_velocity"].sum(axis=-1)).max()),
        "max_guided_difference": float(np.abs(traced["ode_velocity"] - traced["conditional_fused_velocity"]).max()),
    }
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] trace: {output}")
    print(f"[saved] captions: {output.with_suffix('.captions.json')}")
    print(f"[saved] summary: {output.with_suffix('.summary.json')}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
