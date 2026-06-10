from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from effectcma_flow.config import load_config, resolve_data_root
from effectcma_flow.data import (
    WeatherRawCaptionDataset,
    WeatherSemiSyntheticDataset,
    collate_effect_batch,
    collate_raw_caption_batch,
    compute_train_stats,
)
from effectcma_flow.evaluation.metrics import average_metric_dicts, compute_field_metrics, compute_metrics, compute_text2ts_metrics
from effectcma_flow.evaluation.sampler import euler_sample, euler_sample_text2ts
from effectcma_flow.models import build_model
from effectcma_flow.training import cfm_train_step, resolve_device, set_seed
from effectcma_flow.training.checkpoint import CHECKPOINT_SCHEMA_VERSION
from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/weather_core.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf", "longclip"])
    parser.add_argument("--text-encoder-model", default=None, help="Override hf_model_name or longclip_model_name")
    parser.add_argument("--task-mode", default=None, choices=["text2ts", "edit"])
    parser.add_argument("--unbounded-field-gate", action="store_true", help="Ablation: replace bounded normalized field gate with unbounded Softplus alpha")
    parser.add_argument("--disable-text-film", action="store_true", help="Ablation: disable text FiLM modulation inside the operator bank")
    parser.add_argument("--disable-flow-time-field", action="store_true", help="Ablation: do not inject flow time into mapper slot tokens")
    parser.add_argument("--operator-norm", default=None, choices=["group", "batch"], help="Ablation: operator expert normalization")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        cfg["train"]["eval_batch_size"] = args.eval_batch_size
    if args.text_encoder_mode is not None:
        cfg["text_encoder"]["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        key = "longclip_model_name" if str(cfg["text_encoder"].get("mode", "")).lower() == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
    if args.task_mode is not None:
        cfg.setdefault("task", {})["mode"] = args.task_mode
    if args.unbounded_field_gate:
        cfg.setdefault("model", {})["mapper_bounded_field_gate"] = False
    if args.disable_text_film:
        cfg.setdefault("model", {})["operator_context_film"] = False
    if args.disable_flow_time_field:
        cfg.setdefault("model", {})["mapper_flow_time_condition"] = False
    if args.operator_norm is not None:
        cfg.setdefault("model", {})["operator_norm"] = args.operator_norm
    if args.checkpoint_dir is not None:
        cfg["train"]["checkpoint_dir"] = args.checkpoint_dir
    resolve_data_root(cfg, args.data_root)
    run_train(cfg)


def run_train(cfg: dict) -> None:
    set_seed(int(cfg["data"].get("seed", 0)))
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    task_mode = str(cfg.get("task", {}).get("mode", "text2ts")).lower()
    if task_mode == "edit" and text_mode == "precomputed":
        raise ValueError("text_encoder.mode='precomputed' is reserved for raw Weather caption embeddings in text2ts mode")
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    root = cfg["data"]["root"]
    stats = compute_train_stats(root)
    train_ds, valid_ds, collate_fn = build_datasets(cfg, stats, include_embeddings=include_embeddings, task_mode=task_mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["train"].get("eval_batch_size") or cfg["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    model = build_model(cfg, sequence_length=train_ds.sequence_length, num_channels=train_ds.num_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 1e-4)), weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    checkpoint_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints/weather_core"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["train"].get("max_steps", 1000))
    log_every = int(cfg["train"].get("log_every", 20))
    eval_every = int(cfg["train"].get("eval_every", 200))
    save_every = int(cfg["train"].get("save_every", eval_every))
    best_metric = str(cfg["train"].get("best_metric", "mse"))
    step = 0
    progress = tqdm(total=max_steps, desc="train", dynamic_ncols=True)
    last_metrics: dict[str, float] = {}
    best_score = float("inf")
    epoch = 0
    while step < max_steps:
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            model.train()
            out = cfm_train_step(
                model,
                batch,
                optimizer,
                device=device,
                grad_clip=float(cfg["train"].get("grad_clip", 1.0)),
                text_encoder_mode=text_mode,
                cfm_loss_mode=str(cfg["train"].get("cfm_loss_mode", "global")),
                task_mode=task_mode,
                noise_scale=float(cfg["train"].get("noise_scale", 1.0)),
            )
            step += 1
            progress.update(1)
            if step % log_every == 0 or step == 1:
                postfix = {
                    "loss": f"{_scalar(out, 'loss'):.4f}",
                    "opH": f"{_scalar(out, 'operator_gate_entropy'):.2f}",
                    "tH": f"{_scalar(out, 'time_gate_entropy'):.2f}",
                    "cH": f"{_scalar(out, 'channel_gate_entropy'):.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                if "loss_inside" in out:
                    postfix["inside"] = f"{_scalar(out, 'loss_inside'):.4f}"
                    postfix["outside"] = f"{_scalar(out, 'loss_outside'):.4f}"
                if last_metrics:
                    postfix.update(
                        {
                            "val_mse": f"{last_metrics.get('mse', float('nan')):.4f}",
                        }
                    )
                    if "mask_iou" in last_metrics:
                        postfix["val_iou"] = f"{last_metrics.get('mask_iou', float('nan')):.3f}"
                    if "field_scope_precision" in last_metrics:
                        postfix["fieldP"] = f"{last_metrics.get('field_scope_precision', float('nan')):.3f}"
                progress.set_postfix(postfix)
            if step % eval_every == 0 or step == max_steps:
                metrics = evaluate(model, valid_loader, cfg, device, max_batches=int(cfg["train"].get("eval_batches", 4)))
                last_metrics = metrics
                progress.write(json.dumps({"step": step, "valid": metrics}, ensure_ascii=False))
                save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, cfg, stats, step)
                if save_every > 0 and (step % save_every == 0 or step == max_steps):
                    save_checkpoint(checkpoint_dir / f"step_{step:08d}.pt", model, optimizer, cfg, stats, step)
                score = metrics.get(best_metric, float("inf"))
                if score < best_score:
                    best_score = score
                    save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, cfg, stats, step)
            if step >= max_steps:
                break
    progress.close()
    save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, cfg, stats, step)


def build_datasets(cfg: dict, stats: dict, *, include_embeddings: bool, task_mode: str):
    root = cfg["data"]["root"]
    common = dict(
        window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)),
        stats=stats,
        include_precomputed_embeddings=include_embeddings,
        precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
    )
    if task_mode == "text2ts":
        train_ds = WeatherRawCaptionDataset(
            root,
            "train",
            seed=int(cfg["data"].get("seed", 0)),
            caption_policy=str(cfg["data"].get("caption_policy", "random")),
            **common,
        )
        valid_ds = WeatherRawCaptionDataset(
            root,
            "valid",
            seed=int(cfg["data"].get("seed", 0)) + 10_000,
            caption_policy="cyclic",
            **common,
        )
        return train_ds, valid_ds, collate_raw_caption_batch
    if task_mode == "edit":
        train_ds = WeatherSemiSyntheticDataset(
            root,
            "train",
            effect_types=cfg["data"].get("effect_types"),
            seed=int(cfg["data"].get("seed", 0)),
            **common,
        )
        valid_ds = WeatherSemiSyntheticDataset(
            root,
            "valid",
            effect_types=cfg["data"].get("effect_types"),
            seed=int(cfg["data"].get("seed", 0)) + 10_000,
            **common,
        )
        return train_ds, valid_ds, collate_effect_batch
    raise ValueError(f"Unknown task.mode {task_mode!r}; expected 'text2ts' or 'edit'")


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: torch.device, *, max_batches: int) -> dict[str, float]:
    model.eval()
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    task_mode = str(cfg.get("task", {}).get("mode", "text2ts")).lower()
    metrics = []
    eval_generator = torch.Generator(device="cpu")
    eval_generator.manual_seed(int(cfg["data"].get("seed", 0)) + int(cfg["train"].get("eval_seed_offset", 50_000)))
    for i, batch in enumerate(loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        if task_mode == "text2ts":
            text_condition = text_condition_from_batch(batch, text_mode, condition_key="caption")
            noise = torch.randn(batch["Y"].shape, dtype=batch["Y"].dtype, generator=eval_generator).to(batch["Y"].device)
            noise = noise * float(cfg.get("sample", {}).get("noise_scale", 1.0))
            pred, aux = euler_sample_text2ts(
                model,
                batch["Y"],
                text_condition,
                steps=int(cfg.get("sample", {}).get("steps", 16)),
                noise=noise,
            )
            one = compute_text2ts_metrics(pred, batch["Y"])
            one.update(_field_summary(aux))
        else:
            text_condition = text_condition_from_batch(batch, text_mode, condition_key="slots")
            pred, aux = euler_sample(model, batch["B"], text_condition, steps=int(cfg.get("sample", {}).get("steps", 16)))
            one = compute_metrics(pred, batch["Y"], batch["B"], batch["mask"])
            one.update(compute_field_metrics(aux, batch["mask"], batch["spec"]))
        metrics.append(one)
    if not metrics:
        raise ValueError("No validation batches were evaluated; check eval_batches and dataset size")
    return average_metric_dicts(metrics)


def _field_summary(aux: dict[str, torch.Tensor]) -> dict[str, float]:
    out: dict[str, float] = {}
    if "A_o" in aux:
        p = aux["A_o"].detach().clamp_min(1e-8)
        out["operator_entropy"] = float((-(p * p.log()).sum(dim=-1).mean()).detach().cpu())
    if "G" in aux:
        out["field_abs_mean"] = float(aux["G"].detach().abs().mean().cpu())
    return out


def save_checkpoint(path: Path, model, optimizer, cfg: dict, stats: dict, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    task_mode = str(cfg.get("task", {}).get("mode", "")).lower()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "task_mode": task_mode,
        "model_class": model.__class__.__name__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "stats": {k: v.cpu() for k, v in stats.items()},
        "step": step,
    }
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _scalar(output: dict, key: str) -> float:
    value = output.get(key)
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)

if __name__ == "__main__":
    main()
