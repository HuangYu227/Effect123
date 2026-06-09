from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from effectcma_flow.config import load_config, resolve_data_root
from effectcma_flow.data import WeatherSemiSyntheticDataset, collate_effect_batch, compute_train_stats
from effectcma_flow.evaluation.metrics import average_metric_dicts, compute_field_metrics, compute_metrics
from effectcma_flow.evaluation.sampler import euler_sample
from effectcma_flow.models import build_model
from effectcma_flow.training import cfm_train_step, resolve_device, set_seed
from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/weather_core.yaml")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf", "longclip"])
    parser.add_argument("--text-encoder-model", default=None, help="Override hf_model_name or longclip_model_name")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.text_encoder_mode is not None:
        cfg["text_encoder"]["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        key = "longclip_model_name" if str(cfg["text_encoder"].get("mode", "")).lower() == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
    if args.checkpoint_dir is not None:
        cfg["train"]["checkpoint_dir"] = args.checkpoint_dir
    resolve_data_root(cfg, args.data_root)
    run_train(cfg)


def run_train(cfg: dict) -> None:
    set_seed(int(cfg["data"].get("seed", 0)))
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    root = cfg["data"]["root"]
    stats = compute_train_stats(root)
    train_ds = WeatherSemiSyntheticDataset(
        root,
        "train",
        window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)),
        stats=stats,
        effect_types=cfg["data"].get("effect_types"),
        seed=int(cfg["data"].get("seed", 0)),
        include_precomputed_embeddings=include_embeddings,
        precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
    )
    valid_ds = WeatherSemiSyntheticDataset(
        root,
        "valid",
        window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)),
        stats=stats,
        effect_types=cfg["data"].get("effect_types"),
        seed=int(cfg["data"].get("seed", 0)) + 10_000,
        include_precomputed_embeddings=include_embeddings,
        precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_effect_batch,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_effect_batch,
    )
    model = build_model(cfg, sequence_length=train_ds.sequence_length, num_channels=train_ds.num_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 1e-4)), weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    checkpoint_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints/weather_core"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["train"].get("max_steps", 1000))
    log_every = int(cfg["train"].get("log_every", 20))
    eval_every = int(cfg["train"].get("eval_every", 200))
    step = 0
    progress = tqdm(total=max_steps, desc="train", dynamic_ncols=True)
    last_metrics: dict[str, float] = {}
    while step < max_steps:
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
            )
            step += 1
            progress.update(1)
            if step % log_every == 0 or step == 1:
                postfix = {
                    "loss": f"{_scalar(out, 'loss'):.4f}",
                    "inside": f"{_scalar(out, 'loss_inside'):.4f}",
                    "outside": f"{_scalar(out, 'loss_outside'):.4f}",
                    "opH": f"{_scalar(out, 'operator_gate_entropy'):.2f}",
                    "tH": f"{_scalar(out, 'time_gate_entropy'):.2f}",
                    "cH": f"{_scalar(out, 'channel_gate_entropy'):.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
                if last_metrics:
                    postfix.update(
                        {
                            "val_mse": f"{last_metrics.get('mse', float('nan')):.4f}",
                            "val_iou": f"{last_metrics.get('mask_iou', float('nan')):.3f}",
                            "fieldP": f"{last_metrics.get('field_scope_precision', float('nan')):.3f}",
                        }
                    )
                progress.set_postfix(postfix)
            if step % eval_every == 0 or step == max_steps:
                metrics = evaluate(model, valid_loader, cfg, device, max_batches=int(cfg["train"].get("eval_batches", 4)))
                last_metrics = metrics
                progress.write(json.dumps({"step": step, "valid": metrics}, ensure_ascii=False))
                save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, cfg, stats, step)
            if step >= max_steps:
                break
    progress.close()
    save_checkpoint(checkpoint_dir / "latest.pt", model, optimizer, cfg, stats, step)


@torch.no_grad()
def evaluate(model, loader, cfg: dict, device: torch.device, *, max_batches: int) -> dict[str, float]:
    model.eval()
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    metrics = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        text_condition = text_condition_from_batch(batch, text_mode)
        pred, aux = euler_sample(model, batch["B"], text_condition, steps=int(cfg.get("sample", {}).get("steps", 16)))
        one = compute_metrics(pred, batch["Y"], batch["B"], batch["mask"])
        one.update(compute_field_metrics(aux, batch["mask"], batch["spec"]))
        metrics.append(one)
    return average_metric_dicts(metrics)


def save_checkpoint(path: Path, model, optimizer, cfg: dict, stats: dict, step: int) -> None:
    payload = {
        "schema_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
        "stats": {k: v.cpu() for k, v in stats.items()},
        "step": step,
    }
    torch.save(payload, path)


def _scalar(output: dict, key: str) -> float:
    value = output.get(key)
    if value is None:
        return float("nan")
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)

if __name__ == "__main__":
    main()
