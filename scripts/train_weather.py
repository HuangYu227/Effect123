from __future__ import annotations

import argparse
import json
import math
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
from effectcma_flow.evaluation import contsg_metrics
from effectcma_flow.evaluation.sampler import euler_sample, sample_text2ts
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
    parser.add_argument("--caption-shuffle", action="store_true", help="Ablation: shuffle captions within each batch to break text-ts pairing")
    parser.add_argument("--blank-captions", action="store_true", help="Ablation: replace all captions with empty strings")
    parser.add_argument("--caption-ranking-weight", type=float, default=None, help="Weight for caption-negative velocity ranking loss")
    parser.add_argument("--caption-ranking-margin", type=float, default=None, help="Margin for caption-negative velocity ranking loss")
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
    if args.caption_shuffle:
        cfg["train"]["caption_shuffle"] = True
    if args.blank_captions:
        cfg["train"]["blank_captions"] = True
    if args.caption_ranking_weight is not None:
        cfg["train"]["caption_ranking_weight"] = float(args.caption_ranking_weight)
    if args.caption_ranking_margin is not None:
        cfg["train"]["caption_ranking_margin"] = float(args.caption_ranking_margin)
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
    _num_workers = int(cfg["train"].get("num_workers", 4))
    _pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(cfg["train"].get("eval_batch_size") or cfg["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=_num_workers,
        pin_memory=_pin_memory,
        persistent_workers=_num_workers > 0,
        collate_fn=collate_fn,
    )
    model = build_model(cfg, sequence_length=train_ds.sequence_length, num_channels=train_ds.num_channels).to(device)
    lr = float(cfg["train"].get("lr", 1e-4))
    weight_decay = float(cfg["train"].get("weight_decay", 1e-4))
    # Exclude bias, LayerNorm, GroupNorm, and Embedding weights from weight decay.
    # Applying decay to these parameters actively hurts convergence.
    _no_decay = {"bias", "norm.weight", "norm.bias", "norm1.weight", "norm1.bias",
                 "norm2.weight", "norm2.bias", "layer_norm.weight", "layer_norm.bias",
                 "group_norm.weight", "group_norm.bias"}
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in _no_decay) or param.ndim <= 1:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr,
    )
    # Learning rate scheduler: linear warmup + cosine decay
    warmup_steps = int(cfg["train"].get("warmup_steps", 200))
    min_lr = float(cfg["train"].get("min_lr", 1e-6))
    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(1e-8, current_step / max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    checkpoint_dir = Path(cfg["train"].get("checkpoint_dir", "checkpoints/weather_core"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(cfg["train"].get("max_steps", 1000))
    log_every = int(cfg["train"].get("log_every", 20))
    eval_every = int(cfg["train"].get("eval_every", 200))
    save_every = int(cfg["train"].get("save_every", eval_every))
    best_metric = str(cfg["train"].get("best_metric", "mse"))
    best_metric_mode = str(cfg["train"].get("best_metric_mode", "auto"))
    allow_best_metric_fallback = bool(cfg["train"].get("allow_best_metric_fallback", True))
    step = 0
    progress = tqdm(total=max_steps, desc="train", dynamic_ncols=True)
    last_metrics: dict[str, float] = {}
    best_score: float | None = None
    warned_missing_best_metric = False
    epoch = 0
    train_cfg = cfg.get("train", {})
    caption_slot_strategy = str(train_cfg.get("caption_slot_strategy", "single"))
    max_caption_slots = int(train_cfg.get("max_caption_slots", 1))
    include_all_caption_candidates = bool(train_cfg.get("include_all_caption_candidates", False))
    routing_loss_weight = float(train_cfg.get("routing_loss_weight", 0.0))
    caption_shuffle = bool(train_cfg.get("caption_shuffle", False))
    blank_captions = bool(train_cfg.get("blank_captions", False))
    if routing_loss_weight != 0.0:
        raise ValueError("Clean V6 uses a single Flow Matching loss. Set train.routing_loss_weight: 0.0.")
    import random as _random
    while step < max_steps:
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        epoch += 1
        for batch in train_loader:
            model.train()
            if blank_captions:
                batch = _blank_caption_fields(batch)
            elif caption_shuffle:
                batch = _shuffle_caption_fields(batch, rng=_random)
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
                caption_slot_strategy=caption_slot_strategy,
                max_caption_slots=max_caption_slots,
                include_all_caption_candidates=include_all_caption_candidates,
                condition_dropout_prob=float(cfg["train"].get("condition_dropout_prob", 0.0)),
                regime_ortho_weight=float(cfg["train"].get("regime_ortho_weight", 0.0)),
                bridge_alignment_weight=float(cfg["train"].get("bridge_alignment_weight", 0.0)),
                caption_ranking_weight=float(cfg["train"].get("caption_ranking_weight", 0.0)),
                caption_ranking_margin=float(cfg["train"].get("caption_ranking_margin", 0.05)),
                spectral_loss_weight=float(cfg["train"].get("spectral_loss_weight", 0.0)),
                spectral_loss_type=str(cfg["train"].get("spectral_loss_type", "magnitude")),
                spectral_fft_sizes=cfg["train"].get("spectral_fft_sizes", (16, 32, 64, 128)),
                spectral_distance=str(cfg["train"].get("spectral_distance", "l1")),
                spectral_log_magnitude=bool(cfg["train"].get("spectral_log_magnitude", True)),
                spectral_time_weight_power=float(cfg["train"].get("spectral_time_weight_power", 0.0)),
                operator_balance_weight=float(cfg["train"].get("operator_balance_weight", 0.0)),
            )
            step += 1
            scheduler.step()
            progress.update(1)
            if step % log_every == 0 or step == 1:
                postfix = {
                    "loss": f"{_scalar(out, 'loss'):.4f}",
                    "cfm": f"{_scalar(out, 'loss_cfm'):.4f}",
                    "opH": f"{_scalar(out, 'operator_gate_entropy'):.2f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    "vCos": f"{_scalar(out, 'velocity_cos'):.2f}",
                    "vR": f"{_scalar(out, 'pred_target_rms_ratio'):.2f}",
                }
                tH = _scalar(out, "time_gate_entropy")
                cH = _scalar(out, "channel_gate_entropy")
                if tH == tH:  # not NaN
                    postfix["tH"] = f"{tH:.2f}"
                if cH == cH:  # not NaN
                    postfix["cH"] = f"{cH:.2f}"
                regime_ortho = _scalar(out, "loss_regime_ortho")
                if regime_ortho > 0.0:
                    postfix["rOrtho"] = f"{regime_ortho:.6f}"
                regime_ent = _scalar(out, "regime_entropy")
                if regime_ent == regime_ent:  # not NaN
                    postfix["rH"] = f"{regime_ent:.2f}"
                regime_ent_norm = _scalar(out, "regime_entropy_norm")
                if regime_ent_norm == regime_ent_norm:
                    postfix["rHn"] = f"{regime_ent_norm:.2f}"
                regime_max = _scalar(out, "regime_max_prob")
                if regime_max == regime_max:
                    postfix["rP"] = f"{regime_max:.2f}"
                regime_usage = _usage_summary(out.get("regime_usage"))
                if regime_usage is not None:
                    postfix["rU"] = regime_usage
                bridge_align = _scalar(out, "loss_bridge_alignment")
                if bridge_align > 0.0:
                    postfix["bAlign"] = f"{bridge_align:.4f}"
                caption_rank = _scalar(out, "loss_caption_ranking")
                if caption_rank > 0.0:
                    postfix["cRank"] = f"{caption_rank:.4f}"
                    postfix["cNeg"] = f"{_scalar(out, 'caption_neg_mse'):.3f}"
                    postfix["cAcc"] = f"{_scalar(out, 'caption_ranking_acc'):.2f}"
                spectral = _scalar(out, "loss_spectral")
                if spectral > 0.0:
                    postfix["spec"] = f"{spectral:.4f}"
                sp_entropy = _scalar(out, "spectral_prompt_entropy_norm")
                if sp_entropy == sp_entropy:
                    postfix["spH"] = f"{sp_entropy:.2f}"
                    postfix["spG"] = (
                        f"{_scalar(out, 'spectral_prompt_gate_low'):.2f}/"
                        f"{_scalar(out, 'spectral_prompt_gate_mid'):.2f}/"
                        f"{_scalar(out, 'spectral_prompt_gate_high'):.2f}"
                    )
                op_balance = _scalar(out, "loss_operator_balance")
                if op_balance > 0.0:
                    postfix["opBal"] = f"{op_balance:.4f}"
                text_slot_count = _scalar(out, "text_slot_count")
                if text_slot_count == text_slot_count:
                    postfix["txtJ"] = f"{text_slot_count:.1f}"
                bridge_t2s = _scalar(out, "bridge_text_to_state_entropy_norm")
                if bridge_t2s == bridge_t2s:
                    postfix["bT2S"] = f"{bridge_t2s:.2f}"
                bridge_s2t = _scalar(out, "bridge_state_to_text_entropy_norm")
                if bridge_s2t == bridge_s2t:
                    postfix["bS2T"] = f"{bridge_s2t:.2f}"
                op_max = _scalar(out, "operator_gate_max_prob")
                if op_max == op_max:
                    postfix["opP"] = f"{op_max:.2f}"
                op_usage = _usage_summary(out.get("operator_usage"))
                if op_usage is not None:
                    postfix["opU"] = op_usage
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
                score_name, score = _select_best_metric(
                    metrics,
                    best_metric,
                    allow_fallback=allow_best_metric_fallback,
                )
                if score_name is None or score is None:
                    if not warned_missing_best_metric:
                        progress.write(
                            f"[checkpoint] best metric {best_metric!r} is absent from validation metrics; "
                            "best.pt will not be updated until a comparable metric is available."
                        )
                        warned_missing_best_metric = True
                elif score_name != best_metric and not warned_missing_best_metric:
                    progress.write(
                        f"[checkpoint] best metric {best_metric!r} is absent; using {score_name!r} for best.pt."
                    )
                    warned_missing_best_metric = True
                if score_name is not None and score is not None and _is_better_metric(
                    score,
                    best_score,
                    metric_name=score_name,
                    mode=best_metric_mode,
                ):
                    best_score = score
                    save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, cfg, stats, step)
                    progress.write(f"[checkpoint] saved best.pt at step {step} ({score_name}={score:.6g})")
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


def _statistical_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Encoder-free ConTSG statistical metrics (MDD/ACD/SD/KD) for one batch.

    Cheap enough to run every validation pass: no CLIP encoder, just raw-series
    histograms / moments. The batch's own targets serve as the histogram range
    reference, which is adequate for the relative MDD signal during training.
    """
    real = target.detach().float().cpu().numpy()
    gen = pred.detach().float().cpu().numpy()
    return contsg_metrics.compute_statistical_metrics(real, gen, train_reference=real)


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
            text_condition = text_condition_from_batch(
                batch,
                text_mode,
                condition_key="caption",
                caption_slot_strategy=str(cfg.get("train", {}).get("caption_slot_strategy", "single")),
                max_caption_slots=int(cfg.get("train", {}).get("max_caption_slots", 1)),
                include_all_caption_candidates=bool(cfg.get("train", {}).get("include_all_caption_candidates", False)),
            )
            noise = torch.randn(batch["Y"].shape, dtype=batch["Y"].dtype, generator=eval_generator).to(batch["Y"].device)
            noise = noise * float(cfg.get("sample", {}).get("noise_scale", 1.0))
            pred, aux = sample_text2ts(
                model,
                batch["Y"],
                text_condition,
                solver=str(cfg.get("sample", {}).get("solver", "euler")),
                steps=int(cfg.get("sample", {}).get("steps", 16)),
                noise=noise,
                cfg_scale=float(cfg.get("sample", {}).get("cfg_scale", 1.0)),
            )
            one = compute_text2ts_metrics(pred, batch["Y"])
            one.update(_field_summary(aux))
            if bool(cfg.get("train", {}).get("eval_statistical_metrics", True)):
                one.update(_statistical_metrics(pred, batch["Y"]))
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
    for key in ("gate_raw_cell_mass_mean", "gate_cell_mass_mean", "gate_rescale_factor"):
        if key in aux:
            out[key] = float(aux[key].detach().cpu())
    for key in (
        "regime_entropy",
        "regime_entropy_norm",
        "regime_max_prob",
        "regime_state_weight",
        "regime_context_norm",
        "regime_posterior_uniform",
        "operator_gate_entropy",
        "operator_gate_entropy_norm",
        "operator_gate_max_prob",
        "operator_gate_attention_entropy",
        "operator_gate_attention_entropy_norm",
        "operator_gate_attention_text_mass",
        "operator_gate_attention_state_mass",
        "bridge_alignment_loss",
        "text_slot_count",
        "text_slot_count_min",
        "text_slot_count_max",
        "bridge_text_to_state_entropy",
        "bridge_text_to_state_entropy_norm",
        "bridge_text_to_state_max_prob",
        "bridge_state_to_text_entropy",
        "bridge_state_to_text_entropy_norm",
        "bridge_state_to_text_max_prob",
        "bridge_context_norm",
        "bridge_text_context_norm",
        "bridge_state_context_norm",
        "bridge_expert_context_norm",
        "bridge_channel_context_norm",
        "bridge_scale_context_norm",
        "bridge_stage_context_norm",
        "bridge_memory_token_count",
        "bridge_focal_channel_entropy_norm",
        "bridge_focal_expert_entropy_norm",
        "bridge_focal_scale_entropy_norm",
        "bridge_focal_stage_entropy_norm",
        "bridge_alignment_logit_pos",
        "bridge_alignment_logit_std",
        "bridge_patch_token_count",
        "bridge_budget_pool_active",
        "bridge_token_budget",
        "bridge_connector_patch_merger",
        "bridge_alignment_used_clean",
        "time_long_range_rms",
        "spectral_prompt_entropy",
        "spectral_prompt_entropy_norm",
        "spectral_prompt_gate_low",
        "spectral_prompt_gate_mid",
        "spectral_prompt_gate_high",
        "spectral_prompt_delta_norm",
        "spectral_prompt_attn_entropy_norm",
    ):
        if torch.is_tensor(aux.get(key)) and aux[key].numel() == 1:
            out[key] = float(aux[key].detach().cpu())
    _add_usage_summary(out, aux, "regime_usage")
    _add_usage_summary(out, aux, "operator_usage")
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
        if value.numel() != 1:
            return float("nan")
        return float(value.detach().cpu())
    return float(value)


def _usage_summary(value) -> str | None:
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    flat = value.detach().float().cpu().flatten()
    if flat.numel() > 8:
        flat = flat[:8]
    return "/".join(f"{float(v):.2f}" for v in flat)


def _select_best_metric(
    metrics: dict[str, float],
    requested: str,
    *,
    allow_fallback: bool = True,
) -> tuple[str | None, float | None]:
    """Select a finite validation score for checkpoint ranking.

    Training validation is intentionally lightweight and normally does not run
    VerbalTS. If a config requests a missing VerbalTS metric, fall back to the
    available validation metrics only when allow_fallback=True.
    """
    candidates = [requested]
    if allow_fallback:
        for fallback in ("mse", "mae", "mask_iou", "field_scope_precision"):
            if fallback not in candidates:
                candidates.append(fallback)
    for name in candidates:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            return name, score
    return None, None


def _is_better_metric(score: float, best_score: float | None, *, metric_name: str, mode: str = "auto") -> bool:
    if best_score is None:
        return True
    direction = str(mode).lower()
    if direction == "auto":
        lower_name = metric_name.lower()
        higher_is_better = any(token in lower_name for token in ("cttp", "acc", "iou", "precision", "recall", "f1", "cos"))
        direction = "max" if higher_is_better else "min"
    if direction in {"min", "lower"}:
        return score < best_score
    if direction in {"max", "higher"}:
        return score > best_score
    raise ValueError("train.best_metric_mode must be 'auto', 'min', or 'max'")


def _blank_caption_fields(batch: dict) -> dict:
    if "caption" not in batch:
        return batch
    out = dict(batch)
    out["caption"] = [""] * len(batch["caption"])
    if "caption_candidates" in batch and batch["caption_candidates"] is not None:
        out["caption_candidates"] = [[] for _ in batch["caption"]]
    if "captions" in batch and batch["captions"] is not None:
        out["captions"] = [[] for _ in batch["caption"]]
    return out


def _shuffle_caption_fields(batch: dict, *, rng) -> dict:
    if "caption" not in batch:
        return batch
    indices = list(range(len(batch["caption"])))
    rng.shuffle(indices)
    out = dict(batch)
    out["caption"] = [batch["caption"][idx] for idx in indices]
    if "caption_candidates" in batch and batch["caption_candidates"] is not None:
        out["caption_candidates"] = [batch["caption_candidates"][idx] for idx in indices]
    if "captions" in batch and batch["captions"] is not None:
        out["captions"] = [batch["captions"][idx] for idx in indices]
    return out


def _add_usage_summary(out: dict[str, float], aux: dict[str, torch.Tensor], key: str) -> None:
    value = aux.get(key)
    if not torch.is_tensor(value) or value.numel() == 0:
        return
    flat = value.detach().float().cpu().flatten()
    out[f"{key}_min"] = float(flat.min())
    out[f"{key}_max"] = float(flat.max())
    out[f"{key}_std"] = float(flat.std(unbiased=False))

if __name__ == "__main__":
    main()
