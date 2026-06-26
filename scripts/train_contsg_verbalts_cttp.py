from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SPLITS = ("train", "valid", "test")

SOURCE_TO_CANONICAL = {
    "BlindWays": "blindways",
    "ETTm1": "ettm1",
    "istanbul_traffic": "istanbul_traffic",
    "synth-m": "synth-m",
    "synth-u": "synth-u",
    "synthetic_u": "synthetic_u",
    "Weather": "weather",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train ConTSG CTTP checkpoints on VerbalTS npy datasets from the "
            "EffectCMA/Effect123 repo."
        )
    )
    parser.add_argument("--contsg-root", required=True, type=Path, help="Path to ConTSG-Bench.")
    parser.add_argument(
        "--verbalts-data-root",
        required=True,
        type=Path,
        help="Folder containing VerbalTS dataset folders.",
    )
    parser.add_argument("--longclip-root", required=True, type=Path, help="LongCLIP directory.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help=(
            "Source dataset folder names to train. Example: synth-u synth-m ETTm1 "
            "Weather BlindWays istanbul_traffic. Defaults to every supported folder found."
        ),
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("save/contsg_verbalts_cttp_work"),
        help="Where converted ConTSG-format datasets/configs are written.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("save/contsg_verbalts_cttp"),
        help="Where ConTSG CTTP checkpoints are written.",
    )
    parser.add_argument(
        "--caption-policy",
        choices=("first", "random", "all"),
        default="first",
        help="How to reduce multiple VerbalTS captions per sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Rewrite converted arrays/configs.")
    parser.add_argument("--prepare-only", action="store_true", help="Only convert data and write configs.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing checkpoint without training.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for --eval-only. Only valid with one dataset.")
    parser.add_argument("--skip-final-eval", action="store_true", help="Skip retrieval evaluation after training.")
    args = parser.parse_args()

    contsg_root = args.contsg_root.expanduser().resolve()
    data_root = args.verbalts_data_root.expanduser().resolve()
    longclip_root = args.longclip_root.expanduser().resolve()
    if not contsg_root.exists():
        raise FileNotFoundError(f"ConTSG-Bench root not found: {contsg_root}")
    if not data_root.exists():
        raise FileNotFoundError(f"VerbalTS data root not found: {data_root}")
    if not longclip_root.exists():
        raise FileNotFoundError(f"LongCLIP root not found: {longclip_root}")

    selected = resolve_selected_datasets(data_root, args.datasets)
    if not selected:
        raise RuntimeError(f"No supported VerbalTS datasets found under {data_root}")
    if args.checkpoint is not None and len(selected) != 1:
        raise ValueError("--checkpoint can only be used when exactly one dataset is selected")

    if not args.prepare_only:
        setup_contsg(contsg_root)

    for source_name, canonical_name in selected:
        source_dir = data_root / source_name
        data_dir = args.work_root / "datasets" / canonical_name
        config_path = args.work_root / "configs" / f"cttp_verbalts_{canonical_name}.yaml"
        output_dir = args.output_root / canonical_name
        print(f"[prepare] {source_name} -> {data_dir}")
        shape = convert_dataset(
            source_dir=source_dir,
            target_dir=data_dir,
            caption_policy=args.caption_policy,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        config = build_config(
            canonical_name=canonical_name,
            data_dir=data_dir,
            output_dir=output_dir,
            longclip_root=longclip_root,
            seq_length=shape[1],
            n_var=shape[2],
            args=args,
        )
        write_yaml(config_path, config, overwrite=True)
        print(f"[config] {config_path}")
        if args.prepare_only:
            continue
        if args.eval_only:
            checkpoint = resolve_checkpoint(args.checkpoint, output_dir)
            metrics = evaluate_checkpoint(config, checkpoint=checkpoint, contsg_root=contsg_root, eval_batch_size=args.eval_batch_size)
            write_metrics(output_dir, metrics)
            print_metrics(source_name, checkpoint, metrics)
            continue
        final_checkpoint = train_one(config, contsg_root=contsg_root)
        print(f"[done] {source_name}: {final_checkpoint}")
        if not args.skip_final_eval:
            metrics = evaluate_checkpoint(config, checkpoint=final_checkpoint, contsg_root=contsg_root, eval_batch_size=args.eval_batch_size)
            write_metrics(output_dir, metrics)
            print_metrics(source_name, final_checkpoint, metrics)


def resolve_selected_datasets(data_root: Path, requested: list[str] | None) -> list[tuple[str, str]]:
    if requested is not None:
        unknown = [name for name in requested if name not in SOURCE_TO_CANONICAL]
        if unknown:
            raise ValueError(f"Unsupported dataset names: {unknown}. Supported: {sorted(SOURCE_TO_CANONICAL)}")
        return [(name, SOURCE_TO_CANONICAL[name]) for name in requested]

    out: list[tuple[str, str]] = []
    for source_name, canonical_name in SOURCE_TO_CANONICAL.items():
        if (data_root / source_name).exists():
            out.append((source_name, canonical_name))
    return out


def setup_contsg(contsg_root: Path) -> None:
    if str(contsg_root) not in sys.path:
        sys.path.insert(0, str(contsg_root))
    try:
        import contsg.models  # noqa: F401
        import contsg.data.datasets  # noqa: F401
        from contsg.data.datamodule import BaseDataModule
        from contsg.registry import Registry
    except ImportError as exc:
        raise ImportError(
            "Unable to import ConTSG. Install ConTSG dependencies first, e.g. "
            "`cd <ConTSG-Bench> && pip install -e \".[full,dev]\"`."
        ) from exc

    for canonical_name in SOURCE_TO_CANONICAL.values():
        registry_name = dataset_registry_name(canonical_name)
        if registry_name in Registry.list_datasets():
            continue
        class_name = "".join(part.title() for part in registry_name.replace("-", "_").split("_")) + "DataModule"
        dataset_cls = type(class_name, (BaseDataModule,), {"__doc__": f"Converted VerbalTS dataset {canonical_name}."})
        Registry.register_dataset(registry_name)(dataset_cls)


def convert_dataset(
    *,
    source_dir: Path,
    target_dir: Path,
    caption_policy: str,
    seed: int,
    overwrite: bool,
) -> tuple[int, int, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    split_shapes: dict[str, tuple[int, int, int]] = {}
    train_stats: dict[str, np.ndarray] | None = None
    for split in SPLITS:
        ts, attrs, caps = load_split(source_dir, split)
        ts_out, attrs_out, caps_out = select_captions(
            ts=ts,
            attrs=attrs,
            text_caps=caps,
            policy=caption_policy,
            seed=seed + {"train": 0, "valid": 10_000, "test": 20_000}[split],
        )
        split_shapes[split] = tuple(int(x) for x in ts_out.shape)
        ts_out = ts_out.astype(np.float32)
        if split == "train":
            mean = ts_out.mean(axis=(0, 1), keepdims=True)
            std = ts_out.std(axis=(0, 1), keepdims=True)
            std = np.where(std == 0, 1.0, std).astype(np.float32)
            train_stats = {"mean": mean.astype(np.float32), "std": std}
        save_array(target_dir / f"{split}_ts.npy", ts_out, overwrite)
        save_array(target_dir / f"{split}_attrs_idx.npy", attrs_out, overwrite)
        save_array(target_dir / f"{split}_caps.npy", caps_out.astype(object), overwrite)
        save_array(target_dir / f"{split}_text_caps.npy", caps_out.astype(object), overwrite)

    if train_stats is not None:
        save_npz(target_dir / "normalization_stats.npz", train_stats, overwrite)

    meta = load_meta(source_dir)
    meta.update(
        {
            "source": "verbalts",
            "source_dataset": source_dir.name,
            "caption_policy": caption_policy,
            "converted_layout": "contsg_standard_cttp",
            "seq_length": split_shapes["train"][1],
            "n_var": split_shapes["train"][2],
            "split_shapes": {key: list(value) for key, value in split_shapes.items()},
        }
    )
    write_text(target_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2), overwrite=True)
    return split_shapes["train"]


def load_split(source_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts_path = source_dir / f"{split}_ts.npy"
    attrs_path = source_dir / f"{split}_attrs_idx.npy"
    caps_path = source_dir / f"{split}_text_caps.npy"
    for path in (ts_path, attrs_path, caps_path):
        if not path.exists():
            raise FileNotFoundError(f"Required VerbalTS file missing: {path}")
    ts = np.load(ts_path, allow_pickle=False)
    attrs = np.load(attrs_path, allow_pickle=False)
    caps = np.load(caps_path, allow_pickle=True)
    if ts.ndim == 2:
        ts = ts[..., None]
    if ts.ndim != 3:
        raise ValueError(f"{ts_path} must have shape [N,L,C] or [N,L], got {ts.shape}")
    if attrs.shape[0] != ts.shape[0] or caps.shape[0] != ts.shape[0]:
        raise ValueError(f"Split {split} in {source_dir} has inconsistent row counts")
    return ts, attrs, caps


def select_captions(
    *,
    ts: np.ndarray,
    attrs: np.ndarray,
    text_caps: np.ndarray,
    policy: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    caps = normalize_caps(text_caps)
    if policy == "first":
        return ts, attrs, caps[:, 0]
    if policy == "random":
        rng = np.random.default_rng(seed)
        choices = rng.integers(0, caps.shape[1], size=caps.shape[0])
        return ts, attrs, caps[np.arange(caps.shape[0]), choices]
    if policy == "all":
        n_caps = caps.shape[1]
        return np.repeat(ts, n_caps, axis=0), np.repeat(attrs, n_caps, axis=0), caps.reshape(-1)
    raise ValueError(f"Unknown caption policy: {policy}")


def normalize_caps(caps: np.ndarray) -> np.ndarray:
    if caps.ndim == 1:
        caps = caps[:, None]
    if caps.ndim != 2:
        raise ValueError(f"text_caps must have shape [N] or [N,K], got {caps.shape}")
    out = np.asarray(caps, dtype=object)
    for col in range(out.shape[1]):
        out[:, col] = [str(value).strip() for value in out[:, col]]
    return out


def build_config(
    *,
    canonical_name: str,
    data_dir: Path,
    output_dir: Path,
    longclip_root: Path,
    seq_length: int,
    n_var: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    profile = contsg_cttp_profile(canonical_name, seq_length)
    batch_size = args.batch_size or profile["batch_size"] or choose_batch_size(seq_length, n_var)
    registry_name = dataset_registry_name(canonical_name)
    return {
        "seed": args.seed,
        "device": args.device,
        "output_dir": str(output_dir),
        "train": {
            "epochs": args.epochs,
            "batch_size": batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": "cosine",
            "early_stopping_patience": args.early_stopping_patience,
            "gradient_clip_val": 1.0,
            "num_workers": args.num_workers,
            "pin_memory": True,
            "stages": [
                {
                    "name": "finetune",
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "use_condition": True,
                    "early_stopping_patience": args.early_stopping_patience,
                }
            ],
        },
        "data": {
            "name": registry_name,
            "data_folder": str(data_dir),
            "n_var": int(n_var),
            "seq_length": int(seq_length),
            "normalize": profile["normalize"],
        },
        "model": {
            "name": "cttp",
            "mode": "instance",
            "d_model": 64,
            "coemb_dim": 512,
            "patch_len": profile["patch_len"],
            "stride": profile["stride"],
            "padding": profile["padding"],
            "d_ff": 256,
            "e_layers": 2,
            "factor": 1,
            "activation": "gelu",
            "nheads": 8,
            "dropout": 0.1,
            "textemb_hidden_dim": 1024,
            "pretrain_model_path": str(longclip_root),
            "pretrain_model_dim": 768,
            "loss_type": "ce",
            "temperature": 0.07,
            "normalize_embeddings": profile["normalize_embeddings"],
            "ts_encoder_type": "patchtst_mae",
        },
        "condition": {
            "text": {"enabled": True, "input_dim": 768},
            "attribute": {"enabled": False},
            "label": {"enabled": False},
            "fusion": "concat",
        },
    }


def train_one(config: dict[str, Any], *, contsg_root: Path) -> Path:
    setup_contsg(contsg_root)
    import pytorch_lightning as pl
    from contsg.config.model_validation import validate_model_config
    from contsg.config.schema import ExperimentConfig
    from contsg.registry import Registry
    from contsg.train.multi_stage import MultiStageTrainer

    class CTTPEpochMetricsPrinter(pl.Callback):
        def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            if trainer.sanity_checking:
                return
            metrics = trainer.callback_metrics
            parts = [f"[cttp-metrics] epoch={trainer.current_epoch + 1}", f"step={trainer.global_step}"]
            for label, key in (
                ("train_loss", "train/loss"),
                ("train_ts2text", "train/ts2text"),
                ("train_text2ts", "train/text2ts"),
                ("val_loss", "val/loss"),
                ("val_acc", "val/acc"),
                ("grad_norm", "train/grad_norm"),
                ("grad_norm_max", "train/grad_norm_max"),
                ("param_norm", "train/param_norm"),
            ):
                value = metric_to_float(metrics.get(key))
                if value is not None:
                    parts.append(f"{label}={value:.4f}")
            if trainer.optimizers:
                parts.append(f"lr={trainer.optimizers[0].param_groups[0]['lr']:.2e}")
            print(" ".join(parts), flush=True)

    class VerboseCTTPTrainer(MultiStageTrainer):
        def _build_callbacks(self, stage):  # type: ignore[override]
            callbacks = super()._build_callbacks(stage)
            callbacks.append(CTTPEpochMetricsPrinter())
            return callbacks

    cfg = ExperimentConfig(**config)
    cfg = validate_model_config(cfg, strict_schema=False).config
    pl.seed_everything(cfg.seed, workers=True)
    model_cls = make_configured_optimizer_model(Registry.get_model(cfg.model.name))
    dataset_cls = Registry.get_dataset(cfg.data.name)
    train_config = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory,
        "model_name": cfg.model.name,
        "condition": cfg.condition,
        "seed": cfg.seed,
    }
    datamodule = dataset_cls(cfg.data, train_config=train_config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg.output_dir / "model_configs.yaml", cfg.model_dump(mode="json", exclude_none=True), overwrite=True)
    trainer = VerboseCTTPTrainer(cfg, model_cls, datamodule, checkpoint_root=cfg.output_dir)
    return trainer.train()


def metric_to_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_configured_optimizer_model(model_cls):
    class ConfiguredOptimizerModel(model_cls):
        def configure_optimizers(self):
            import torch

            train_cfg = self.config.train
            weight_decay = float(getattr(train_cfg, "weight_decay", 1e-6))
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=weight_decay,
            )
            scheduler_name = str(getattr(train_cfg, "scheduler", "none")).lower()
            if scheduler_name == "none":
                return optimizer

            params = getattr(train_cfg, "scheduler_params", {}) or {}
            if scheduler_name == "cosine":
                t_max = int(getattr(self.trainer, "max_epochs", 0) or getattr(train_cfg, "epochs", 1) or 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, t_max),
                    eta_min=float(params.get("eta_min", 0.0)),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            if scheduler_name == "step":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=int(params.get("step_size", 30)),
                    gamma=float(params.get("gamma", 0.1)),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            if scheduler_name == "plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=str(params.get("mode", "min")),
                    factor=float(params.get("factor", 0.1)),
                    patience=int(params.get("patience", 10)),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": str(params.get("monitor", "val/loss")),
                        "interval": "epoch",
                        "frequency": 1,
                    },
                }
            raise ValueError(f"Unsupported CTTP scheduler: {scheduler_name}")

    ConfiguredOptimizerModel.__name__ = f"ConfiguredOptimizer{model_cls.__name__}"
    ConfiguredOptimizerModel.__qualname__ = ConfiguredOptimizerModel.__name__
    return ConfiguredOptimizerModel


def resolve_checkpoint(explicit: Path | None, output_dir: Path) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    ckpt_root = output_dir / "checkpoints" / "finetune"
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt_root}")
    candidates = sorted(path for path in ckpt_root.rglob("*.ckpt") if path.name != "last.ckpt")
    if not candidates:
        last = ckpt_root / "last.ckpt"
        if last.exists():
            return last
        raise FileNotFoundError(f"No checkpoint found under {ckpt_root}")

    def score(path: Path) -> float:
        import re

        match = re.search(r"loss=([0-9]+(?:\.[0-9]+)?)\.ckpt$", str(path))
        return float(match.group(1)) if match else float("inf")

    return min(candidates, key=score)


def evaluate_checkpoint(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    contsg_root: Path,
    eval_batch_size: int | None,
) -> dict[str, Any]:
    setup_contsg(contsg_root)
    import torch
    import torch.nn.functional as F
    from contsg.config.model_validation import validate_model_config
    from contsg.config.schema import ExperimentConfig
    from contsg.registry import Registry

    cfg = validate_model_config(ExperimentConfig(**config), strict_schema=False).config
    model_cls = Registry.get_model(cfg.model.name)
    dataset_cls = Registry.get_dataset(cfg.data.name)
    batch_size = int(eval_batch_size or cfg.train.batch_size)
    train_config = {
        "batch_size": batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory,
        "model_name": cfg.model.name,
        "condition": cfg.condition,
        "seed": cfg.seed,
    }
    datamodule = dataset_cls(cfg.data, train_config=train_config)
    datamodule.setup(None)

    model = model_cls.load_from_checkpoint(
        str(checkpoint),
        config=cfg,
        learning_rate=cfg.train.stages[0].lr,
        use_condition=True,
    )
    device = torch.device(cfg.device if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu")
    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metrics: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "dataset": cfg.data.name,
        "batch_size": batch_size,
    }
    with torch.no_grad():
        for split, loader in (("valid", datamodule.val_dataloader()), ("test", datamodule.test_dataloader())):
            metrics[split] = retrieval_metrics(model, loader, device=device)
    return metrics


def retrieval_metrics(model, loader, *, device) -> dict[str, float]:
    import torch
    import torch.nn.functional as F

    ts_embeddings = []
    text_embeddings = []
    raw_losses = []
    n_items = 0
    for batch in loader:
        ts = batch["ts"].to(device).float()
        text = [str(value) for value in batch["cap"]]
        loss_dict = model({"ts": ts, "cap": text})
        raw_losses.append(float(loss_dict["loss"].detach().cpu()) * int(ts.shape[0]))
        ts_embeddings.append(model.get_global_ts_embedding(ts).detach().cpu())
        text_embeddings.append(model.get_text_embedding(text).detach().cpu())
        n_items += int(ts.shape[0])

    if n_items == 0:
        raise ValueError("Cannot evaluate an empty split")
    ts_emb = torch.cat(ts_embeddings, dim=0)
    text_emb = torch.cat(text_embeddings, dim=0)
    sim = torch.mm(ts_emb, text_emb.t())
    labels = torch.arange(sim.shape[0])
    return {
        "n": float(sim.shape[0]),
        "batch_loss": float(sum(raw_losses) / n_items),
        "global_ce": float((F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2.0),
        "diag_sim": float(sim.diag().mean()),
        **rank_metrics(sim, labels, prefix="ts2text"),
        **rank_metrics(sim.t(), labels, prefix="text2ts"),
    }


def rank_metrics(sim, labels, *, prefix: str) -> dict[str, float]:
    import torch

    order = torch.argsort(sim, dim=1, descending=True)
    matches = order.eq(labels[:, None])
    ranks = matches.float().argmax(dim=1) + 1
    return {
        f"{prefix}_top1": float((ranks <= 1).float().mean()),
        f"{prefix}_top5": float((ranks <= 5).float().mean()),
        f"{prefix}_top10": float((ranks <= 10).float().mean()),
        f"{prefix}_mean_rank": float(ranks.float().mean()),
        f"{prefix}_mrr": float((1.0 / ranks.float()).mean()),
    }


def print_metrics(dataset: str, checkpoint: Path, metrics: dict[str, Any]) -> None:
    print(f"[cttp-eval] {dataset}: {checkpoint}", flush=True)
    for split in ("valid", "test"):
        values = metrics[split]
        print(
            "[cttp-eval] {split} n={n:.0f} batch_loss={batch_loss:.4f} global_ce={global_ce:.4f} "
            "diag_sim={diag_sim:.4f} "
            "ts2text@1/@5/@10={ts2text_top1:.4f}/{ts2text_top5:.4f}/{ts2text_top10:.4f} "
            "text2ts@1/@5/@10={text2ts_top1:.4f}/{text2ts_top5:.4f}/{text2ts_top10:.4f} "
            "mean_rank=({ts2text_mean_rank:.1f},{text2ts_mean_rank:.1f}) "
            "mrr=({ts2text_mrr:.4f},{text2ts_mrr:.4f})".format(
                split=split,
                **values,
            ),
            flush=True,
        )


def write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    write_text(
        output_dir / "cttp_retrieval_metrics.json",
        json.dumps(metrics, indent=2, sort_keys=True),
        overwrite=True,
    )


def dataset_registry_name(canonical_name: str) -> str:
    return f"verbalts_{canonical_name}"


def contsg_cttp_profile(canonical_name: str, seq_length: int) -> dict[str, Any]:
    explicit_profiles = {
        "blindways": {
            "batch_size": 128,
            "normalize": True,
            "patch_len": 32,
            "stride": 32,
            "padding": 24,
            "normalize_embeddings": True,
        },
        "weather": {
            "batch_size": 256,
            "normalize": True,
            "patch_len": 32,
            "stride": 32,
            "padding": 24,
            "normalize_embeddings": True,
        },
        "ettm1": {
            "batch_size": 256,
            "normalize": False,
            "patch_len": 4,
            "stride": 4,
            "padding": 0,
            "normalize_embeddings": False,
        },
        "istanbul_traffic": {
            "batch_size": 256,
            "normalize": False,
            "patch_len": 4,
            "stride": 4,
            "padding": 0,
            "normalize_embeddings": False,
        },
        "synth-m": {
            "batch_size": 256,
            "normalize": False,
            "patch_len": 4,
            "stride": 4,
            "padding": 0,
            "normalize_embeddings": False,
        },
        "synth-u": {
            "batch_size": 256,
            "normalize": False,
            "patch_len": 4,
            "stride": 4,
            "padding": 0,
            "normalize_embeddings": False,
        },
        "synthetic_u": {
            "batch_size": 256,
            "normalize": False,
            "patch_len": 4,
            "stride": 4,
            "padding": 0,
            "normalize_embeddings": False,
        },
    }
    if canonical_name in explicit_profiles:
        return explicit_profiles[canonical_name].copy()

    patch_len = choose_patch_len(seq_length)
    return {
        "batch_size": 256,
        "normalize": False,
        "patch_len": patch_len,
        "stride": patch_len,
        "padding": 0,
        "normalize_embeddings": False,
    }


def choose_patch_len(seq_length: int) -> int:
    if seq_length <= 160:
        return 4
    if seq_length <= 256:
        return 8
    return 16


def choose_batch_size(seq_length: int, n_var: int) -> int:
    footprint = seq_length * n_var
    if footprint <= 1024:
        return 128
    if footprint <= 4096:
        return 64
    if footprint <= 16384:
        return 32
    return 8


def load_meta(source_dir: Path) -> dict[str, Any]:
    meta_path = source_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def save_array(path: Path, value: np.ndarray, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def save_npz(path: Path, values: dict[str, np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **values)


def write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_yaml(path: Path, data: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    write_text(path, yaml.safe_dump(data, sort_keys=False, allow_unicode=True), overwrite=True)


if __name__ == "__main__":
    main()
