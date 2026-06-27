from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_contsg_verbalts_cttp import (
    make_configured_optimizer_model,
    metric_to_float,
    print_metrics,
    retrieval_metrics,
    setup_contsg,
    write_metrics,
    write_yaml,
)

try:
    import pytorch_lightning as pl

    _LightningDataModuleBase = pl.LightningDataModule
except ModuleNotFoundError:
    pl = None
    _LightningDataModuleBase = object


SPLITS = ("train", "valid", "test")

SOURCE_TO_CANONICAL = {
    "BlindWays": "blindways",
    "ETTm1": "ettm1",
    "Traffic": "istanbul_traffic",
    "istanbul_traffic": "istanbul_traffic",
    "synth-m": "synth-m",
    "synth-u": "synth-u",
    "synthetic_u": "synthetic_u",
    "Weather": "weather",
}

CANONICAL_TO_SOURCE = {
    "blindways": "BlindWays",
    "ettm1": "ETTm1",
    "istanbul_traffic": "istanbul_traffic",
    "synth-m": "synth-m",
    "synth-u": "synth-u",
    "synthetic_u": "synthetic_u",
    "weather": "Weather",
}

OFFICIAL_VERBALTS_CTTP_PROFILES = {
    "blindways": {
        "batch_size": 8,
        "normalize": False,
        "patch_len": 16,
        "stride": 16,
        "padding": 0,
        "normalize_embeddings": False,
    },
    "ettm1": {
        "batch_size": 128,
        "normalize": False,
        "patch_len": 4,
        "stride": 4,
        "padding": 0,
        "normalize_embeddings": False,
    },
    "istanbul_traffic": {
        "batch_size": 128,
        "normalize": False,
        "patch_len": 4,
        "stride": 4,
        "padding": 0,
        "normalize_embeddings": False,
    },
    "synth-m": {
        "batch_size": 128,
        "normalize": False,
        "patch_len": 4,
        "stride": 4,
        "padding": 0,
        "normalize_embeddings": False,
    },
    "synth-u": {
        "batch_size": 128,
        "normalize": False,
        "patch_len": 4,
        "stride": 4,
        "padding": 0,
        "normalize_embeddings": False,
    },
    "synthetic_u": {
        "batch_size": 128,
        "normalize": False,
        "patch_len": 4,
        "stride": 4,
        "padding": 0,
        "normalize_embeddings": False,
    },
    "weather": {
        "batch_size": 128,
        "normalize": False,
        "patch_len": 4,
        "stride": 4,
        "padding": 0,
        "normalize_embeddings": False,
    },
}


class VerbalTSRawCTTPDataset(Dataset):
    """Direct reader for released VerbalTS npy splits.

    It intentionally keeps the original files in place. Multi-caption samples are
    expanded virtually through index mapping instead of materializing repeated
    arrays on disk.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        caption_policy: str,
        normalize: bool,
        stats: dict[str, np.ndarray] | None,
        seed: int,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if caption_policy not in {"first", "random", "all", "cyclic"}:
            raise ValueError("caption_policy must be one of {'first', 'random', 'all', 'cyclic'}")
        self.root = Path(root)
        self.split = split
        self.caption_policy = caption_policy
        self.normalize = bool(normalize)
        self.seed = int(seed)

        self.ts = self._load_ts(split)
        self.attrs = self._load_optional_array(f"{split}_attrs_idx.npy", allow_pickle=False)
        self.caps = self._load_caps(split)
        if self.ts.ndim == 2:
            self.ts = self.ts[..., None]
        if self.ts.ndim != 3:
            raise ValueError(f"{self.root / f'{split}_ts.npy'} must have shape [N,L,C] or [N,L], got {self.ts.shape}")
        if self.caps.shape[0] != self.ts.shape[0]:
            raise ValueError(f"{split}_text_caps.npy row count {self.caps.shape[0]} != ts row count {self.ts.shape[0]}")
        if self.attrs is not None and self.attrs.shape[0] != self.ts.shape[0]:
            raise ValueError(f"{split}_attrs_idx.npy row count {self.attrs.shape[0]} != ts row count {self.ts.shape[0]}")
        if self.normalize:
            if stats is None:
                raise ValueError("stats are required when normalize=True")
            self.ts = ((self.ts - stats["mean"]) / stats["std"]).astype(np.float32, copy=False)

    @property
    def sequence_length(self) -> int:
        return int(self.ts.shape[1])

    @property
    def num_channels(self) -> int:
        return int(self.ts.shape[2])

    @property
    def num_captions(self) -> int:
        return int(self.caps.shape[1])

    def __len__(self) -> int:
        n = int(self.ts.shape[0])
        return n * self.num_captions if self.caption_policy == "all" else n

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_idx, cap_idx = self._resolve_index(index)
        seq_len = int(self.ts.shape[1])
        item: dict[str, Any] = {
            "ts": torch.from_numpy(self.ts[sample_idx]),
            "tp": torch.arange(seq_len, dtype=torch.float32),
            "cap": str(self.caps[sample_idx, cap_idx]),
            "idx": torch.tensor(sample_idx, dtype=torch.long),
            "cap_idx": torch.tensor(cap_idx, dtype=torch.long),
        }
        if self.attrs is not None:
            item["attrs"] = torch.from_numpy(np.asarray(self.attrs[sample_idx]))
        return item

    def _resolve_index(self, index: int) -> tuple[int, int]:
        n = int(self.ts.shape[0])
        k = self.num_captions
        if self.caption_policy == "all":
            return int(index // k), int(index % k)
        sample_idx = int(index % n)
        if self.caption_policy == "first":
            return sample_idx, 0
        if self.caption_policy == "cyclic":
            return sample_idx, int(sample_idx % k)
        rng = np.random.default_rng(self.seed + sample_idx + 1_000_003 * split_offset(self.split))
        return sample_idx, int(rng.integers(0, k))

    def _load_ts(self, split: str) -> np.ndarray:
        path = self.root / f"{split}_ts.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing VerbalTS time series file: {path}")
        return np.load(path, allow_pickle=False).astype(np.float32, copy=False)

    def _load_caps(self, split: str) -> np.ndarray:
        candidates = [self.root / f"{split}_text_caps.npy", self.root / f"{split}_caps.npy"]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"Missing VerbalTS caption file. Tried: {candidates}")
        caps = np.load(path, allow_pickle=True)
        if caps.ndim == 1:
            caps = caps[:, None]
        if caps.ndim != 2:
            raise ValueError(f"{path} must have shape [N] or [N,K], got {caps.shape}")
        out = np.asarray(caps, dtype=object)
        for col in range(out.shape[1]):
            out[:, col] = [str(value).strip() for value in out[:, col]]
        return out

    def _load_optional_array(self, name: str, *, allow_pickle: bool) -> np.ndarray | None:
        path = self.root / name
        if not path.exists():
            return None
        return np.load(path, allow_pickle=allow_pickle)


def collate_verbalts_raw(samples: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "ts": torch.stack([sample["ts"] for sample in samples], dim=0).float(),
        "tp": torch.stack([sample["tp"] for sample in samples], dim=0).float(),
        "cap": [str(sample["cap"]) for sample in samples],
        "idx": torch.stack([sample["idx"] for sample in samples], dim=0),
        "cap_idx": torch.stack([sample["cap_idx"] for sample in samples], dim=0),
    }
    if "attrs" in samples[0]:
        out["attrs"] = torch.stack([sample["attrs"] for sample in samples], dim=0)
    return out


class VerbalTSRawCTTPDataModule(_LightningDataModuleBase):
    def __init__(self, config: Any, train_config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.train_config = train_config
        self.train_dataset: VerbalTSRawCTTPDataset | None = None
        self.val_dataset: VerbalTSRawCTTPDataset | None = None
        self.test_dataset: VerbalTSRawCTTPDataset | None = None
        self._stats: dict[str, np.ndarray] | None = None

    def prepare_data(self) -> None:
        root = Path(self.config.data_folder)
        if not root.exists():
            raise FileNotFoundError(f"VerbalTS raw dataset folder not found: {root}")
        for split in SPLITS:
            if not (root / f"{split}_ts.npy").exists():
                raise FileNotFoundError(f"Missing required file: {root / f'{split}_ts.npy'}")
            if not (root / f"{split}_text_caps.npy").exists() and not (root / f"{split}_caps.npy").exists():
                raise FileNotFoundError(f"Missing caption file for split {split} under {root}")

    def setup(self, stage: str | None = None) -> None:
        root = Path(self.config.data_folder)
        if bool(self.config.normalize):
            self._stats = compute_train_stats(root)
        if stage == "fit" or stage is None:
            self.train_dataset = VerbalTSRawCTTPDataset(
                root,
                "train",
                caption_policy=str(self.train_config.get("train_caption_policy", "first")),
                normalize=bool(self.config.normalize),
                stats=self._stats,
                seed=int(self.train_config.get("seed", 42)),
            )
            self.val_dataset = VerbalTSRawCTTPDataset(
                root,
                "valid",
                caption_policy=str(self.train_config.get("eval_caption_policy", "first")),
                normalize=bool(self.config.normalize),
                stats=self._stats,
                seed=int(self.train_config.get("seed", 42)) + 10_000,
            )
        if stage == "test" or stage is None:
            self.test_dataset = VerbalTSRawCTTPDataset(
                root,
                "test",
                caption_policy=str(self.train_config.get("eval_caption_policy", "first")),
                normalize=bool(self.config.normalize),
                stats=self._stats,
                seed=int(self.train_config.get("seed", 42)) + 20_000,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule.setup('fit') must be called before train_dataloader")
        return DataLoader(
            self.train_dataset,
            batch_size=self._batch_size(),
            shuffle=True,
            num_workers=self._num_workers(),
            pin_memory=self._pin_memory(),
            drop_last=True,
            collate_fn=collate_verbalts_raw,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule.setup('fit') must be called before val_dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self._eval_batch_size(),
            shuffle=False,
            num_workers=self._num_workers(),
            pin_memory=self._pin_memory(),
            drop_last=False,
            collate_fn=collate_verbalts_raw,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule.setup('test') must be called before test_dataloader")
        return DataLoader(
            self.test_dataset,
            batch_size=self._eval_batch_size(),
            shuffle=False,
            num_workers=self._num_workers(),
            pin_memory=self._pin_memory(),
            drop_last=False,
            collate_fn=collate_verbalts_raw,
        )

    def _batch_size(self) -> int:
        return int(getattr(self.config, "batch_size", None) or self.train_config.get("batch_size", 128))

    def _eval_batch_size(self) -> int:
        return int(self.train_config.get("eval_batch_size") or self._batch_size())

    def _num_workers(self) -> int:
        return int(self.train_config.get("num_workers", 4))

    def _pin_memory(self) -> bool:
        return bool(self.train_config.get("pin_memory", True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a ConTSG-style CTTP checkpoint directly on raw VerbalTS npy datasets."
    )
    parser.add_argument("--contsg-root", required=True, type=Path, help="Path to ConTSG-Bench.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Raw VerbalTS dataset folder.")
    parser.add_argument("--dataset-name", default=None, help="Dataset name, inferred from --dataset-root if omitted.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for the CTTP experiment.")
    parser.add_argument("--longclip-root", required=True, type=Path, help="LongCLIP directory used by ConTSG CTTP.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("cosine", "step", "plateau", "none"), default="cosine")
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-caption-policy", choices=("first", "random", "all", "cyclic"), default="first")
    parser.add_argument("--eval-caption-policy", choices=("first", "random", "all", "cyclic"), default="first")
    parser.add_argument("--patch-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--padding", type=int, default=None)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--loss-type", choices=("ce", "contrastive", "supcon"), default="ce")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()

    contsg_root = args.contsg_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    longclip_root = args.longclip_root.expanduser().resolve()
    if not contsg_root.exists():
        raise FileNotFoundError(f"ConTSG-Bench root not found: {contsg_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not longclip_root.exists():
        raise FileNotFoundError(f"LongCLIP root not found: {longclip_root}")

    shape = infer_dataset_shape(dataset_root)
    canonical_name = canonicalize_dataset_name(args.dataset_name or dataset_root.name)
    config = build_raw_config(
        canonical_name=canonical_name,
        dataset_root=dataset_root,
        output_dir=args.output_dir.expanduser().resolve(),
        longclip_root=longclip_root,
        seq_length=shape[1],
        n_var=shape[2],
        args=args,
    )
    config_path = Path(config["output_dir"]) / "model_configs.yaml"
    write_yaml(config_path, config, overwrite=True)
    print(f"[config] {config_path}", flush=True)
    print(
        "[dataset] {name} root={root} train_shape={shape} train_caption_policy={train_policy} "
        "eval_caption_policy={eval_policy}".format(
            name=canonical_name,
            root=dataset_root,
            shape=list(shape),
            train_policy=args.train_caption_policy,
            eval_policy=args.eval_caption_policy,
        ),
        flush=True,
    )
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required with --eval-only")
        metrics = evaluate_raw_checkpoint(
            config,
            checkpoint=args.checkpoint.expanduser().resolve(),
            contsg_root=contsg_root,
            eval_batch_size=args.eval_batch_size,
            eval_caption_policy=args.eval_caption_policy,
        )
        write_metrics(Path(config["output_dir"]), metrics)
        print_metrics(canonical_name, args.checkpoint, metrics)
        return

    checkpoint = train_one_raw(
        config,
        contsg_root=contsg_root,
        train_caption_policy=args.train_caption_policy,
        eval_caption_policy=args.eval_caption_policy,
        eval_batch_size=args.eval_batch_size,
    )
    print(f"[done] {canonical_name}: {checkpoint}", flush=True)
    if not args.skip_final_eval:
        metrics = evaluate_raw_checkpoint(
            config,
            checkpoint=checkpoint,
            contsg_root=contsg_root,
            eval_batch_size=args.eval_batch_size,
            eval_caption_policy=args.eval_caption_policy,
        )
        write_metrics(Path(config["output_dir"]), metrics)
        print_metrics(canonical_name, checkpoint, metrics)


def train_one_raw(
    config: dict[str, Any],
    *,
    contsg_root: Path,
    train_caption_policy: str,
    eval_caption_policy: str,
    eval_batch_size: int | None,
) -> Path:
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

    cfg = validate_model_config(ExperimentConfig(**config), strict_schema=False).config
    pl.seed_everything(cfg.seed, workers=True)
    train_config = {
        "batch_size": cfg.train.batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory,
        "model_name": cfg.model.name,
        "condition": cfg.condition,
        "seed": cfg.seed,
        "train_caption_policy": train_caption_policy,
        "eval_caption_policy": eval_caption_policy,
    }
    datamodule = VerbalTSRawCTTPDataModule(cfg.data, train_config=train_config)
    model_cls = make_configured_optimizer_model(Registry.get_model(cfg.model.name))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(cfg.output_dir / "model_configs.yaml", cfg.model_dump(mode="json", exclude_none=True), overwrite=True)
    write_yaml(
        cfg.output_dir / "raw_verbalts_train_config.yaml",
        {
            "train_caption_policy": train_caption_policy,
            "eval_caption_policy": eval_caption_policy,
            "eval_batch_size": eval_batch_size,
            "data_folder": str(cfg.data.data_folder),
        },
        overwrite=True,
    )
    trainer = VerboseCTTPTrainer(cfg, model_cls, datamodule, checkpoint_root=cfg.output_dir)
    return trainer.train()


def evaluate_raw_checkpoint(
    config: dict[str, Any],
    *,
    checkpoint: Path,
    contsg_root: Path,
    eval_batch_size: int | None,
    eval_caption_policy: str,
) -> dict[str, Any]:
    setup_contsg(contsg_root)
    import torch
    from contsg.config.model_validation import validate_model_config
    from contsg.config.schema import ExperimentConfig
    from contsg.registry import Registry

    cfg = validate_model_config(ExperimentConfig(**config), strict_schema=False).config
    model_cls = Registry.get_model(cfg.model.name)
    batch_size = int(eval_batch_size or cfg.train.batch_size)
    train_config = {
        "batch_size": batch_size,
        "eval_batch_size": batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": cfg.train.pin_memory,
        "model_name": cfg.model.name,
        "condition": cfg.condition,
        "seed": cfg.seed,
        "train_caption_policy": eval_caption_policy,
        "eval_caption_policy": eval_caption_policy,
    }
    datamodule = VerbalTSRawCTTPDataModule(cfg.data, train_config=train_config)
    datamodule.setup(None)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
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
        "eval_caption_policy": eval_caption_policy,
    }
    with torch.no_grad():
        for split, loader in (("valid", datamodule.val_dataloader()), ("test", datamodule.test_dataloader())):
            metrics[split] = retrieval_metrics(model, loader, device=device)
    return metrics


def build_raw_config(
    *,
    canonical_name: str,
    dataset_root: Path,
    output_dir: Path,
    longclip_root: Path,
    seq_length: int,
    n_var: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    profile = official_profile(canonical_name)
    batch_size = int(args.batch_size or profile["batch_size"])
    patch_len = int(args.patch_len if args.patch_len is not None else profile["patch_len"])
    stride = int(args.stride if args.stride is not None else profile["stride"])
    padding = int(args.padding if args.padding is not None else profile["padding"])
    normalize = bool(profile["normalize"] if args.normalize is None else args.normalize)
    normalize_embeddings = bool(
        profile["normalize_embeddings"] if args.normalize_embeddings is None else args.normalize_embeddings
    )
    return {
        "seed": int(args.seed),
        "device": str(args.device),
        "output_dir": str(output_dir),
        "train": {
            "epochs": int(args.epochs),
            "batch_size": batch_size,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "scheduler": str(args.scheduler),
            "early_stopping_patience": int(args.early_stopping_patience),
            "gradient_clip_val": 1.0,
            "num_workers": int(args.num_workers),
            "pin_memory": True,
            "stages": [
                {
                    "name": "finetune",
                    "epochs": int(args.epochs),
                    "lr": float(args.lr),
                    "use_condition": True,
                    "early_stopping_patience": int(args.early_stopping_patience),
                }
            ],
        },
        "data": {
            "name": f"verbalts_raw_{canonical_name}",
            "data_folder": str(dataset_root),
            "n_var": int(n_var),
            "seq_length": int(seq_length),
            "normalize": normalize,
        },
        "model": {
            "name": "cttp",
            "mode": "instance",
            "d_model": 64,
            "coemb_dim": 512,
            "patch_len": patch_len,
            "stride": stride,
            "padding": padding,
            "d_ff": 256,
            "e_layers": 2,
            "factor": 1,
            "activation": "gelu",
            "nheads": 8,
            "dropout": 0.1,
            "textemb_hidden_dim": 1024,
            "pretrain_model_path": str(longclip_root),
            "pretrain_model_dim": 768,
            "loss_type": str(args.loss_type).lower(),
            "temperature": float(args.temperature),
            "normalize_embeddings": normalize_embeddings,
            "ts_encoder_type": "patchtst_mae",
        },
        "condition": {
            "text": {"enabled": True, "input_dim": 768},
            "attribute": {"enabled": False},
            "label": {"enabled": False},
            "fusion": "concat",
        },
    }


def infer_dataset_shape(dataset_root: Path) -> tuple[int, int, int]:
    path = dataset_root / "train_ts.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing train split: {path}")
    ts = np.load(path, mmap_mode="r", allow_pickle=False)
    if ts.ndim == 2:
        return int(ts.shape[0]), int(ts.shape[1]), 1
    if ts.ndim == 3:
        return int(ts.shape[0]), int(ts.shape[1]), int(ts.shape[2])
    raise ValueError(f"{path} must have shape [N,L,C] or [N,L], got {ts.shape}")


def compute_train_stats(dataset_root: Path) -> dict[str, np.ndarray]:
    ts = np.load(dataset_root / "train_ts.npy", allow_pickle=False).astype(np.float32, copy=False)
    if ts.ndim == 2:
        ts = ts[..., None]
    mean = ts.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = ts.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.where(std == 0, 1.0, std).astype(np.float32)
    return {"mean": mean, "std": std}


def canonicalize_dataset_name(name: str) -> str:
    if name in SOURCE_TO_CANONICAL:
        return SOURCE_TO_CANONICAL[name]
    lowered = name.lower()
    if lowered in OFFICIAL_VERBALTS_CTTP_PROFILES:
        return lowered
    if lowered == "traffic":
        return "istanbul_traffic"
    if lowered == "synthetic-m":
        return "synth-m"
    if lowered in {"synthetic-u", "synthetic_u"}:
        return "synthetic_u"
    raise ValueError(f"Unsupported dataset name {name!r}. Supported: {sorted(SOURCE_TO_CANONICAL)}")


def official_profile(canonical_name: str) -> dict[str, Any]:
    if canonical_name not in OFFICIAL_VERBALTS_CTTP_PROFILES:
        raise ValueError(f"No CTTP profile registered for dataset {canonical_name!r}")
    return OFFICIAL_VERBALTS_CTTP_PROFILES[canonical_name].copy()


def split_offset(split: str) -> int:
    return {"train": 0, "valid": 10_000, "test": 20_000}[split]


if __name__ == "__main__":
    main()
