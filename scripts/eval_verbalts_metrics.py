from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from effectcma_flow.config import load_config, resolve_data_root
from effectcma_flow.data import (
    WeatherRawCaptionDataset,
    WeatherSemiSyntheticDataset,
    collate_effect_batch,
    collate_raw_caption_batch,
    compute_train_stats,
)
from effectcma_flow.evaluation.verbalts_metrics import VerbalTSMetricComputer
from effectcma_flow.models import build_model
from effectcma_flow.training import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/weather_core.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument(
        "--metric-protocol",
        default="verbalts_raw_caption",
        choices=["counterfactual", "verbalts_raw_caption"],
        help="counterfactual uses synthetic Y/effect text; verbalts_raw_caption uses raw train ts/caption reference and generated pred/caption pairs",
    )
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--reference-max-batches", type=int, default=0, help="0 means all train batches")
    parser.add_argument("--n-samples", type=int, default=10, help="Generate this many samples per caption and evaluate their median, matching VerbalTS")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf", "longclip"])
    parser.add_argument("--text-encoder-model", default=None, help="Override hf_model_name or longclip_model_name")
    parser.add_argument("--task-mode", default=None, choices=["text2ts", "edit"])
    parser.add_argument("--verbalts-root", default=None)
    parser.add_argument("--clip-folder", default=None, help="Folder containing model_configs.yaml and clip_model_best.pth")
    parser.add_argument("--clip-config", default=None)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--cache-dir", default="cache/verbalts_metrics")
    parser.add_argument("--caption-shuffle", action="store_true", help="Ablation: shuffle captions within each batch")
    parser.add_argument("--blank-captions", action="store_true", help="Ablation: replace all captions with empty strings")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.text_encoder_mode is not None:
        cfg["text_encoder"]["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        key = "longclip_model_name" if str(cfg["text_encoder"].get("mode", "")).lower() == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
    if args.task_mode is not None:
        cfg.setdefault("task", {})["mode"] = args.task_mode
    resolve_data_root(cfg, args.data_root)
    metrics = run(args, cfg)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def run(args: argparse.Namespace, cfg: dict) -> dict[str, float]:
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    payload = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint_eval_config(cfg, payload, text_encoder_override=args.text_encoder_mode, task_mode_override=args.task_mode)
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    task_mode = str(cfg.get("task", {}).get("mode", "text2ts")).lower()
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    stats = checkpoint_stats_or_none(payload) or compute_train_stats(cfg["data"]["root"])
    batch_size = int(cfg["train"].get("batch_size", 32))

    if args.metric_protocol == "verbalts_raw_caption":
        if task_mode != "text2ts":
            raise ValueError("metric_protocol='verbalts_raw_caption' requires a text2ts checkpoint/config")
        reference_ds = WeatherRawCaptionDataset(
            cfg["data"]["root"],
            "train",
            window_length=cfg["data"].get("window_length"),
            normalize=False,
            seed=int(cfg["data"].get("seed", 0)) + 40_000,
            caption_policy="random",
            include_precomputed_embeddings=False,
        )
        reference_loader = DataLoader(reference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_caption_batch)
        generated_ds = WeatherRawCaptionDataset(
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
        generated_loader = DataLoader(generated_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_caption_batch)
        reference_series_key = "ts"
        reference_text_key = "caption"
        generated_text_key = "caption"
        reference_denormalize = False
    else:
        if task_mode != "edit":
            raise ValueError("metric_protocol='counterfactual' requires an edit checkpoint/config")
        generated_ds = WeatherSemiSyntheticDataset(
            cfg["data"]["root"],
            args.split,
            window_length=cfg["data"].get("window_length"),
            normalize=bool(cfg["data"].get("normalize", True)),
            stats=stats,
            effect_types=cfg["data"].get("effect_types"),
            seed=int(cfg["data"].get("seed", 0)) + 20_000,
            include_precomputed_embeddings=include_embeddings,
            precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
        )
        generated_loader = DataLoader(generated_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_effect_batch)
        reference_ds = WeatherSemiSyntheticDataset(
            cfg["data"]["root"],
            "train",
            window_length=cfg["data"].get("window_length"),
            normalize=bool(cfg["data"].get("normalize", True)),
            stats=stats,
            effect_types=cfg["data"].get("effect_types"),
            seed=int(cfg["data"].get("seed", 0)),
            include_precomputed_embeddings=include_embeddings,
            precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
        )
        reference_loader = DataLoader(reference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_effect_batch)
        reference_series_key = "Y"
        reference_text_key = "full_text"
        generated_text_key = "full_text"
        reference_denormalize = True

    if args.caption_shuffle or args.blank_captions:
        generated_loader = _CaptionTransformLoader(generated_loader, blank=args.blank_captions, shuffle=args.caption_shuffle)

    model = build_model(cfg, sequence_length=generated_ds.sequence_length, num_channels=generated_ds.num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    verbalts_root, clip_config, clip_model = _resolve_verbalts_paths(args)
    cache_metadata = {
        "metric_protocol": args.metric_protocol,
        "data_root": str(Path(cfg["data"]["root"]).resolve()),
        "window_length": cfg["data"].get("window_length"),
        "reference_split": "train",
        "reference_caption_policy": "random" if args.metric_protocol == "verbalts_raw_caption" else "synthetic_effect",
        "reference_seed": int(cfg["data"].get("seed", 0)) + (40_000 if args.metric_protocol == "verbalts_raw_caption" else 0),
        "clip_config": _file_fingerprint(clip_config),
        "clip_model": _file_fingerprint(clip_model),
    }
    computer = VerbalTSMetricComputer(
        verbalts_root=verbalts_root,
        clip_config_path=clip_config,
        clip_model_path=clip_model,
        device=device,
        stats=stats,
    )
    return computer.compute(
        model=model,
        reference_loader=reference_loader,
        generated_loader=generated_loader,
        text_encoder_mode=text_mode,
        steps=int(cfg.get("sample", {}).get("steps", 16)),
        solver=str(cfg.get("sample", {}).get("solver", "euler")),
        task_mode=task_mode,
        noise_scale=float(cfg.get("sample", {}).get("noise_scale", 1.0)),
        cfg_scale=float(cfg.get("sample", {}).get("cfg_scale", 1.0)),
        n_samples=int(args.n_samples),
        caption_slot_strategy=str(cfg.get("train", {}).get("caption_slot_strategy", "single")),
        max_caption_slots=int(cfg.get("train", {}).get("max_caption_slots", 8)),
        include_all_caption_candidates=bool(cfg.get("train", {}).get("include_all_caption_candidates", False)),
        reference_series_key=reference_series_key,
        reference_text_key=reference_text_key,
        generated_text_key=generated_text_key,
        reference_denormalize=reference_denormalize,
        reference_max_batches=_none_if_zero(args.reference_max_batches),
        generated_max_batches=_none_if_zero(args.max_batches),
        cache_dir=str(Path(args.cache_dir) / args.metric_protocol),
        cache_metadata=cache_metadata,
    )


def _resolve_verbalts_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.verbalts_root is None:
        raise ValueError("--verbalts-root is required, e.g. /home/newuser001/huangyu/Research/VerbalTS")
    root = Path(args.verbalts_root)
    if args.clip_folder is not None:
        clip_folder = Path(args.clip_folder)
        return root, clip_folder / "model_configs.yaml", clip_folder / "clip_model_best.pth"
    if args.clip_config is None or args.clip_model is None:
        raise ValueError("Provide either --clip-folder or both --clip-config and --clip-model")
    return root, Path(args.clip_config), Path(args.clip_model)


def _none_if_zero(value: int) -> int | None:
    if value <= 0:
        return None
    return value


def _file_fingerprint(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class _CaptionTransformLoader:
    """Wraps a DataLoader to shuffle or blank captions in each batch."""

    def __init__(self, loader: DataLoader, *, blank: bool = False, shuffle: bool = False):
        self._loader = loader
        self._blank = blank
        self._shuffle = shuffle

    def __iter__(self):
        import random as _random
        for batch in self._loader:
            if self._blank:
                batch = _blank_caption_fields(batch)
            elif self._shuffle:
                batch = _shuffle_caption_fields(batch, rng=_random)
            yield batch

    def __len__(self):
        return len(self._loader)


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


if __name__ == "__main__":
    main()
