from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from effectcma_flow.config import load_config, resolve_data_root
from effectcma_flow.data import WeatherSemiSyntheticDataset, collate_effect_batch, compute_train_stats
from effectcma_flow.evaluation.verbalts_metrics import VerbalTSMetricComputer
from effectcma_flow.models import build_model
from effectcma_flow.training import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/weather_core.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--reference-max-batches", type=int, default=0, help="0 means all train batches")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf"])
    parser.add_argument("--verbalts-root", default=None)
    parser.add_argument("--clip-folder", default=None, help="Folder containing model_configs.yaml and clip_model_best.pth")
    parser.add_argument("--clip-config", default=None)
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--cache-dir", default="cache/verbalts_metrics")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.text_encoder_mode is not None:
        cfg["text_encoder"]["mode"] = args.text_encoder_mode
    resolve_data_root(cfg, args.data_root)
    metrics = run(args, cfg)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def run(args: argparse.Namespace, cfg: dict) -> dict[str, float]:
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    payload = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint_eval_config(cfg, payload, text_encoder_override=args.text_encoder_mode)
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    stats = checkpoint_stats_or_none(payload) or compute_train_stats(cfg["data"]["root"])
    batch_size = int(cfg["train"].get("batch_size", 32))

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
    reference_loader = DataLoader(reference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_effect_batch)
    generated_loader = DataLoader(generated_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_effect_batch)

    model = build_model(cfg, sequence_length=generated_ds.sequence_length, num_channels=generated_ds.num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    verbalts_root, clip_config, clip_model = _resolve_verbalts_paths(args)
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
        reference_max_batches=_none_if_zero(args.reference_max_batches),
        generated_max_batches=_none_if_zero(args.max_batches),
        cache_dir=args.cache_dir,
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


if __name__ == "__main__":
    main()
