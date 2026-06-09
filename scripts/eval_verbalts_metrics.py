from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np

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
    parser.add_argument(
        "--metric-protocol",
        default="counterfactual",
        choices=["counterfactual", "verbalts_raw_caption"],
        help="counterfactual uses synthetic Y/effect text; verbalts_raw_caption uses raw train ts/caption reference and generated pred/caption pairs",
    )
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--reference-max-batches", type=int, default=0, help="0 means all train batches")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf", "longclip"])
    parser.add_argument("--text-encoder-model", default=None, help="Override hf_model_name or longclip_model_name")
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
    if args.text_encoder_model is not None:
        key = "longclip_model_name" if str(cfg["text_encoder"].get("mode", "")).lower() == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
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
    if args.metric_protocol == "verbalts_raw_caption":
        reference_ds = RawWeatherCaptionDataset(cfg["data"]["root"], "train")
        reference_loader = DataLoader(reference_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_raw_caption_batch)
        reference_series_key = "ts"
        reference_text_key = "caption"
        generated_text_key = "caption"
        reference_denormalize = False
    else:
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
        reference_series_key=reference_series_key,
        reference_text_key=reference_text_key,
        generated_text_key=generated_text_key,
        reference_denormalize=reference_denormalize,
        reference_max_batches=_none_if_zero(args.reference_max_batches),
        generated_max_batches=_none_if_zero(args.max_batches),
        cache_dir=str(Path(args.cache_dir) / args.metric_protocol),
    )


class RawWeatherCaptionDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.root = Path(root)
        self.split = split
        self.ts = np.load(self.root / f"{split}_ts.npy", allow_pickle=False).astype(np.float32)
        self.captions = np.load(self.root / f"{split}_text_caps.npy", allow_pickle=False)
        if self.ts.ndim != 3:
            raise ValueError(f"{split}_ts.npy must be [N, L, C], got {self.ts.shape}")
        if self.captions.shape[:2] != (self.ts.shape[0], 3):
            raise ValueError(f"{split}_text_caps.npy must be [N, 3], got {self.captions.shape}")

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    def __getitem__(self, index: int) -> dict:
        cap_idx = int(index) % int(self.captions.shape[1])
        return {
            "ts": torch.from_numpy(self.ts[index]).float(),
            "caption": str(self.captions[index, cap_idx]),
        }


def collate_raw_caption_batch(samples: list[dict]) -> dict:
    return {
        "ts": torch.stack([sample["ts"] for sample in samples]),
        "caption": [sample["caption"] for sample in samples],
    }


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
