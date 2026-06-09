from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from effectcma_flow.config import load_config, resolve_data_root
from effectcma_flow.data import WeatherSemiSyntheticDataset, collate_effect_batch, compute_train_stats
from effectcma_flow.evaluation.metrics import compute_field_metrics, compute_metrics
from effectcma_flow.evaluation.sampler import euler_sample
from effectcma_flow.evaluation.visualize import dump_sample_plot
from effectcma_flow.models import build_model
from effectcma_flow.training import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint, resolve_device
from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/weather_core.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="artifacts/sample_weather.png")
    parser.add_argument("--text-encoder-mode", default=None, choices=["hash", "precomputed", "hf", "longclip"])
    parser.add_argument("--text-encoder-model", default=None, help="Override hf_model_name or longclip_model_name")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.text_encoder_mode is not None:
        cfg["text_encoder"]["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        key = "longclip_model_name" if str(cfg["text_encoder"].get("mode", "")).lower() == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
    resolve_data_root(cfg, args.data_root)
    result = run_sample(
        cfg,
        checkpoint=args.checkpoint,
        split=args.split,
        index=args.index,
        output=args.output,
        allow_random_init=args.allow_random_init,
        text_encoder_override=args.text_encoder_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_sample(
    cfg: dict,
    *,
    checkpoint: str | None,
    split: str,
    index: int,
    output: str,
    allow_random_init: bool = False,
    text_encoder_override: str | None = None,
) -> dict:
    device = resolve_device(str(cfg["train"].get("device", "auto")))
    payload = None
    if checkpoint is None:
        if not allow_random_init:
            raise ValueError("Sampling requires --checkpoint. Use --allow-random-init only for smoke tests.")
    else:
        payload = load_training_checkpoint(checkpoint, device)
        cfg = checkpoint_eval_config(cfg, payload, text_encoder_override=text_encoder_override)
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"
    stats = checkpoint_stats_or_none(payload) if payload is not None else compute_train_stats(cfg["data"]["root"])
    ds = WeatherSemiSyntheticDataset(
        cfg["data"]["root"],
        split,
        window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)),
        stats=stats,
        effect_types=cfg["data"].get("effect_types"),
        seed=int(cfg["data"].get("seed", 0)) + 30_000,
        include_precomputed_embeddings=include_embeddings,
        precomputed_dim=int(cfg["text_encoder"].get("precomputed_dim", 128)),
    )
    batch = collate_effect_batch([ds[index]])
    model = build_model(cfg, sequence_length=ds.sequence_length, num_channels=ds.num_channels).to(device)
    if payload is not None:
        model.load_state_dict(payload["model"])
    batch = batch_to_device(batch, device)
    model.eval()
    with torch.no_grad():
        pred, aux = euler_sample(model, batch["B"], text_condition_from_batch(batch, text_mode), steps=int(cfg.get("sample", {}).get("steps", 16)))
    metrics = compute_metrics(pred, batch["Y"], batch["B"], batch["mask"])
    metrics.update(compute_field_metrics(aux, batch["mask"], batch["spec"]))
    channel = int(batch["spec"][0]["channels"][0])
    dump_sample_plot(output, batch["B"][0], batch["Y"][0], pred[0], batch["mask"][0], channel=channel)
    return {"slot": batch["slots"][0], "effect_type": batch["effect_type"][0], "plot": output, "metrics": metrics}

if __name__ == "__main__":
    main()
