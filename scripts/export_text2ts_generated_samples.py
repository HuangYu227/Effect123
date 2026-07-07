from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export generated time-series samples for visualization. "
            "Use subcommand 'ours' for EffectCMA/TextToTSFlow checkpoints and "
            "'verbalts' for VerbalTS ConditionalGenerator checkpoints."
        )
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    ours = sub.add_parser("ours", help="Export generated samples from an EffectCMA text2ts checkpoint.")
    ours.add_argument("--config", required=True, type=Path)
    ours.add_argument("--checkpoint", required=True, type=Path)
    ours.add_argument("--data-root", required=True, type=Path)
    ours.add_argument("--output", required=True, type=Path)
    ours.add_argument("--split", default="test", choices=("valid", "test"))
    ours.add_argument("--batch-size", type=int, default=256)
    ours.add_argument("--n-samples", type=int, default=10)
    ours.add_argument("--max-batches", type=int, default=0, help="0 means all batches.")
    ours.add_argument("--text-encoder-mode", default=None, choices=("hash", "precomputed", "hf", "longclip"))
    ours.add_argument("--text-encoder-model", default=None)
    ours.add_argument("--task-mode", default="text2ts", choices=("text2ts",))
    ours.add_argument("--seed", type=int, default=1)
    ours.add_argument("--save-all-samples", action="store_true", help="Also save [N,S,L,C] samples next to median output.")

    verbalts = sub.add_parser("verbalts", help="Export generated samples from a VerbalTS run.")
    verbalts.add_argument("--verbalts-root", required=True, type=Path)
    verbalts.add_argument("--data-root", required=True, type=Path)
    verbalts.add_argument("--run-dir", required=True, type=Path)
    verbalts.add_argument("--checkpoint", type=Path, default=None)
    verbalts.add_argument("--diff-config", type=Path, default=None)
    verbalts.add_argument("--cond-config", type=Path, default=None)
    verbalts.add_argument("--output", required=True, type=Path)
    verbalts.add_argument("--batch-size", type=int, default=256)
    verbalts.add_argument("--num-workers", type=int, default=4)
    verbalts.add_argument("--n-samples", type=int, default=10)
    verbalts.add_argument("--max-batches", type=int, default=0, help="0 means all batches.")
    verbalts.add_argument("--sampler", default="ddim", choices=("ddim", "ddpm"))
    verbalts.add_argument("--seed", type=int, default=1)
    verbalts.add_argument("--cond-modal", default="text", choices=("text", "simple_text"))
    verbalts.add_argument("--text-output-type", default="all")
    verbalts.add_argument("--text-pos-emb", default="none")
    verbalts.add_argument("--diff-stage-num", type=int, default=3)
    verbalts.add_argument("--base-patch", type=int, default=None)
    verbalts.add_argument("--multipatch-num", type=int, default=None)
    verbalts.add_argument("--l-patch-len", type=int, default=None)
    verbalts.add_argument("--longclip-root", type=Path, default=Path("/home/newuser001/huangyu/Research/VerbalTS/save/Longclip"))
    verbalts.add_argument("--save-all-samples", action="store_true", help="Also save [N,S,L,C] samples next to median output.")

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_payload(
    *,
    output: Path,
    generated: np.ndarray,
    real: np.ndarray,
    captions: list[str],
    all_samples: np.ndarray | None,
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, generated.astype(np.float32, copy=False))
    np.save(output.with_suffix(".real.npy"), real.astype(np.float32, copy=False))
    np.save(output.with_suffix(".captions.npy"), np.asarray(captions, dtype=object), allow_pickle=True)
    if all_samples is not None:
        np.save(output.with_suffix(".samples.npy"), all_samples.astype(np.float32, copy=False))
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] generated: {output} shape={generated.shape}")
    print(f"[saved] real:      {output.with_suffix('.real.npy')} shape={real.shape}")
    print(f"[saved] captions:  {output.with_suffix('.captions.npy')} n={len(captions)}")
    if all_samples is not None:
        print(f"[saved] samples:   {output.with_suffix('.samples.npy')} shape={all_samples.shape}")
    print(f"[saved] metadata:  {output.with_suffix('.metadata.json')}")


def export_ours(args: argparse.Namespace) -> None:
    from effectcma_flow.config import load_config, resolve_data_root
    from effectcma_flow.data import WeatherRawCaptionDataset, collate_raw_caption_batch
    from effectcma_flow.evaluation.sampler import sample_text2ts
    from effectcma_flow.models import build_model
    from effectcma_flow.training.checkpoint import (
        checkpoint_eval_config,
        checkpoint_stats_or_none,
        load_training_checkpoint,
    )
    from effectcma_flow.training.utils import batch_to_device, resolve_device, text_condition_from_batch

    set_seed(args.seed)
    cfg = load_config(args.config)
    cfg.setdefault("train", {})["batch_size"] = int(args.batch_size)
    if args.text_encoder_mode is not None:
        cfg.setdefault("text_encoder", {})["mode"] = args.text_encoder_mode
    if args.text_encoder_model is not None:
        mode = str(cfg.setdefault("text_encoder", {}).get("mode", "")).lower()
        key = "longclip_model_name" if mode == "longclip" else "hf_model_name"
        cfg["text_encoder"][key] = args.text_encoder_model
    cfg.setdefault("task", {})["mode"] = "text2ts"
    resolve_data_root(cfg, str(args.data_root))

    device = resolve_device(str(cfg["train"].get("device", "auto")))
    payload = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint_eval_config(cfg, payload, text_encoder_override=args.text_encoder_mode, task_mode_override="text2ts")
    stats = checkpoint_stats_or_none(payload)
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))
    include_embeddings = bool(cfg["data"].get("include_precomputed_embeddings", False)) or text_mode == "precomputed"

    ds = WeatherRawCaptionDataset(
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
    loader = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_raw_caption_batch)

    model = build_model(cfg, sequence_length=ds.sequence_length, num_channels=ds.num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    generated_batches: list[torch.Tensor] = []
    real_batches: list[torch.Tensor] = []
    sample_batches: list[torch.Tensor] = []
    captions: list[str] = []
    steps = int(cfg.get("sample", {}).get("steps", 16))
    solver = str(cfg.get("sample", {}).get("solver", "euler"))
    noise_scale = float(cfg.get("sample", {}).get("noise_scale", 1.0))
    cfg_scale = float(cfg.get("sample", {}).get("cfg_scale", 1.0))
    guidance_t_lo = float(cfg.get("sample", {}).get("guidance_t_lo", 0.0))
    guidance_t_hi = float(cfg.get("sample", {}).get("guidance_t_hi", 1.0))
    caption_slot_strategy = str(cfg.get("train", {}).get("caption_slot_strategy", "single"))
    max_caption_slots = int(cfg.get("train", {}).get("max_caption_slots", 8))
    include_all_caption_candidates = bool(cfg.get("train", {}).get("include_all_caption_candidates", False))

    mean = stats["mean"].to(device)
    std = stats["std"].to(device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="ours-gen", dynamic_ncols=True)):
            if args.max_batches and batch_idx >= int(args.max_batches):
                break
            batch = batch_to_device(batch, device)
            text_condition = text_condition_from_batch(
                batch,
                text_mode,
                condition_key="caption",
                caption_slot_strategy=caption_slot_strategy,
                max_caption_slots=max_caption_slots,
                include_all_caption_candidates=include_all_caption_candidates,
            )
            preds = []
            for _ in range(max(1, int(args.n_samples))):
                pred_one, _ = sample_text2ts(
                    model,
                    batch["Y"],
                    text_condition,
                    solver=solver,
                    steps=steps,
                    noise_scale=noise_scale,
                    cfg_scale=cfg_scale,
                    guidance_t_lo=guidance_t_lo,
                    guidance_t_hi=guidance_t_hi,
                )
                preds.append(pred_one.float())
            stacked = torch.stack(preds, dim=1)  # [B,S,L,C]
            median = stacked.median(dim=1).values
            generated_batches.append((median * std + mean).detach().cpu())
            real_batches.append((batch["Y"].float() * std + mean).detach().cpu())
            if args.save_all_samples:
                sample_batches.append((stacked * std + mean).detach().cpu())
            captions.extend([str(x) for x in batch["caption"]])

    generated = torch.cat(generated_batches, dim=0).numpy()
    real = torch.cat(real_batches, dim=0).numpy()
    all_samples = torch.cat(sample_batches, dim=0).numpy() if sample_batches else None
    save_payload(
        output=args.output,
        generated=generated,
        real=real,
        captions=captions,
        all_samples=all_samples,
        metadata={
            "mode": "ours",
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "data_root": str(args.data_root),
            "split": args.split,
            "n_samples": int(args.n_samples),
            "solver": solver,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "noise_scale": noise_scale,
        },
    )


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def configure_verbalts_model(
    diff_cfg: dict[str, Any],
    cond_cfg: dict[str, Any],
    *,
    device: torch.device,
    longclip_root: Path,
    cond_modal: str,
    text_output_type: str,
    text_pos_emb: str,
    diff_stage_num: int,
    base_patch: int | None,
    multipatch_num: int | None,
    l_patch_len: int | None,
) -> None:
    diff_cfg["device"] = str(device)
    diff_cfg["generator_pretrain_path"] = ""
    diffusion = diff_cfg.setdefault("diffusion", {})
    if base_patch is not None:
        diffusion["base_patch"] = int(base_patch)
    else:
        diffusion.setdefault("base_patch", 1)
    if multipatch_num is not None:
        diffusion["multipatch_num"] = int(multipatch_num)
    if l_patch_len is not None:
        diffusion["L_patch_len"] = int(l_patch_len)

    cond_cfg["device"] = str(device)
    cond_cfg["cond_modal"] = cond_modal
    text_cfg = cond_cfg.setdefault("text", {})
    text_cfg["device"] = str(device)
    text_cfg["pretrain_model_path"] = str(longclip_root)
    text_cfg["tokenizer_path"] = str(longclip_root)
    text_cfg["output_type"] = text_output_type
    text_cfg["num_stages"] = int(diff_stage_num)
    text_cfg["pos_emb"] = text_pos_emb


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def export_verbalts(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    verbalts_root = args.verbalts_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or run_dir / "ckpts" / "model_best_loss.pth").expanduser().resolve()
    diff_config = (args.diff_config or run_dir / "model_diff_configs.yaml").expanduser().resolve()
    cond_config = (args.cond_config or run_dir / "model_cond_configs.yaml").expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()

    for path in (verbalts_root, run_dir, checkpoint, diff_config, cond_config, data_root, args.longclip_root):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(verbalts_root))
    from data import GenerationDataset
    from models.conditional_generator import ConditionalGenerator

    device = resolve_device("auto")
    diff_cfg = read_yaml(diff_config)
    cond_cfg = read_yaml(cond_config)
    configure_verbalts_model(
        diff_cfg,
        cond_cfg,
        device=device,
        longclip_root=args.longclip_root.expanduser().resolve(),
        cond_modal=args.cond_modal,
        text_output_type=args.text_output_type,
        text_pos_emb=args.text_pos_emb,
        diff_stage_num=args.diff_stage_num,
        base_patch=args.base_patch,
        multipatch_num=args.multipatch_num,
        l_patch_len=args.l_patch_len,
    )

    dataset = GenerationDataset({"name": "custom", "folder": str(data_root)})
    loader = dataset.get_loader(
        "test",
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        include_self=False,
    )
    model = ConditionalGenerator(diff_cfg, cond_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    generated_batches: list[torch.Tensor] = []
    real_batches: list[torch.Tensor] = []
    sample_batches: list[torch.Tensor] = []
    captions: list[str] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="verbalts-gen", dynamic_ncols=True)):
            if args.max_batches and batch_idx >= int(args.max_batches):
                break
            multi_preds = model.generate(batch, int(args.n_samples), sampler=args.sampler)
            # VerbalTS returns [S,B,C,L]; convert to [B,S,L,C].
            multi_preds = multi_preds.permute(1, 0, 3, 2).float()
            pred = multi_preds.median(dim=1).values
            generated_batches.append(pred.detach().cpu())
            ts = batch["ts"].float()
            real_batches.append(ts.detach().cpu())
            if args.save_all_samples:
                sample_batches.append(multi_preds.detach().cpu())
            captions.extend([str(x) for x in batch["cap"]])

    generated = torch.cat(generated_batches, dim=0).numpy()
    real = torch.cat(real_batches, dim=0).numpy()
    all_samples = torch.cat(sample_batches, dim=0).numpy() if sample_batches else None
    save_payload(
        output=args.output,
        generated=generated,
        real=real,
        captions=captions,
        all_samples=all_samples,
        metadata={
            "mode": "verbalts",
            "verbalts_root": str(verbalts_root),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "diff_config": str(diff_config),
            "cond_config": str(cond_config),
            "data_root": str(data_root),
            "n_samples": int(args.n_samples),
            "sampler": args.sampler,
            "cond_modal": args.cond_modal,
            "base_patch": args.base_patch,
            "multipatch_num": args.multipatch_num,
            "l_patch_len": args.l_patch_len,
        },
    )


def main() -> None:
    args = parse_args()
    if args.mode == "ours":
        export_ours(args)
    elif args.mode == "verbalts":
        export_verbalts(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
