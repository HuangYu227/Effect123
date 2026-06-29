from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export generated .npy samples from a released VerbalTS text2ts checkpoint."
    )
    parser.add_argument("--verbalts-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path, help="VerbalTS run folder containing ckpts/model_best_loss.pth.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--diff-config", type=Path, default=None)
    parser.add_argument("--cond-config", type=Path, default=None)
    parser.add_argument("--longclip-root", type=Path, default=None)
    parser.add_argument("--cases-dir", type=Path, default=None, help="Directory containing indices.json/captions.json.")
    parser.add_argument("--indices", default=None, help="Comma-separated test indices. Ignored when --cases-dir has indices.json.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cond-modal", choices=("text", "simple_text"), default="text")
    parser.add_argument("--text-output-type", default="all")
    parser.add_argument("--text-pos-emb", default="none")
    parser.add_argument("--diff-stage-num", type=int, default=3)
    parser.add_argument("--base-patch", type=int, default=None)
    parser.add_argument("--multipatch-num", type=int, default=None)
    parser.add_argument("--l-patch-len", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    verbalts_root = args.verbalts_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or run_dir / "ckpts" / "model_best_loss.pth").expanduser().resolve()
    diff_config = (
        args.diff_config
        or verbalts_root / "configs" / "synth-m" / "diff" / "model_text2ts_dep.yaml"
    ).expanduser().resolve()
    cond_config = (
        args.cond_config
        or verbalts_root / "configs" / "synth-m" / "cond" / "text_msmdiffmv.yaml"
    ).expanduser().resolve()
    longclip_root = (args.longclip_root or verbalts_root / "save" / "Longclip").expanduser().resolve()

    for path in (verbalts_root, data_root, run_dir, checkpoint, diff_config, cond_config, longclip_root):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(verbalts_root))

    from data import GenerationDataset
    from models.conditional_generator import ConditionalGenerator
    from scripts.eval_verbalts_official_contsg_metrics import (
        configure_verbalts_model,
        read_yaml,
        resolve_device,
        seed_everything,
    )

    seed_everything(int(args.seed))
    device = resolve_device(args.device)

    diff_cfg = read_yaml(diff_config)
    cond_cfg = read_yaml(cond_config)
    configure_verbalts_model(
        diff_cfg,
        cond_cfg,
        device=device,
        longclip_root=longclip_root,
        cond_modal=args.cond_modal,
        text_output_type=args.text_output_type,
        text_pos_emb=args.text_pos_emb,
        diff_stage_num=int(args.diff_stage_num),
        base_patch=args.base_patch,
        multipatch_num=args.multipatch_num,
        l_patch_len=args.l_patch_len,
    )

    dataset = GenerationDataset({"name": "custom", "folder": str(data_root)})
    test_dataset = dataset.dataset.get_split("test", include_self=False)
    indices = _resolve_indices(args, len(test_dataset))
    loader = DataLoader(
        Subset(test_dataset, indices),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    model = ConditionalGenerator(diff_cfg, cond_cfg).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    generated_chunks = []
    reference_chunks = []
    captions = []
    with torch.no_grad():
        for batch in loader:
            multi_preds = model.generate(batch, int(args.n_samples), sampler=args.sampler)
            # VerbalTS returns [S, B, C, L]; use [B, S, L, C] for EffectCMA plotting.
            multi_preds = multi_preds.permute(1, 0, 3, 2).contiguous().float().cpu()
            generated_chunks.append(multi_preds)
            reference_chunks.append(batch["ts"].float().cpu())
            captions.extend([[str(value)] for value in batch["cap"]])

    generated = torch.cat(generated_chunks, dim=0).numpy().astype(np.float32)
    reference = torch.cat(reference_chunks, dim=0).numpy().astype(np.float32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "generated.npy", generated)
    np.save(args.output_dir / "reference.npy", reference)
    (args.output_dir / "captions.json").write_text(
        json.dumps(captions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "verbalts_root": str(verbalts_root),
                "data_root": str(data_root),
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint),
                "diff_config": str(diff_config),
                "cond_config": str(cond_config),
                "longclip_root": str(longclip_root),
                "indices": indices,
                "n_samples": int(args.n_samples),
                "sampler": args.sampler,
                "generated_shape": list(generated.shape),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "generated": str((args.output_dir / "generated.npy").resolve()),
                "shape": list(generated.shape),
                "indices": indices,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _resolve_indices(args: argparse.Namespace, dataset_len: int) -> list[int]:
    if args.cases_dir is not None:
        path = args.cases_dir / "indices.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            indices = [int(value) for value in data["indices"]]
            _validate_indices(indices, dataset_len)
            return indices
    if args.indices:
        indices = [int(part.strip()) for part in args.indices.split(",") if part.strip()]
        _validate_indices(indices, dataset_len)
        return indices
    raise ValueError("Provide --cases-dir with indices.json or pass --indices")


def _validate_indices(indices: list[int], dataset_len: int) -> None:
    if not indices:
        raise ValueError("No indices were selected")
    bad = [idx for idx in indices if idx < 0 or idx >= dataset_len]
    if bad:
        raise ValueError(f"indices out of range [0,{dataset_len - 1}]: {bad}")


if __name__ == "__main__":
    main()
