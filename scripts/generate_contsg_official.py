from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export selected generated .npy samples from a ConTSG-Bench experiment directory."
    )
    parser.add_argument("--contsg-root", required=True, type=Path)
    parser.add_argument("--experiment", required=True, type=Path, help="ConTSG experiment directory with config.yaml.")
    parser.add_argument("--checkpoint", default="best", help="'best', 'last', or checkpoint filename.")
    parser.add_argument("--cases-dir", type=Path, default=None, help="Directory containing indices.json.")
    parser.add_argument("--indices", default=None, help="Comma-separated test indices.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--sampler", default="ddim")
    parser.add_argument("--max-batches", type=int, default=0, help="0 means enough batches to cover selected indices.")
    args = parser.parse_args()

    contsg_root = args.contsg_root.expanduser().resolve()
    experiment = args.experiment.expanduser().resolve()
    if not contsg_root.exists():
        raise FileNotFoundError(contsg_root)
    if not experiment.exists():
        raise FileNotFoundError(experiment)
    if not (experiment / "config.yaml").exists():
        raise FileNotFoundError(
            f"{experiment}/config.yaml not found. --experiment must be a ConTSG generator experiment directory, "
            "not a CTTP resource folder."
        )

    sys.path.insert(0, str(contsg_root))

    from contsg.eval import Evaluator

    device = _resolve_device(args.device)
    evaluator = Evaluator.from_experiment(
        experiment,
        checkpoint=str(args.checkpoint),
        device=device,
        cache_only=False,
    )
    if evaluator.model is None:
        raise RuntimeError("ConTSG evaluator did not load a model")
    if evaluator.model.__class__.__name__.lower().startswith("cttp"):
        raise RuntimeError("Loaded model is CTTP, which is an evaluator, not a generator")
    if not hasattr(evaluator.model, "generate"):
        raise RuntimeError(f"Loaded ConTSG model {type(evaluator.model).__name__} has no generate() method")

    test_loader = evaluator.datamodule.test_dataloader()
    dataset_len = len(test_loader.dataset)
    indices = _resolve_indices(args, dataset_len)
    selected = set(indices)

    generated_by_index: dict[int, torch.Tensor] = {}
    reference_by_index: dict[int, torch.Tensor] = {}
    captions_by_index: dict[int, list[str]] = {}

    seen = 0
    batches = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_size = int(batch["ts"].shape[0])
            batch_indices = list(range(seen, seen + batch_size))
            local_positions = [pos for pos, idx in enumerate(batch_indices) if idx in selected]
            if local_positions:
                moved = evaluator._move_batch_to_device(batch)
                multi_preds = evaluator._generate_predictions(
                    moved,
                    n_samples=int(args.n_samples),
                    sampler=str(args.sampler),
                )
                # Evaluator returns [S, B, L, F]; save [B, S, L, F].
                multi_preds = multi_preds.permute(1, 0, 2, 3).detach().cpu().float()
                real = batch["ts"].detach().cpu().float()
                for pos in local_positions:
                    idx = batch_indices[pos]
                    generated_by_index[idx] = multi_preds[pos]
                    reference_by_index[idx] = real[pos]
                    captions_by_index[idx] = _caption_from_batch(batch, pos)
            seen += batch_size
            batches += 1
            if len(generated_by_index) == len(selected):
                break
            if int(args.max_batches) > 0 and batches >= int(args.max_batches):
                break

    missing = [idx for idx in indices if idx not in generated_by_index]
    if missing:
        raise RuntimeError(f"Did not generate selected indices: {missing}")

    generated = torch.stack([generated_by_index[idx] for idx in indices], dim=0).numpy().astype(np.float32)
    reference = torch.stack([reference_by_index[idx] for idx in indices], dim=0).numpy().astype(np.float32)
    captions = [captions_by_index.get(idx, [""]) for idx in indices]

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
                "contsg_root": str(contsg_root),
                "experiment": str(experiment),
                "checkpoint": str(args.checkpoint),
                "indices": indices,
                "n_samples": int(args.n_samples),
                "sampler": str(args.sampler),
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


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


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


def _caption_from_batch(batch: dict, pos: int) -> list[str]:
    for key in ("cap", "caption", "text"):
        if key not in batch:
            continue
        value = batch[key]
        if isinstance(value, (list, tuple)):
            return [str(value[pos])]
        try:
            return [str(value[pos])]
        except Exception:
            return [str(value)]
    return [""]


if __name__ == "__main__":
    main()
