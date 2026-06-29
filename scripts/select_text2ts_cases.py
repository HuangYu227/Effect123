from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select real test cases and captions for qualitative text-to-time-series comparison."
    )
    parser.add_argument("--data-root", required=True, help="VerbalTS/ConTSG-style dataset folder.")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", default=None, help="Comma-separated sample indices. Overrides random sampling.")
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--caption-id", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--channels", default=None, help="Comma-separated channels to plot. Defaults to all.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    ts_path = data_root / f"{args.split}_ts.npy"
    cap_path = data_root / f"{args.split}_text_caps.npy"
    if not ts_path.exists():
        raise FileNotFoundError(ts_path)
    if not cap_path.exists():
        raise FileNotFoundError(cap_path)

    ts = np.load(ts_path, allow_pickle=False)
    captions = np.load(cap_path, allow_pickle=True)
    if ts.ndim != 3:
        raise ValueError(f"{ts_path} must have shape [N,L,C], got {ts.shape}")
    if captions.ndim != 2 or captions.shape[0] != ts.shape[0]:
        raise ValueError(f"{cap_path} must have shape [N,J] with N={ts.shape[0]}, got {captions.shape}")

    indices = _resolve_indices(args.indices, int(args.num_cases), int(args.seed), int(ts.shape[0]))
    caption_id = int(args.caption_id)
    if caption_id < 0 or caption_id >= captions.shape[1]:
        raise ValueError(f"--caption-id out of range [0,{captions.shape[1] - 1}]: {caption_id}")

    selected_ts = ts[indices].astype(np.float32)
    selected_captions = [[str(captions[idx, caption_id])] for idx in indices]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "reference.npy", selected_ts)
    (output_dir / "captions.json").write_text(
        json.dumps(selected_captions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "captions.txt").write_text(
        "\n".join(item[0] for item in selected_captions) + "\n",
        encoding="utf-8",
    )
    (output_dir / "indices.json").write_text(
        json.dumps(
            {
                "data_root": str(data_root.resolve()),
                "split": args.split,
                "indices": indices,
                "caption_id": caption_id,
                "reference_shape": list(selected_ts.shape),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if args.plot:
        _plot_reference(output_dir / "reference_plots", selected_ts, selected_captions, _parse_channels(args.channels, ts.shape[-1]))

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "indices": indices,
                "reference": str((output_dir / "reference.npy").resolve()),
                "captions_json": str((output_dir / "captions.json").resolve()),
                "captions_txt": str((output_dir / "captions.txt").resolve()),
                "shape": list(selected_ts.shape),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _resolve_indices(value: str | None, num_cases: int, seed: int, total: int) -> list[int]:
    if value:
        indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        if num_cases <= 0:
            raise ValueError("--num-cases must be positive")
        rng = np.random.default_rng(seed)
        indices = rng.choice(total, size=min(num_cases, total), replace=False).tolist()
        indices = [int(idx) for idx in indices]
    bad = [idx for idx in indices if idx < 0 or idx >= total]
    if bad:
        raise ValueError(f"indices out of range [0,{total - 1}]: {bad}")
    return indices


def _parse_channels(value: str | None, num_channels: int) -> list[int]:
    if value is None:
        return list(range(num_channels))
    channels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not channels:
        raise ValueError("--channels did not contain any channel indices")
    bad = [idx for idx in channels if idx < 0 or idx >= num_channels]
    if bad:
        raise ValueError(f"Channel indices out of range for C={num_channels}: {bad}")
    return channels


def _plot_reference(output_dir: Path, values: np.ndarray, captions: list[list[str]], channels: list[int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(values.shape[0]):
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=180)
        for channel in channels:
            ax.plot(values[idx, :, channel], linewidth=1.4, label=f"ch{channel}")
        ax.set_title(captions[idx][0][:150], fontsize=9)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.22)
        if len(channels) <= 8:
            ax.legend(loc="best", fontsize=7, ncol=min(len(channels), 4))
        fig.tight_layout()
        fig.savefig(output_dir / f"case_{idx:03d}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
