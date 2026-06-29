from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot side-by-side time-series samples from multiple text-to-TS generators. "
            "Inputs are .npy arrays with shape [N,S,L,C], [S,L,C], or [L,C]."
        )
    )
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        help="Method series in LABEL=PATH format. Repeat for Our/VerbalTS/ConTSG.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Optional reference .npy array. Supports [N,L,C], [L,C], or [N,S,L,C].",
    )
    parser.add_argument("--caption-json", default=None, help="Optional captions.json from generate_text2ts.py.")
    parser.add_argument("--caption-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel indices to plot. Defaults to all channels.",
    )
    parser.add_argument("--output", required=True, help="Output PNG/PDF/SVG path.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--max-title-chars", type=int, default=150)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--fig-width", type=float, default=12.0)
    parser.add_argument("--row-height", type=float, default=2.25)
    parser.add_argument("--share-y", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    methods = [_parse_series_arg(item) for item in args.series]
    if not methods:
        raise ValueError("At least one --series LABEL=PATH is required")

    samples = []
    for label, path in methods:
        arr = np.load(path, allow_pickle=False)
        sample = _select_sample(arr, caption_index=args.caption_index, sample_index=args.sample_index)
        samples.append((label, sample))

    reference = None
    if args.reference is not None:
        ref_arr = np.load(args.reference, allow_pickle=False)
        reference = _select_reference(ref_arr, caption_index=args.caption_index, sample_index=args.sample_index)

    channels = _parse_channels(args.channels, samples[0][1].shape[-1])
    _validate_shapes(samples, reference)

    title = args.title
    if title is None:
        title = _caption_title(args.caption_json, args.caption_index)
    if title:
        title = title[: int(args.max_title_chars)]

    _plot(
        samples=samples,
        reference=reference,
        channels=channels,
        output=Path(args.output),
        title=title,
        dpi=int(args.dpi),
        fig_width=float(args.fig_width),
        row_height=float(args.row_height),
        share_y=bool(args.share_y),
        grid=not bool(args.no_grid),
    )
    print(json.dumps({"output": str(Path(args.output).resolve()), "methods": [m[0] for m in methods]}, indent=2))


def _parse_series_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--series must use LABEL=PATH format")
    label, path = value.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise ValueError("--series label cannot be empty")
    if not path.exists():
        raise FileNotFoundError(path)
    return label, path


def _select_sample(arr: np.ndarray, *, caption_index: int, sample_index: int) -> np.ndarray:
    if arr.ndim == 4:
        return arr[int(caption_index), int(sample_index)]
    if arr.ndim == 3:
        return arr[int(sample_index)]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Series array must be [N,S,L,C], [S,L,C], or [L,C], got {arr.shape}")


def _select_reference(arr: np.ndarray, *, caption_index: int, sample_index: int) -> np.ndarray:
    if arr.ndim == 4:
        return arr[int(caption_index), int(sample_index)]
    if arr.ndim == 3:
        return arr[int(caption_index)]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Reference array must be [N,S,L,C], [N,L,C], or [L,C], got {arr.shape}")


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


def _validate_shapes(samples: list[tuple[str, np.ndarray]], reference: np.ndarray | None) -> None:
    base_shape = samples[0][1].shape
    if len(base_shape) != 2:
        raise ValueError(f"Selected samples must be [L,C], got {base_shape}")
    for label, sample in samples:
        if sample.shape != base_shape:
            raise ValueError(f"{label} sample shape {sample.shape} does not match {base_shape}")
    if reference is not None and reference.shape != base_shape:
        raise ValueError(f"Reference shape {reference.shape} does not match generated shape {base_shape}")


def _caption_title(path: str | None, caption_index: int) -> str | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    caption = data[int(caption_index)]
    if isinstance(caption, list):
        return " | ".join(str(x) for x in caption)
    return str(caption)


def _plot(
    *,
    samples: list[tuple[str, np.ndarray]],
    reference: np.ndarray | None,
    channels: list[int],
    output: Path,
    title: str | None,
    dpi: int,
    fig_width: float,
    row_height: float,
    share_y: bool,
    grid: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = len(samples) + (1 if reference is not None else 0)
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(fig_width, max(row_height, row_height * rows)),
        dpi=dpi,
        sharex=True,
        sharey=share_y,
    )
    if rows == 1:
        axes = [axes]

    row = 0
    if reference is not None:
        _plot_one(axes[row], reference, channels, "Reference", grid=grid, linestyle="--")
        row += 1
    for label, sample in samples:
        _plot_one(axes[row], sample, channels, label, grid=grid, linestyle="-")
        row += 1

    axes[-1].set_xlabel("Time Step")
    if title:
        fig.suptitle(title, fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _plot_one(ax, sample: np.ndarray, channels: list[int], label: str, *, grid: bool, linestyle: str) -> None:
    for channel in channels:
        ax.plot(sample[:, channel], linewidth=1.4, linestyle=linestyle, label=f"ch{channel}")
    ax.set_ylabel(label)
    if grid:
        ax.grid(alpha=0.22, linewidth=0.7)
    if len(channels) <= 8:
        ax.legend(loc="upper right", fontsize=7, ncol=min(len(channels), 4))


if __name__ == "__main__":
    main()
