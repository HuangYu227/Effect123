"""Plot one real time-series condition against several generated methods.

The output is sized for an AAAI single column.  All curves must correspond to
the same conditioning example; the script does not align or shift methods.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLE = {
    "Ours": ("#005F73", "-", 1.8),
    "VerbalTS": ("#CA6702", "--", 1.35),
    "TEdit": ("#6A994E", "-.", 1.35),
    "TimeVQVAE": ("#B56576", ":", 1.55),
    "Text2Motion": ("#7B6D8D", (0, (4, 1, 1, 1)), 1.35),
}
REAL_STYLE = ("#2B2B2B", (0, (6, 2)), 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real and multi-method time-series generations in an AAAI single column.")
    parser.add_argument("--real", required=True, type=Path, help="Ground-truth .npy/.npz file.")
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--verbalts", type=Path, default=None)
    parser.add_argument("--tedit", type=Path, default=None)
    parser.add_argument("--timevqvae", type=Path, default=None)
    parser.add_argument("--text2motion", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--real-sample-index", type=int, default=None,
        help="Index in the real-data array; defaults to --sample-index.",
    )
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--aggregate", choices=("first", "mean", "median"), default="median")
    parser.add_argument("--title", default=None, help="Optional short in-figure title. Omit for paper use.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as archive:
            preferred = ("generated", "samples", "final_state", "real", "target", "data")
            for key in preferred:
                if key in archive:
                    return np.asarray(archive[key])
            if len(archive.files) == 1:
                return np.asarray(archive[archive.files[0]])
            raise ValueError(f"Cannot choose an array from {path}; keys={archive.files}")
    return np.asarray(np.load(path))


def select_series(array: np.ndarray, sample_index: int, channel_index: int, aggregate: str, *, generated: bool) -> np.ndarray:
    """Reduce [N,S,L,C], [N,L,C], [L,C], or [L] to one [L] curve."""
    values = np.asarray(array)
    if values.ndim == 4:
        if not 0 <= sample_index < values.shape[0]:
            raise IndexError(f"sample_index {sample_index} is outside {values.shape[0]} samples")
        values = values[sample_index]
        values = values[0] if aggregate == "first" else getattr(np, aggregate)(values, axis=0)
    elif values.ndim == 3:
        if not 0 <= sample_index < values.shape[0]:
            raise IndexError(f"sample_index {sample_index} is outside {values.shape[0]} samples")
        values = values[sample_index]
    if values.ndim == 2:
        if not 0 <= channel_index < values.shape[-1]:
            raise IndexError(f"channel_index {channel_index} is outside {values.shape[-1]} channels")
        values = values[:, channel_index]
    if values.ndim != 1:
        raise ValueError(f"Expected one reducible time series, got shape {values.shape}")
    return values.astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    real_index = args.sample_index if args.real_sample_index is None else args.real_sample_index
    real = select_series(load_array(args.real), real_index, args.channel_index, "first", generated=False)
    method_paths = {
        "Ours": args.ours,
        "VerbalTS": args.verbalts,
        "TEdit": args.tedit,
        "TimeVQVAE": args.timevqvae,
        "Text2Motion": args.text2motion,
    }
    methods: dict[str, np.ndarray] = {}
    for name, path in method_paths.items():
        if path is None:
            continue
        curve = select_series(load_array(path), args.sample_index, args.channel_index, args.aggregate, generated=True)
        if curve.shape != real.shape:
            raise ValueError(f"{name} has length {curve.size}, but real series has length {real.size}; use matched windows.")
        methods[name] = curve

    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(3.25, 2.65), facecolor="white")
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.22, top=0.80)
    x = np.arange(real.size)
    real_color, real_style, real_width = REAL_STYLE
    ax.plot(x, real, color=real_color, linestyle=real_style, linewidth=real_width, label="Ground Truth", zorder=5)
    for name, curve in methods.items():
        color, linestyle, width = METHOD_STYLE[name]
        ax.plot(x, curve, color=color, linestyle=linestyle, linewidth=width, label=name, alpha=0.96)
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Normalized value", fontsize=9)
    if args.title:
        ax.set_title(args.title, fontsize=9.4, pad=4)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    handles, _ = ax.get_legend_handles_labels()
    # Short labels keep a six-entry legend inside the 3.25-inch column.
    # The paper caption expands GT/VTS/TE/VQ/T2M to their full names.
    short_labels = ["GT", "Ours", "VTS", "TE", "VQ", "T2M"]
    ax.legend(
        handles, short_labels, loc="upper center", bbox_to_anchor=(0.5, 1.22),
        ncol=6, fontsize=9, frameon=False, handlelength=1.0,
        columnspacing=0.30, handletextpad=0.25, borderaxespad=0.0,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print(f"[saved] {args.output} size=3.25x2.65in dpi={args.dpi}")
    print(f"[saved] {args.output.with_suffix('.pdf')}")
    print(
        f"[methods] {', '.join(methods)}; generated_sample={args.sample_index}; "
        f"real_sample={real_index}; aggregate={args.aggregate}; channel={args.channel_index}"
    )


if __name__ == "__main__":
    main()
