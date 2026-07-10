"""Plot an operator-wise vector-field decomposition exported during ODE sampling."""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"temporal": "#0072B2", "frequency": "#D55E00", "channel": "#009E73", "fused": "#202020", "state": "#6F6F6F"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a real ODE operator-decomposition trace.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--captions", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.trace) as data:
        state = data["state"]
        candidate = data["candidate_velocity"]
        gate = data["gate"]
        weighted = data["weighted_velocity"]
        fused = data["conditional_fused_velocity"]
        ode_velocity = data["ode_velocity"]
        next_state = data["next_state"]
        flow_time = data["flow_time"]
    if not 0 <= args.sample_index < state.shape[0]:
        raise IndexError(f"--sample-index must be in [0,{state.shape[0] - 1}]")
    if not 0 <= args.channel_index < state.shape[3]:
        raise IndexError(f"--channel-index must be in [0,{state.shape[3] - 1}]")
    captions_path = args.captions or args.trace.with_suffix(".captions.json")
    caption = ""
    if captions_path.exists():
        caption = str(json.loads(captions_path.read_text(encoding="utf-8"))[args.sample_index])

    sample = args.sample_index
    channel = args.channel_index
    snapshots = state.shape[1]
    length = state.shape[2]
    x_axis = np.arange(length)
    names = ("temporal", "frequency", "channel")
    fig, axes = plt.subplots(3, snapshots + 1, figsize=(3.05 * (snapshots + 1), 7.1), gridspec_kw={"width_ratios": [1] * snapshots + [1.12]})
    fig.patch.set_facecolor("white")
    for idx in range(snapshots):
        time_label = f"t = {float(flow_time[idx]):.2f}"
        ax = axes[0, idx]
        ax.plot(x_axis, state[sample, idx, :, channel], color=COLORS["state"], lw=1.8, label=r"$x_t$")
        ax.plot(x_axis, next_state[sample, idx, :, channel], color="#B7B7B7", lw=1.2, ls="--", label=r"$x_{t+\Delta t}$")
        ax.set_title(time_label, fontsize=11, pad=7)
        ax.set_ylabel("State" if idx == 0 else "")
        ax.grid(alpha=0.18, lw=0.6)

        ax = axes[1, idx]
        for operator, name in enumerate(names):
            ax.plot(x_axis, candidate[sample, idx, :, channel, operator], color=COLORS[name], lw=1.45, label=rf"$v_{{{name[:4]}}}$")
        ax.axhline(0.0, color="#A8A8A8", lw=0.7)
        ax.set_ylabel("Candidate velocity" if idx == 0 else "")
        ax.grid(alpha=0.18, lw=0.6)

        ax = axes[2, idx]
        for operator, name in enumerate(names):
            ax.plot(x_axis, weighted[sample, idx, :, channel, operator], color=COLORS[name], lw=1.4, label=rf"$w_{{{name[:4]}}}v_{{{name[:4]}}}$")
        ax.plot(x_axis, fused[sample, idx, :, channel], color=COLORS["fused"], lw=1.9, ls="--", label=r"$\hat v$")
        ax.axhline(0.0, color="#A8A8A8", lw=0.7)
        ax.set_xlabel("Sequence position")
        ax.set_ylabel("Weighted velocity" if idx == 0 else "")
        ax.grid(alpha=0.18, lw=0.6)
        weights = gate[sample, idx]
        ax.text(0.02, 0.97, "w = (" + ", ".join(f"{value:.2f}" for value in weights) + ")", transform=ax.transAxes, va="top", fontsize=8.2, bbox={"facecolor": "white", "edgecolor": "#D0D0D0", "alpha": 0.92, "pad": 2.2})

    final_ax = axes[0, -1]
    final_ax.plot(x_axis, next_state[sample, -1, :, channel], color=COLORS["state"], lw=2.1)
    final_ax.set_title("Generated state", fontsize=11, pad=7)
    final_ax.set_ylabel("State")
    final_ax.grid(alpha=0.18, lw=0.6)
    axes[1, -1].axis("off")
    axes[2, -1].axis("off")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.987, 0.295), ncol=2, fontsize=9, frameon=True)
    if caption:
        fig.suptitle("Operator-wise Vector-Field Decomposition\n" + textwrap.fill(caption, width=105), fontsize=12.5, y=0.992)
    else:
        fig.suptitle("Operator-wise Vector-Field Decomposition Along an ODE Trajectory", fontsize=13, y=0.992)
    fig.tight_layout(rect=(0.015, 0.02, 0.99, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"[saved] {args.output}")
    print(f"[saved] {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
