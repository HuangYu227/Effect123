"""Create a publication-oriented plot of operator contributions along one ODE path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Temporal": "#0072B2", "Frequency": "#D55E00", "Channel": "#009E73"}
STATE_COLORS = ("#C8C8C8", "#7AA6C2", "#005F73")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real operator contribution dynamics from an ODE trace.")
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--captions", type=Path, default=None)
    parser.add_argument(
        "--sample-index", type=int, default=-1,
        help="Trace sample to draw. Use -1 to automatically select the strongest contribution-dynamics example.",
    )
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument(
        "--layout", choices=("wide", "single-column"), default="wide",
        help="Use single-column for a 3.25-inch AAAI figure with vertically stacked panels.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def choose_sample(energy: np.ndarray, gates: np.ndarray) -> int:
    """Prefer real samples with changing *contributions*, not merely near-tied gates."""
    normalized_energy = energy / np.maximum(energy.sum(axis=-1, keepdims=True), 1e-8)
    contribution_range = normalized_energy.max(axis=1) - normalized_energy.min(axis=1)
    gate_range = gates.max(axis=1) - gates.min(axis=1)
    score = contribution_range.mean(axis=-1) + 0.25 * gate_range.mean(axis=-1)
    return int(score.argmax())


def set_panel_style(ax: plt.Axes) -> None:
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8.5)


def plot_single_column(
    flow_time: np.ndarray,
    energy: np.ndarray,
    gates: np.ndarray,
    sample: int,
    output: Path,
    dpi: int,
) -> None:
    """Render a compliant 3.25 x 3.35 inch AAAI single-column figure."""
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    labels = tuple(COLORS)
    styles = ("-", "--", ":")
    fig, (ax_energy, ax_gate) = plt.subplots(2, 1, figsize=(3.25, 3.35), sharex=True, facecolor="white")
    # Fixed margins retain the 9pt labels in the exported physical dimensions.
    fig.subplots_adjust(left=0.225, right=0.985, bottom=0.175, top=0.695, hspace=0.68)
    for operator, (label, style) in enumerate(zip(labels, styles)):
        ax_energy.plot(
            flow_time, energy[sample, :, operator], color=COLORS[label],
            linestyle=style, linewidth=1.45, label=label,
        )
        ax_gate.plot(
            flow_time, gates[sample, :, operator], color=COLORS[label],
            linestyle=style, linewidth=1.45, label=label,
        )
    ax_energy.set_title("(a) Operator contribution magnitude", fontsize=9.4, pad=3)
    ax_energy.set_ylabel("Contribution", fontsize=9)
    ax_gate.set_title("(b) Dynamic mixture weights", fontsize=9.4, pad=3)
    ax_gate.set_xlabel("ODE flow time", fontsize=9)
    ax_gate.set_ylabel("Gate weight", fontsize=9)
    ax_gate.set_ylim(0.0, min(1.0, max(0.55, float(gates[sample].max()) + 0.05)))
    for ax in (ax_energy, ax_gate):
        set_panel_style(ax)
        ax.tick_params(labelsize=9)
        ax.set_xlim(float(flow_time.min()), float(flow_time.max()))
        ax.set_xticks(np.linspace(0.0, 1.0, 5))
    handles, legend_labels = ax_energy.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="upper center", bbox_to_anchor=(0.55, 0.995),
        ncol=2, fontsize=9, frameon=False, handlelength=1.55,
        columnspacing=0.72, handletextpad=0.32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Do not use bbox_inches='tight': physical dimensions must remain stable.
    fig.savefig(output, dpi=dpi, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    with np.load(args.trace) as data:
        state = data["state"]
        weighted = data["weighted_velocity"]
        gates = data["gate"]
        fused = data["conditional_fused_velocity"]
        next_state = data["next_state"]
        flow_time = data["flow_time"]
    if state.shape[1] < 3:
        raise ValueError("This plot requires at least three recorded ODE states; use --record-times all.")
    if not 0 <= args.channel_index < state.shape[3]:
        raise IndexError(f"--channel-index must be in [0,{state.shape[3] - 1}]")

    # [N,S,K]: mean absolute operator contribution at each ODE state.
    energy = np.abs(weighted).mean(axis=(2, 3))
    sample = choose_sample(energy, gates) if args.sample_index < 0 else args.sample_index
    if not 0 <= sample < state.shape[0]:
        raise IndexError(f"--sample-index must be in [0,{state.shape[0] - 1}]")
    captions_path = args.captions or args.trace.with_suffix(".captions.json")
    if captions_path.exists():
        # Keep the caption available for provenance, but do not put a long
        # natural-language paragraph inside a publication figure.
        json.loads(captions_path.read_text(encoding="utf-8"))[sample]

    if args.layout == "single-column":
        plot_single_column(flow_time, energy, gates, sample, args.output, args.dpi)
        print(f"[selected] sample={sample}")
        print(f"[saved] {args.output}")
        print(f"[saved] {args.output.with_suffix('.pdf')}")
        return

    snapshots = np.asarray([0, state.shape[1] // 2, state.shape[1] - 1], dtype=int)
    labels = tuple(COLORS)
    channel = args.channel_index
    x = np.arange(state.shape[2])
    fig = plt.figure(figsize=(10.6, 5.35), constrained_layout=True, facecolor="white")
    outer = fig.add_gridspec(2, 2, width_ratios=(1.28, 1.0), height_ratios=(0.98, 1.02))
    state_grid = outer[0, :].subgridspec(1, 3, wspace=0.14)

    state_handles = []
    for col, index in enumerate(snapshots):
        ax = fig.add_subplot(state_grid[0, col])
        handle = ax.plot(x, state[sample, index, :, channel], color=STATE_COLORS[col], linewidth=1.8)[0]
        state_handles.append(handle)
        ax.plot(x, next_state[sample, index, :, channel], color="#B5B5B5", linewidth=1.0, linestyle="--", alpha=0.9)
        ax.set_title(rf"{chr(97 + col)}) State at $t={flow_time[index]:.2f}$", fontsize=10.2, pad=5)
        ax.set_xlabel("Sequence position", fontsize=8.8)
        if col == 0:
            ax.set_ylabel("State", fontsize=9.5)
        set_panel_style(ax)

    ax_energy = fig.add_subplot(outer[1, 0])
    for operator, label in enumerate(labels):
        ax_energy.plot(flow_time, energy[sample, :, operator], color=COLORS[label], linewidth=2.1, label=label)
    ax_energy.set_title("d) Operator contribution magnitude", fontsize=10.5, pad=5)
    ax_energy.set_xlabel("ODE flow time", fontsize=9.5)
    ax_energy.set_ylabel(r"mean $|w_k v_k|$", fontsize=9.5)
    ax_energy.legend(frameon=False, fontsize=8.6, ncol=3, loc="upper center")
    set_panel_style(ax_energy)

    ax_gate = fig.add_subplot(outer[1, 1])
    for operator, label in enumerate(labels):
        ax_gate.plot(flow_time, gates[sample, :, operator], color=COLORS[label], linewidth=2.1, label=label)
    ax_gate.set_ylim(0.0, min(1.0, max(0.45, float(gates[sample].max()) + 0.06)))
    ax_gate.set_title("e) Dynamic mixture weights", fontsize=10.5, pad=5)
    ax_gate.set_xlabel("ODE flow time", fontsize=9.5)
    ax_gate.set_ylabel(r"$w_k$", fontsize=9.5)
    set_panel_style(ax_gate)

    error = float(np.abs(fused - weighted.sum(axis=-1)).max())
    fig.suptitle("Operator-wise Contribution Dynamics Along an ODE Trajectory", fontsize=13, y=1.015)
    fig.text(
        0.5, -0.015,
        rf"Illustrative conditional trajectory (sample {sample}). "
        rf"Contribution: mean $|w_kv_k|$ over sequence positions; decomposition error: {error:.1e}.",
        ha="center", fontsize=8.5, color="#4B4B4B",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    print(f"[selected] sample={sample}")
    print(f"[saved] {args.output}")
    print(f"[saved] {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
