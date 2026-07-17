#!/usr/bin/env python3
"""Visualize counterfactual caption edits with a fixed Synth-M flow trajectory.

The script selects one eligible Synth-M example deterministically, keeps its
initial Gaussian noise fixed, and changes exactly one template field per
generation: trend direction or seasonal cycle.  The resulting figure is a
qualitative controllability visualization, not a fidelity comparison: edited
captions do not have paired ground-truth series.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True)
class CaptionEdit:
    """One caption and the single template field changed from the base caption."""

    key: str
    panel: str
    caption: str
    changed_field: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fixed-noise Synth-M counterfactual caption-edit figure. "
            "Each edited caption differs from the base caption in exactly one field."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Full STEER Synth-M training config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full STEER Synth-M checkpoint.")
    parser.add_argument("--data-root", type=Path, required=True, help="Synth-M dataset root.")
    parser.add_argument(
        "--text-encoder-model",
        type=Path,
        default=None,
        help="Local LongCLIP directory. Overrides the path recorded in the config.",
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Use this eligible split index. By default the first seeded eligible index is used.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=20260717,
        help="Seed for the deterministic eligible-caption selection rule.",
    )
    parser.add_argument(
        "--noise-seed",
        type=int,
        default=20260717,
        help="Seed for the single shared initial Gaussian noise tensor.",
    )
    parser.add_argument("--channel-index", type=int, default=0, help="Synth-M variable to render.")
    parser.add_argument("--solver", choices=("euler", "midpoint", "rk4"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--guidance-t-lo", type=float, default=None)
    parser.add_argument("--guidance-t-hi", type=float, default=None)
    parser.add_argument("--device", default="auto", help="PyTorch device, normally 'auto'.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--figure-stem", default="synthm_counterfactual_caption_edits")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only select and record the base/edit captions; do not load the generator or sample.",
    )
    return parser.parse_args()


def replace_in_one_sentence(
    caption: str,
    *,
    sentence_selector,
    replacement,
) -> str:
    """Apply one controlled replacement and reject ambiguous template captions."""
    pieces = re.split(r"(?<=[.!?])|(?=\n)|(?<=\n)", str(caption))
    changed = 0
    output: list[str] = []
    for piece in pieces:
        if sentence_selector(piece):
            new_piece, did_change = replacement(piece)
            if did_change:
                changed += 1
                piece = new_piece
        output.append(piece)
    if changed != 1:
        raise ValueError(f"expected exactly one controlled template edit, observed {changed}")
    result = "".join(output)
    if result == caption:
        raise ValueError("caption edit produced no change")
    return result


def swap_trend_direction(caption: str) -> str:
    def selector(sentence: str) -> bool:
        lower = sentence.lower()
        return "trend" in lower and re.search(r"\b(up|down)\b", lower) is not None

    def replacement(sentence: str) -> tuple[str, bool]:
        match = re.search(r"\b(up|down)\b", sentence, flags=re.IGNORECASE)
        if match is None:
            return sentence, False
        old = match.group(0)
        new = "down" if old.lower() == "up" else "up"
        if old[:1].isupper():
            new = new.capitalize()
        return sentence[: match.start()] + new + sentence[match.end() :], True

    return replace_in_one_sentence(caption, sentence_selector=selector, replacement=replacement)


def swap_season_cycle(caption: str) -> str:
    """Use the dataset's discrete season-cycle vocabulary: 0<->4 and 1<->2."""
    cycle_swap = {"0": "4", "1": "2", "2": "1", "4": "0"}

    def selector(sentence: str) -> bool:
        return "season" in sentence.lower() and re.search(r"\b[0124]\b", sentence) is not None

    def replacement(sentence: str) -> tuple[str, bool]:
        match = re.search(r"\b([0124])\b", sentence)
        if match is None:
            return sentence, False
        return sentence[: match.start()] + cycle_swap[match.group(1)] + sentence[match.end() :], True

    return replace_in_one_sentence(caption, sentence_selector=selector, replacement=replacement)


def extract_semantic_fields(caption: str) -> dict[str, str]:
    """Return short labels for the three panels without rewriting the full caption."""
    season = re.search(r"season(?:\s+cycle)?\s+is\s+([0124])\b", caption, flags=re.IGNORECASE)
    trend_sentence = next((part for part in re.split(r"(?<=[.!?])", caption) if "trend" in part.lower()), "")
    trend = re.search(r"\b(up|down)\b", trend_sentence, flags=re.IGNORECASE)
    return {
        "trend": trend.group(1).lower() if trend else "edited",
        "cycle": season.group(1) if season else "edited",
    }


def build_edits(base_caption: str) -> list[CaptionEdit]:
    trend_caption = swap_trend_direction(base_caption)
    periodicity_caption = swap_season_cycle(base_caption)
    values = [base_caption, trend_caption, periodicity_caption]
    if len(set(values)) != len(values):
        raise ValueError("base and counterfactual captions must be distinct")
    return [
        CaptionEdit("base", "(a) Base caption", base_caption, "none"),
        CaptionEdit("trend", "(b) Trend edit", trend_caption, "trend direction"),
        CaptionEdit("periodicity", "(c) Periodicity edit", periodicity_caption, "season cycle"),
    ]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_caption(dataset: Any, *, sample_index: int | None, selection_seed: int) -> tuple[int, list[CaptionEdit]]:
    eligible: list[int] = []
    cache: dict[int, list[CaptionEdit]] = {}
    for index in range(len(dataset)):
        caption = str(dataset.captions[index, 0])
        try:
            cache[index] = build_edits(caption)
        except ValueError:
            continue
        eligible.append(index)
    if not eligible:
        raise ValueError("No captions in this split support the controlled trend and season-cycle edits.")

    if sample_index is not None:
        if sample_index not in cache:
            raise ValueError(
                f"split index {sample_index} is not eligible for both controlled edits; "
                f"there are {len(eligible)} eligible examples."
            )
        return int(sample_index), cache[int(sample_index)]

    # This makes selection reproducible without selecting a visually favourable output.
    rng = np.random.default_rng(int(selection_seed))
    index = int(rng.permutation(np.asarray(eligible, dtype=np.int64))[0])
    return index, cache[index]


def resolve_sample_settings(args: argparse.Namespace, sample_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "solver": str(args.solver or sample_cfg.get("solver", "rk4")).lower(),
        "steps": int(args.steps if args.steps is not None else sample_cfg.get("steps", 64)),
        "noise_scale": float(args.noise_scale if args.noise_scale is not None else sample_cfg.get("noise_scale", 1.0)),
        "cfg_scale": float(args.cfg_scale if args.cfg_scale is not None else sample_cfg.get("cfg_scale", 1.0)),
        "guidance_t_lo": float(
            args.guidance_t_lo if args.guidance_t_lo is not None else sample_cfg.get("guidance_t_lo", 0.0)
        ),
        "guidance_t_hi": float(
            args.guidance_t_hi if args.guidance_t_hi is not None else sample_cfg.get("guidance_t_hi", 1.0)
        ),
    }


def generate_counterfactuals(
    *,
    model: torch.nn.Module,
    edits: list[CaptionEdit],
    initial_noise: torch.Tensor,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    from effectcma_flow.evaluation.sampler import sample_text2ts

    output: dict[str, np.ndarray] = {}
    shape_like = torch.empty_like(initial_noise)
    for edit in edits:
        # `noise=` is explicit: every caption starts from precisely the same x_0.
        condition = model.prepare_condition(
            [[edit.caption]], device=initial_noise.device, dtype=initial_noise.dtype
        )
        generated, _ = sample_text2ts(
            model,
            shape_like,
            condition,
            noise=initial_noise,
            **settings,
        )
        output[edit.key] = generated.detach().cpu().float().numpy()[0]
    return output


def plot_counterfactuals(
    *,
    series: dict[str, np.ndarray],
    edits: list[CaptionEdit],
    channel_index: int,
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> None:
    if channel_index < 0 or channel_index >= next(iter(series.values())).shape[-1]:
        channels = next(iter(series.values())).shape[-1]
        raise IndexError(f"channel-index {channel_index} is invalid for {channels} channels")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    values = [series[edit.key][:, channel_index] for edit in edits]
    minimum = float(min(np.min(value) for value in values))
    maximum = float(max(np.max(value) for value in values))
    padding = max(0.04 * (maximum - minimum), 1e-5)

    # AAAI two-column wide figure: 6.8 in wide, compact enough for one visual row.
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 1.95), sharex=True, sharey=True)
    accents = {"base": "#5B6770", "trend": "#B44D3A", "periodicity": "#B7791F"}
    model_color = "#0077BB"
    for axis, edit, value in zip(axes, edits, values):
        fields = extract_semantic_fields(edit.caption)
        axis.plot(np.arange(value.shape[0]), value, color=model_color, linewidth=1.45, solid_capstyle="round")
        axis.set_title(edit.panel, color="black", pad=6.0, fontweight="bold")
        if edit.key == "base":
            label = f"trend = {fields['trend']}; cycle = {fields['cycle']}"
        elif edit.key == "trend":
            label = f"trend = {fields['trend']}; cycle = {fields['cycle']}"
        else:
            label = f"trend = {fields['trend']}; cycle = {fields['cycle']}"
        axis.text(
            0.5,
            1.015,
            label,
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.0,
            color=accents[edit.key],
        )
        axis.set_xlim(0, value.shape[0] - 1)
        axis.set_ylim(minimum - padding, maximum + padding)
        axis.grid(axis="y", color="#D9DEE2", linewidth=0.45, alpha=0.85)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(length=2.4, width=0.65, pad=1.6)
        axis.set_xlabel("Time step", labelpad=1.0)

    axes[0].set_ylabel("Generated value", labelpad=1.5)
    fig.text(
        0.5,
        0.992,
        "Fixed initial noise; one caption field changed per panel",
        ha="center",
        va="top",
        fontsize=7.3,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.20, top=0.78, wspace=0.16)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_png, dpi=int(dpi), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    for path in (args.config, args.checkpoint, args.data_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.text_encoder_model is not None and not args.text_encoder_model.exists():
        raise FileNotFoundError(args.text_encoder_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_reproducible_seed(int(args.noise_seed))

    from effectcma_flow.config import load_config, resolve_data_root
    from effectcma_flow.data import WeatherRawCaptionDataset
    from effectcma_flow.training.checkpoint import checkpoint_eval_config, checkpoint_stats_or_none, load_training_checkpoint
    from effectcma_flow.training.utils import resolve_device

    runtime_cfg = load_config(args.config)
    runtime_cfg.setdefault("task", {})["mode"] = "text2ts"
    runtime_cfg.setdefault("train", {})["device"] = str(args.device)
    resolve_data_root(runtime_cfg, str(args.data_root))
    if args.text_encoder_model is not None:
        text_cfg = runtime_cfg.setdefault("text_encoder", {})
        mode = str(text_cfg.get("mode", "longclip")).lower()
        text_cfg["longclip_model_name" if mode == "longclip" else "hf_model_name"] = str(args.text_encoder_model)

    device = resolve_device(str(args.device))
    payload = load_training_checkpoint(args.checkpoint, device)
    cfg = checkpoint_eval_config(runtime_cfg, payload, task_mode_override="text2ts")
    stats = checkpoint_stats_or_none(payload)
    text_mode = str(cfg.get("text_encoder", {}).get("mode", "hash")).lower()
    if text_mode == "precomputed":
        raise ValueError(
            "This qualitative edit requires token-level text encoding. The selected checkpoint uses precomputed embeddings."
        )

    dataset = WeatherRawCaptionDataset(
        cfg["data"]["root"],
        args.split,
        window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)),
        stats=stats,
        seed=int(cfg["data"].get("seed", 0)) + 40_000,
        caption_policy="first",
        include_precomputed_embeddings=False,
    )
    index, edits = select_caption(dataset, sample_index=args.sample_index, selection_seed=args.selection_seed)
    reference = np.asarray(dataset[index]["Y"], dtype=np.float32)
    if reference.ndim != 2:
        raise ValueError(f"Selected sequence must be [L,C], got {reference.shape}")
    if args.channel_index < 0 or args.channel_index >= reference.shape[-1]:
        raise IndexError(f"channel-index {args.channel_index} invalid for selected shape {reference.shape}")

    selection_payload = {
        "experiment": "synthm_counterfactual_caption_editing",
        "interpretation": (
            "Qualitative directional controllability only. Edited captions have no paired real target; "
            "the three samples share the same explicit initial Gaussian noise."
        ),
        "selection_rule": "first eligible caption under a seeded permutation",
        "split": args.split,
        "selected_split_index": int(index),
        "selection_seed": int(args.selection_seed),
        "noise_seed": int(args.noise_seed),
        "channel_index": int(args.channel_index),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(payload.get("step", -1)),
        "text_mode": text_mode,
        "captions": [
            {"key": edit.key, "panel": edit.panel, "changed_field": edit.changed_field, "caption": edit.caption}
            for edit in edits
        ],
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(selection_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.save(args.output_dir / "reference_normalized_series.npy", reference)
    print(f"[selected] {args.split}[{index}]", flush=True)
    for edit in edits:
        print(f"[{edit.key}] {edit.caption}", flush=True)
    print(f"[saved] metadata: {args.output_dir / 'metadata.json'}", flush=True)

    if args.dry_run:
        print("[dry-run] no model generation requested", flush=True)
        return

    from effectcma_flow.models import build_model

    model = build_model(cfg, sequence_length=dataset.sequence_length, num_channels=dataset.num_channels).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    settings = resolve_sample_settings(args, cfg.get("sample", {}))
    if settings["steps"] <= 0:
        raise ValueError("--steps must be positive")

    dtype = next(model.parameters()).dtype
    noise_generator = torch.Generator(device=device).manual_seed(int(args.noise_seed))
    initial_noise = torch.randn(
        (1, dataset.sequence_length, dataset.num_channels), device=device, dtype=dtype, generator=noise_generator
    )
    generated_normalized = generate_counterfactuals(
        model=model, edits=edits, initial_noise=initial_noise, settings=settings
    )
    if bool(cfg.get("data", {}).get("normalize", True)):
        mean = stats["mean"].detach().cpu().float().numpy().reshape(1, -1)
        std = stats["std"].detach().cpu().float().numpy().reshape(1, -1)
        generated = {key: value * std + mean for key, value in generated_normalized.items()}
    else:
        generated = generated_normalized

    for key, value in generated.items():
        np.save(args.output_dir / f"{key}.npy", value.astype(np.float32, copy=False))
    np.save(args.output_dir / "initial_noise.npy", initial_noise.detach().cpu().float().numpy())
    selection_payload["sampling"] = settings
    (args.output_dir / "metadata.json").write_text(
        json.dumps(selection_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    output_pdf = args.output_dir / f"{args.figure_stem}.pdf"
    output_png = args.output_dir / f"{args.figure_stem}.png"
    plot_counterfactuals(
        series=generated,
        edits=edits,
        channel_index=int(args.channel_index),
        output_pdf=output_pdf,
        output_png=output_png,
        dpi=int(args.dpi),
    )
    print(f"[saved] fixed initial noise: {args.output_dir / 'initial_noise.npy'}", flush=True)
    print(f"[saved] figure: {output_pdf}", flush=True)
    print(f"[saved] figure: {output_png}", flush=True)


if __name__ == "__main__":
    main()
