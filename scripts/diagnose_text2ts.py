"""One-shot health probe for a text2ts EffectCMA-Flow configuration.

This answers the audit questions that the loss curve alone cannot:

* Which routing / connector path is actually live (mapper vs global gate,
  legacy vs patch-merger)?
* Does the Perceiver token-budget pool actually fire, or is ``token_budget``
  so large it is an identity pass-through?
* Does the operator gate actually read the text tokens (attention text mass)?
* Is the cross-modal alignment dense (token/region-level) or pooled?
* **Is the text condition actually used?** We compare the predicted velocity for
  the true captions against (a) batch-shuffled captions and (b) blank captions.
  If the prediction barely changes, the model ignores text regardless of what
  the loss says.

It runs on a tiny number of samples and never trains, so it is safe to run
against a real data root or the bundled fake weather data.

Usage::

    python scripts/diagnose_text2ts.py --config configs/weather_v65_patch_merger_rank001.yaml
    python scripts/diagnose_text2ts.py --config <cfg> --checkpoint checkpoints/.../best.pt
    python scripts/diagnose_text2ts.py --config <cfg> --fake-data   # no data root needed
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from effectcma_flow.config import load_config, resolve_data_root
from effectcma_flow.data import WeatherRawCaptionDataset, collate_raw_caption_batch, compute_train_stats
from effectcma_flow.models import build_model
from effectcma_flow.training import resolve_device
from effectcma_flow.training.utils import batch_to_device, text_condition_from_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint; otherwise random init")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--fake-data", action="store_true", help="Use bundled fake weather data (no real root needed)")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    task_mode = str(cfg.get("task", {}).get("mode", "text2ts")).lower()
    if task_mode != "text2ts":
        raise SystemExit(f"diagnose_text2ts only supports task.mode='text2ts', got {task_mode!r}")

    device = resolve_device(str(cfg["train"].get("device", "auto")))
    text_mode = str(cfg["text_encoder"].get("mode", "hash"))

    if args.fake_data:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
        from fake_weather_data import create_fake_weather_root

        tmp_root = Path(tempfile.mkdtemp(prefix="diag_fake_weather_"))
        create_fake_weather_root(tmp_root)
        cfg.setdefault("data", {})["root"] = str(tmp_root)
        # Fake data only ships the hash-compatible captions; force offline encoder.
        text_mode = "hash"
        cfg["text_encoder"]["mode"] = "hash"
    else:
        resolve_data_root(cfg, args.data_root)

    stats = compute_train_stats(cfg["data"]["root"])
    ds = WeatherRawCaptionDataset(
        cfg["data"]["root"],
        args.split,
        window_length=cfg["data"].get("window_length"),
        normalize=bool(cfg["data"].get("normalize", True)),
        stats=stats,
        seed=int(cfg["data"].get("seed", 0)),
        caption_policy="cyclic",
    )
    n = min(int(args.num_samples), len(ds))
    batch = collate_raw_caption_batch([ds[i] for i in range(n)])
    batch = batch_to_device(batch, device)

    model = build_model(cfg, sequence_length=ds.sequence_length, num_channels=ds.num_channels).to(device)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
    model.eval()

    report = _probe(model, batch, cfg, text_mode, device)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    _print_verdicts(report)


@torch.no_grad()
def _probe(model, batch, cfg, text_mode, device) -> dict:
    target = batch["Y"].float()
    b, length, channels = target.shape
    t = torch.full((b,), 0.5, device=device, dtype=target.dtype)
    source = torch.randn_like(target)
    x_t = (1.0 - t[:, None, None]) * source + t[:, None, None] * target

    def cond(batch_in):
        return text_condition_from_batch(
            batch_in,
            text_mode,
            condition_key="caption",
            caption_slot_strategy=str(cfg.get("train", {}).get("caption_slot_strategy", "single")),
            max_caption_slots=int(cfg.get("train", {}).get("max_caption_slots", 1)),
            include_all_caption_candidates=bool(cfg.get("train", {}).get("include_all_caption_candidates", False)),
        )

    def run(batch_in):
        try:
            return model(x_t, t, cond(batch_in), target=target)
        except TypeError:
            return model(x_t, t, cond(batch_in))

    pred_true, aux = run(batch)

    # Text-sensitivity probes: same x_t / t / source, only the caption changes.
    shuffled = dict(batch)
    perm = torch.roll(torch.arange(b), 1).tolist()
    for key in ("caption", "caption_candidates", "captions"):
        if batch.get(key) is not None:
            shuffled[key] = [batch[key][i] for i in perm]
    pred_shuf, _ = run(shuffled)

    blanked = dict(batch)
    blanked["caption"] = [""] * b
    if batch.get("caption_candidates") is not None:
        blanked["caption_candidates"] = [[] for _ in range(b)]
    pred_blank, _ = run(blanked)

    pred_rms = pred_true.square().mean().sqrt().clamp_min(1e-8)
    shuffle_delta = (pred_true - pred_shuf).square().mean().sqrt() / pred_rms
    blank_delta = (pred_true - pred_blank).square().mean().sqrt() / pred_rms

    def scal(key):
        v = aux.get(key)
        return float(v.detach().cpu()) if torch.is_tensor(v) and v.numel() == 1 else None

    bridge = getattr(model, "cross_modal_bridge", None)
    connector = getattr(bridge, "patch_merger_connector", None) if bridge is not None else None

    return {
        "shapes": {"batch": b, "length": length, "channels": channels},
        "active_path": {
            "router_mode": getattr(model, "router_mode", None),
            "uses_global_gate": getattr(model, "_use_global_gate", None),
            "uses_effect_mapper": getattr(model, "mapper", None) is not None,
            "uses_cross_modal_bridge": bridge is not None,
            "state_connector": getattr(bridge, "state_connector", None) if bridge is not None else None,
            "alignment_dense": getattr(connector, "alignment_dense", None) if connector is not None else None,
            "alignment_target": getattr(connector, "alignment_target", None) if connector is not None else None,
            "operator_architecture": getattr(model.operator_bank, "architecture", None),
        },
        "budget_pool": {
            "patch_token_count": scal("bridge_patch_token_count"),
            "token_budget": scal("bridge_token_budget"),
            "active": scal("bridge_budget_pool_active"),
        },
        "text_routing": {
            "operator_gate_attention_text_mass": scal("operator_gate_attention_text_mass"),
            "operator_gate_attention_state_mass": scal("operator_gate_attention_state_mass"),
            "operator_gate_entropy_norm": scal("operator_gate_entropy_norm"),
            "operator_gate_max_prob": scal("operator_gate_max_prob"),
        },
        "alignment": {
            "bridge_alignment_loss": scal("bridge_alignment_loss"),
            "bridge_alignment_dense": scal("bridge_alignment_dense"),
            "bridge_alignment_logit_pos": scal("bridge_alignment_logit_pos"),
        },
        "text_sensitivity": {
            "pred_rms": float(pred_rms.detach().cpu()),
            "shuffle_caption_rel_delta": float(shuffle_delta.detach().cpu()),
            "blank_caption_rel_delta": float(blank_delta.detach().cpu()),
        },
    }


def _print_verdicts(report: dict) -> None:
    print("\n=== verdicts ===")
    ap = report["active_path"]
    if ap.get("uses_global_gate") and not ap.get("uses_effect_mapper"):
        print("[path] global operator gate is LIVE; EffectMapper is bypassed (mapper_* keys are inert).")
    bp = report["budget_pool"]
    if bp.get("active") == 0.0:
        print(
            f"[budget] WARNING: token-budget pool is INACTIVE "
            f"(patch_tokens={bp.get('patch_token_count')} <= budget={bp.get('token_budget')}). "
            f"It is an identity pass-through; lower bridge_token_budget to make it fire."
        )
    elif bp.get("active") == 1.0:
        print(f"[budget] token-budget pool is active ({bp.get('patch_token_count')} > {bp.get('token_budget')}).")
    tm = report["text_routing"].get("operator_gate_attention_text_mass")
    if tm is not None:
        verdict = "OK" if tm > 0.15 else "WARNING: router barely attends to text"
        print(f"[router] attention text mass={tm:.3f} -> {verdict}")
    al = report["alignment"].get("bridge_alignment_dense")
    if al is not None:
        print(f"[align] dense (token/region-level) alignment is {'ON' if al == 1.0 else 'OFF (pooled)'}.")
    ts = report["text_sensitivity"]
    sd, bd = ts["shuffle_caption_rel_delta"], ts["blank_caption_rel_delta"]
    print(f"[text] shuffle Δ={sd:.3f}, blank Δ={bd:.3f} (relative to pred RMS)")
    if max(sd, bd) < 0.02:
        print("[text] WARNING: prediction barely changes with the caption -> text is effectively IGNORED.")
    else:
        print("[text] prediction responds to the caption -> text condition is used.")


if __name__ == "__main__":
    main()
