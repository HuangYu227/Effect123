from __future__ import annotations

from typing import Any


def scalarize_logs(logs: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key, val in logs.items():
        if hasattr(val, "item"):
            try:
                out[key] = float(val.item())
            except (ValueError, RuntimeError):
                pass
        elif isinstance(val, (int, float)):
            out[key] = float(val)
    return out


def print_core_flow_report(logs: dict[str, Any]) -> None:
    s = scalarize_logs(logs)
    parts = [f"loss={s.get('loss', float('nan')):.4f}"]
    for key in [
        "gate_rescale_factor",
        "gate_raw_cell_mass_mean",
        "gate_cell_mass_mean",
        "operator_gate_entropy",
        "time_gate_entropy",
        "channel_gate_entropy",
    ]:
        if key in s:
            parts.append(f"{key}={s[key]:.4f}")
    print(" | ".join(parts))


def assert_text_encoder_is_semantic(model: object) -> None:
    from effectcma_flow.models.text_encoder import HashTextEncoder
    text_enc = getattr(model, "text_encoder", None)
    if isinstance(text_enc, HashTextEncoder):
        raise RuntimeError(
            "Model is using HashTextEncoder which has no semantic ability. "
            "For real CTTP/J-FTSD evaluation, use mode='hf' or mode='longclip'."
        )


def gate_health_status(logs: dict[str, Any]) -> str:
    s = scalarize_logs(logs)
    raw = s.get("gate_raw_cell_mass_mean", float("nan"))
    fixed = s.get("gate_cell_mass_mean", float("nan"))
    ratio = s.get("pred_target_rms_ratio", float("nan"))
    if raw < 0.05 and fixed < 0.2:
        return f"WARNING: gate severely diluted (raw={raw:.4f}, fixed={fixed:.4f})"
    if ratio < 0.3:
        return f"WARNING: model velocity much smaller than target (ratio={ratio:.3f})"
    return f"OK: gate mass={fixed:.4f}, v-ratio={ratio:.3f}"
