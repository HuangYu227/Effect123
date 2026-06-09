from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch


def load_training_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    validate_checkpoint_payload(payload)
    return payload


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    required = {"model", "config", "stats", "step", "schema_version"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"Checkpoint payload is missing required keys: {missing}")
    if int(payload["schema_version"]) != 1:
        raise ValueError(f"Unsupported checkpoint schema_version={payload['schema_version']!r}")


def checkpoint_eval_config(runtime_cfg: dict[str, Any], payload: dict[str, Any], *, text_encoder_override: str | None = None) -> dict[str, Any]:
    """Use checkpoint config for model/data semantics while preserving runtime location/device."""
    if "config" not in payload:
        raise ValueError("Checkpoint payload is missing 'config'; cannot reproduce evaluation safely")
    cfg = deepcopy(payload["config"])
    runtime_root = runtime_cfg.get("data", {}).get("root")
    runtime_device = runtime_cfg.get("train", {}).get("device")
    if runtime_root:
        cfg.setdefault("data", {})["root"] = runtime_root
    if runtime_device:
        cfg.setdefault("train", {})["device"] = runtime_device
    if text_encoder_override is not None:
        ckpt_mode = str(cfg.get("text_encoder", {}).get("mode", "hash"))
        if text_encoder_override != ckpt_mode:
            raise ValueError(
                f"Checkpoint was trained with text_encoder.mode={ckpt_mode!r}; "
                f"refusing incompatible override {text_encoder_override!r}"
            )
    return cfg


def checkpoint_stats_or_none(payload: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    stats = payload.get("stats")
    if not stats:
        raise ValueError("Checkpoint payload is missing 'stats'; cannot reproduce normalization safely")
    return {key: value.detach().cpu().float() for key, value in stats.items()}
