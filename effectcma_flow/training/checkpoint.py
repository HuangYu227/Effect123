from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA_VERSION = 2
TASK_MODES = {"text2ts", "edit"}


def load_training_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {path}. The file may be incomplete/corrupt; "
            "use best.pt or a step_*.pt snapshot if latest.pt was interrupted during writing."
        ) from exc
    validate_checkpoint_payload(payload)
    return payload


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    required = {"model", "config", "stats", "step", "schema_version", "task_mode", "model_class"}
    missing = sorted(required.difference(payload.keys()))
    if missing:
        raise ValueError(f"Checkpoint payload is missing required keys: {missing}")
    if int(payload["schema_version"]) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version={payload['schema_version']!r}")
    config_task_mode = _task_mode_from_config(payload["config"])
    if payload["task_mode"] != config_task_mode:
        raise ValueError(f"Checkpoint task_mode={payload['task_mode']!r} does not match config task.mode={config_task_mode!r}")
    expected_classes = _expected_model_classes(payload["config"], config_task_mode)
    if payload["model_class"] not in expected_classes:
        raise ValueError(
            f"Checkpoint model_class={payload['model_class']!r} does not match expected classes "
            f"{sorted(expected_classes)} for task.mode={config_task_mode!r}"
        )


def checkpoint_eval_config(
    runtime_cfg: dict[str, Any],
    payload: dict[str, Any],
    *,
    text_encoder_override: str | None = None,
    task_mode_override: str | None = None,
) -> dict[str, Any]:
    """Use checkpoint config for model/data semantics while preserving runtime location/device."""
    if "config" not in payload:
        raise ValueError("Checkpoint payload is missing 'config'; cannot reproduce evaluation safely")
    cfg = deepcopy(payload["config"])
    runtime_root = runtime_cfg.get("data", {}).get("root")
    if runtime_root:
        cfg.setdefault("data", {})["root"] = runtime_root
    _copy_runtime_train_overrides(cfg, runtime_cfg)
    _copy_runtime_text_model_override(cfg, runtime_cfg)
    if task_mode_override is not None:
        ckpt_task_mode = _task_mode_from_config(cfg)
        if task_mode_override != ckpt_task_mode:
            raise ValueError(
                f"Checkpoint was trained with task.mode={ckpt_task_mode!r}; "
                f"refusing incompatible override {task_mode_override!r}"
            )
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


def _task_mode_from_config(cfg: dict[str, Any]) -> str:
    task_mode = str(cfg.get("task", {}).get("mode", "text2ts")).lower()
    if task_mode not in TASK_MODES:
        raise ValueError(f"Checkpoint config must contain task.mode in {sorted(TASK_MODES)}, got {task_mode!r}")
    return task_mode


def _expected_model_classes(cfg: dict[str, Any], task_mode: str) -> set[str]:
    if task_mode == "edit":
        return {"EffectCMAFlow"}
    model_cfg = cfg.get("model", {})
    model_type = str(model_cfg.get("model_type", model_cfg.get("architecture_type", "flow"))).lower()
    if model_type in {"blueprint", "blueprint_flow", "v4"}:
        return {"BlueprintTextToTSFlow"}
    return {"TextToTSFlow"}


def _copy_runtime_train_overrides(cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> None:
    runtime_train = runtime_cfg.get("train", {})
    train = cfg.setdefault("train", {})
    for key in ("device", "batch_size", "eval_batch_size", "num_workers"):
        value = runtime_train.get(key)
        if value is not None:
            train[key] = value


def _copy_runtime_text_model_override(cfg: dict[str, Any], runtime_cfg: dict[str, Any]) -> None:
    runtime_text = runtime_cfg.get("text_encoder", {})
    text = cfg.setdefault("text_encoder", {})
    mode = str(text.get("mode", "hash")).lower()
    for key in ("hf_model_name", "longclip_model_name"):
        runtime_value = runtime_text.get(key)
        checkpoint_value = text.get(key)
        if runtime_value is not None and checkpoint_value is not None and str(runtime_value) != str(checkpoint_value):
            raise ValueError(
                f"Checkpoint was trained with text_encoder.{key}={checkpoint_value!r}; "
                f"refusing runtime override {runtime_value!r}. Use a checkpoint/config conversion for model path migration."
            )
    for key in ("local_files_only", "longclip_local_files_only"):
        if key in runtime_text:
            if mode == "hash":
                continue
            text[key] = runtime_text[key]
