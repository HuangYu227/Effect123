from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    base = cfg.pop("_base_", None)
    if base is None:
        return cfg
    base_path = (config_path.parent / str(base)).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    return _deep_merge(load_config(base_path), cfg)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_data_root(cfg: dict[str, Any], override: str | None = None) -> str:
    root = override or os.environ.get("WEATHER_ROOT") or cfg.get("data", {}).get("root")
    if not root:
        raise ValueError("Weather data root is required. Pass --data-root or set WEATHER_ROOT.")
    root = os.path.expandvars(os.path.expanduser(str(root)))
    cfg.setdefault("data", {})["root"] = root
    return root
