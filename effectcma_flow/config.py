from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file is empty: {path}")
    return cfg


def resolve_data_root(cfg: dict[str, Any], override: str | None = None) -> str:
    root = override or os.environ.get("WEATHER_ROOT") or cfg.get("data", {}).get("root")
    if not root:
        raise ValueError("Weather data root is required. Pass --data-root or set WEATHER_ROOT.")
    root = os.path.expandvars(os.path.expanduser(str(root)))
    cfg.setdefault("data", {})["root"] = root
    return root
