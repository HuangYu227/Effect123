from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def text_condition_from_batch(batch: dict[str, Any], mode: str):
    if mode == "precomputed":
        if "slot_embeddings" not in batch:
            raise ValueError(
                "precomputed text encoder mode requires batch['slot_embeddings'] derived from the synthetic effect slots; "
                "raw Weather caption embeddings are intentionally not used for semi-synthetic effect control"
            )
        embeddings = batch["slot_embeddings"]
        if embeddings.ndim == 2:
            mask_shape = (embeddings.shape[0], 1)
        elif embeddings.ndim == 3:
            mask_shape = embeddings.shape[:2]
        else:
            raise ValueError(f"slot_embeddings must be [B, D] or [B, J, D], got {tuple(embeddings.shape)}")
        return {"embeddings": embeddings, "mask": torch.ones(mask_shape, device=embeddings.device, dtype=embeddings.dtype)}
    return batch["slots"]
