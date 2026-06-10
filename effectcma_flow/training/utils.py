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


def text_condition_from_batch(batch: dict[str, Any], mode: str, *, condition_key: str = "slots"):
    if mode == "precomputed":
        embedding_key = _embedding_key(condition_key)
        if embedding_key not in batch:
            raise ValueError(f"precomputed text encoder mode requires batch[{embedding_key!r}] for {condition_key!r} conditioning")
        embeddings = batch[embedding_key]
        if embeddings.ndim == 2:
            mask_shape = (embeddings.shape[0], 1)
        elif embeddings.ndim == 3:
            mask_shape = embeddings.shape[:2]
        else:
            raise ValueError(f"{embedding_key} must be [B, D] or [B, J, D], got {tuple(embeddings.shape)}")
        return {"embeddings": embeddings, "mask": torch.ones(mask_shape, device=embeddings.device, dtype=embeddings.dtype)}
    if condition_key == "caption":
        if "caption" not in batch:
            raise ValueError("caption conditioning requires batch['caption']")
        return [[str(text)] for text in batch["caption"]]
    return batch["slots"]


def _embedding_key(condition_key: str) -> str:
    if condition_key == "slots":
        return "slot_embeddings"
    if condition_key == "caption":
        return "caption_embeddings"
    if condition_key.endswith("_embeddings"):
        return condition_key
    return f"{condition_key}_embeddings"
