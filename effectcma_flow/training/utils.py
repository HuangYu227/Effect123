"""Training utilities for Clean V6: single-loss Flow Matching.

This module provides text-condition construction without artificial semantic
role slots. The caption is passed as-is to the text encoder. If the dataset
naturally provides multiple caption candidates, they can be used as separate
condition slots via ``caption_slot_strategy='candidates'``.
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Any, Iterable

import numpy as np
import torch


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

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
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


# ---------------------------------------------------------------------------
# Text condition construction
# ---------------------------------------------------------------------------

def text_condition_from_batch(
    batch: dict[str, Any],
    mode: str,
    *,
    condition_key: str = "slots",
    caption_slot_strategy: str = "single",
    max_caption_slots: int = 1,
    include_all_caption_candidates: bool = False,
):
    """Build text condition for the model.

    Clean V6 does NOT use artificial semantic role slots ([trend], [frequency],
    etc.). The caption is passed as-is to the text encoder.

    Strategies:
        ``single``: one slot per sample — the raw caption.
        ``candidates``: one slot per natural caption candidate from the dataset
            (requires ``include_all_caption_candidates=True``).

    For precomputed embeddings, the stored embedding is returned directly.
    """
    if mode == "precomputed":
        embedding_key = _embedding_key(condition_key)
        if embedding_key not in batch:
            raise ValueError(
                f"precomputed text encoder mode requires batch[{embedding_key!r}] "
                f"for {condition_key!r} conditioning"
            )
        embeddings = batch[embedding_key]
        if not torch.is_tensor(embeddings):
            raise TypeError(f"batch[{embedding_key!r}] must be a torch.Tensor")
        if embeddings.ndim == 2:
            mask_shape = (embeddings.shape[0], 1)
        elif embeddings.ndim == 3:
            mask_shape = embeddings.shape[:2]
        else:
            raise ValueError(f"{embedding_key} must be [B, D] or [B, J, D], got {tuple(embeddings.shape)}")
        return {
            "embeddings": embeddings,
            "mask": torch.ones(mask_shape, device=embeddings.device, dtype=embeddings.dtype),
        }

    if condition_key == "caption":
        captions = _extract_caption_candidates(batch, include_all=include_all_caption_candidates)
        strategy = str(caption_slot_strategy).lower()
        if strategy == "single":
            return [[str(cands[0])] for cands in captions]
        if strategy == "candidates":
            return [_dedupe_and_clip([str(x) for x in cands], max(1, int(max_caption_slots))) for cands in captions]
        raise ValueError(
            f"caption_slot_strategy must be 'single' or 'candidates' in Clean V6, got {strategy!r}. "
            f"Artificial semantic role slots are not supported."
        )

    if condition_key not in batch:
        raise ValueError(f"batch does not contain text condition key {condition_key!r}")
    return batch[condition_key]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _embedding_key(condition_key: str) -> str:
    if condition_key == "slots":
        return "slot_embeddings"
    if condition_key == "caption":
        return "caption_embeddings"
    if condition_key.endswith("_embeddings"):
        return condition_key
    return f"{condition_key}_embeddings"


def _extract_caption_candidates(
    batch: dict[str, Any],
    *,
    include_all: bool,
) -> list[list[str]]:
    if "caption" not in batch:
        raise ValueError("caption conditioning requires batch['caption']")
    captions = [str(text) for text in batch["caption"]]
    if not include_all:
        return [[text] for text in captions]
    candidates = batch.get("caption_candidates", None)
    if candidates is None:
        candidates = batch.get("captions", None)
    if candidates is None:
        return [[text] for text in captions]
    if len(candidates) != len(captions):
        raise ValueError(
            f"caption_candidates length {len(candidates)} does not match batch caption length {len(captions)}"
        )
    output: list[list[str]] = []
    for selected, row in zip(captions, candidates):
        if isinstance(row, str):
            row_values = [row]
        else:
            row_values = [str(x) for x in row if str(x).strip()]
        output.append(_dedupe_and_clip([selected, *row_values], max_items=16))
    return output


def _normalize_text(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, max_chars: int) -> str:
    text = _normalize_text(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def _dedupe_and_clip(items: Iterable[str], max_items: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = _normalize_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
        if len(out) >= max_items:
            break
    return out or [""]


def stable_hash_int(text: str, *, bits: int = 32) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    value = int.from_bytes(digest[: max(1, bits // 8)], "little", signed=False)
    return value % (2**bits)
