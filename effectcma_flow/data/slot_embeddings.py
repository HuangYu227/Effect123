from __future__ import annotations

import hashlib
import re

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def hash_slot_texts(slots: list[str], dim: int) -> np.ndarray:
    """Deterministic offline slot embeddings for precomputed smoke paths.

    These embeddings are derived from the synthetic effect slots, not from the
    original Weather captions. Server experiments should replace this with a
    frozen encoder cache for the same synthetic slot strings.
    """
    out = np.zeros((len(slots), dim), dtype=np.float32)
    for i, text in enumerate(slots):
        out[i] = hash_text(text, dim)
    return out


def hash_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    tokens = TOKEN_RE.findall(text.lower()) or [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec

