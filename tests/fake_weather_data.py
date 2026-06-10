from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def create_fake_weather_root(root: str | Path) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sizes = {"train": 5, "valid": 3, "test": 2}
    length, channels = 12, 4
    offset = 0
    for split, n in sizes.items():
        ts = np.arange(offset, offset + n * length * channels, dtype=np.float32).reshape(n, length, channels)
        ts = ts / 100.0
        np.save(root / f"{split}_ts.npy", ts)
        caps = np.array([[f"{split} sample {i} caption {j}" for j in range(3)] for i in range(n)])
        np.save(root / f"{split}_text_caps.npy", caps)
        np.save(root / f"{split}_attrs_idx.npy", np.zeros((n, 7), dtype=np.int64))
        offset += n * length * channels
    total = sum(sizes.values())
    np.save(root / "text_embeddings_128_all_caps.npy", np.arange(total * 3 * 128, dtype=np.float32).reshape(total * 3, 128))
    np.save(root / "text_embedding_caption_counts.npy", np.full((total,), 3, dtype=np.int64))
    meta = {
        "attr_list": ["season", "time", "weather", "temperature", "wind", "atmospher", "humidity"],
        "attr_n_ops": [4, 4, 6, 4, 9, 4, 4],
        "attrs_split": sizes,
        "final_split": sizes,
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return root

