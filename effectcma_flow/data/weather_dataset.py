from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from effectcma_flow.data.effects import EffectSpec, SUPPORTED_EFFECTS, apply_effect, normalize_channels
from effectcma_flow.data.slot_embeddings import hash_slot_texts
from effectcma_flow.data.text_templates import spec_to_text


SPLITS = ("train", "valid", "test")


def compute_train_stats(root: str | Path) -> dict[str, torch.Tensor]:
    train = _load_ts(root, "train")
    mean = torch.from_numpy(train.mean(axis=(0, 1), keepdims=True).astype(np.float32))
    std = torch.from_numpy(train.std(axis=(0, 1), keepdims=True).astype(np.float32))
    std = torch.clamp(std, min=1e-6)
    return {"mean": mean, "std": std}


def load_weather_caption_embeddings(root: str | Path, *, expected_dim: int = 128) -> dict[str, np.ndarray]:
    """Load caption embeddings as `[samples, captions_per_sample, dim]`.

    VerbalTS Weather stores all caption embeddings in one flat array with a
    separate count vector. Some converted datasets instead store one embedding
    array per split, either as `[N, J, D]` or `[N, D]`. Support both layouts so
    precomputed text conditioning can be used without changing the dataset
    files.
    """
    root = Path(root)
    split_embeddings = _load_split_caption_embeddings(root, expected_dim=expected_dim)
    if split_embeddings is not None:
        return split_embeddings

    counts_path = root / "text_embedding_caption_counts.npy"
    flat_path = root / "text_embeddings_128_all_caps.npy"
    if not counts_path.exists() or not flat_path.exists():
        raise FileNotFoundError(
            "Precomputed caption embeddings require either split files like "
            "'train_text_caps_embeddings_128.npy'/'train_cap_emb.npy' or the "
            "legacy Weather files 'text_embedding_caption_counts.npy' and "
            "'text_embeddings_128_all_caps.npy'."
        )

    counts = np.load(counts_path, allow_pickle=False)
    flat = np.load(flat_path, allow_pickle=False)
    if counts.ndim != 1:
        raise ValueError(f"caption counts must be 1-D, got {counts.shape}")
    split_sizes = {split: int(_load_ts(root, split).shape[0]) for split in SPLITS}
    total_samples = sum(split_sizes.values())
    if counts.shape[0] != total_samples:
        raise ValueError(f"caption counts rows {counts.shape[0]} do not match split total {total_samples}")
    if not np.all(counts == counts[0]):
        raise ValueError("Only a uniform number of captions per sample is supported for caption embeddings")
    captions_per_sample = int(counts[0])
    if captions_per_sample <= 0:
        raise ValueError("caption counts must be positive")
    expected = int(counts.sum())
    if flat.ndim != 2:
        raise ValueError(f"caption embeddings must be 2-D [sum(counts), {expected_dim}], got {flat.shape}")
    if flat.shape != (expected, expected_dim):
        raise ValueError(f"caption embeddings must have shape {(expected, expected_dim)}, got {flat.shape}")
    sample_embeddings = flat.reshape(total_samples, captions_per_sample, expected_dim).astype(np.float32)

    offsets = _split_offsets(root)
    return {
        split: sample_embeddings[start:end]
        for split, (start, end) in offsets.items()
    }


def _load_split_caption_embeddings(root: Path, *, expected_dim: int) -> dict[str, np.ndarray] | None:
    out: dict[str, np.ndarray] = {}
    for split in SPLITS:
        candidates = (
            root / f"{split}_text_caps_embeddings_{expected_dim}.npy",
            root / f"{split}_text_caps_embeddings.npy",
            root / f"{split}_cap_emb.npy",
        )
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return None
        arr = np.load(path, allow_pickle=False).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, None, :]
        if arr.ndim != 3:
            raise ValueError(f"{path.name} must have shape [N,D] or [N,J,D], got {arr.shape}")
        if arr.shape[-1] != expected_dim:
            raise ValueError(f"{path.name} expected embedding dim {expected_dim}, got {arr.shape[-1]}")
        rows = int(_load_ts(root, split).shape[0])
        if arr.shape[0] != rows:
            raise ValueError(f"{path.name} rows {arr.shape[0]} do not match {split}_ts rows {rows}")
        out[split] = arr
    return out


class WeatherSemiSyntheticDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        window_length: int | None = None,
        normalize: bool = True,
        stats: dict[str, torch.Tensor] | None = None,
        effect_types: list[str] | tuple[str, ...] | None = None,
        seed: int = 0,
        include_precomputed_embeddings: bool = False,
        precomputed_dim: int = 128,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.ts = _load_ts(self.root, split).astype(np.float32)
        self.captions = np.load(self.root / f"{split}_text_caps.npy", allow_pickle=True)
        self.attrs_idx = np.load(self.root / f"{split}_attrs_idx.npy", allow_pickle=False)
        self.meta = _load_meta(self.root)
        if self.ts.ndim != 3:
            raise ValueError(f"{split}_ts.npy must have shape [N, L, C], got {self.ts.shape}")
        _validate_caption_array(self.captions, self.ts.shape[0], split)
        if self.attrs_idx.shape[0] != self.ts.shape[0]:
            raise ValueError(f"{split}_attrs_idx.npy row count does not match time series")

        self.raw_length = int(self.ts.shape[1])
        self.num_channels = int(self.ts.shape[2])
        self.window_length = int(window_length) if window_length is not None else self.raw_length
        if not 1 <= self.window_length <= self.raw_length:
            raise ValueError(f"window_length must be in [1, {self.raw_length}], got {self.window_length}")

        self.normalize = bool(normalize)
        self.stats = stats if stats is not None else compute_train_stats(self.root)
        if self.normalize:
            self.mean = self.stats["mean"].float()
            self.std = self.stats["std"].float()
        else:
            self.mean = torch.zeros(1, 1, self.num_channels)
            self.std = torch.ones(1, 1, self.num_channels)

        self.effect_types = tuple(effect_types or SUPPORTED_EFFECTS)
        unknown = [name for name in self.effect_types if name not in SUPPORTED_EFFECTS]
        if unknown:
            raise ValueError(f"Unsupported effects in effect_types: {unknown}")

        self.include_precomputed_embeddings = bool(include_precomputed_embeddings)
        self.precomputed_dim = int(precomputed_dim)

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    @property
    def sequence_length(self) -> int:
        return self.window_length

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + int(index))
        raw = torch.from_numpy(self.ts[index]).float()
        base = self._crop(raw, rng)
        if self.normalize:
            base = (base - self.mean.squeeze(0)) / self.std.squeeze(0)
        spec = self._sample_spec(rng)
        target, mask = apply_effect(base, spec)
        full_text, slots = spec_to_text(spec, self.window_length)
        item: dict[str, Any] = {
            "B": base,
            "Y": target,
            "mask": mask,
            "slots": slots,
            "full_text": full_text,
            "effect_type": spec.effect_type,
            "spec": spec.to_dict(),
            "caption": str(self.captions[index, 0]),
        }
        if self.include_precomputed_embeddings:
            item["slot_embeddings"] = torch.from_numpy(hash_slot_texts(slots, self.precomputed_dim)).float()
        return item

    def _crop(self, series: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if self.window_length == self.raw_length:
            return series
        max_start = self.raw_length - self.window_length
        start = int(rng.integers(0, max_start + 1))
        return series[start : start + self.window_length]

    def _sample_spec(self, rng: np.random.Generator) -> EffectSpec:
        effect_type = str(rng.choice(self.effect_types))
        min_duration = max(2, int(round(0.15 * self.window_length)))
        max_duration = max(min_duration, int(round(0.55 * self.window_length)))
        duration = int(rng.integers(min_duration, max_duration + 1))
        start = int(rng.integers(0, self.window_length - duration + 1))
        end = start + duration
        num_channels = int(rng.integers(1, min(3, self.num_channels) + 1))
        channels = normalize_channels(rng.choice(self.num_channels, size=num_channels, replace=False).tolist())
        strength = _sample_strength(effect_type, rng)
        return EffectSpec(effect_type=effect_type, start=start, end=end, channels=channels, strength=strength)


class WeatherRawCaptionDataset(Dataset):
    """Weather split for VerbalTS-style text-to-time-series generation.

    The model condition is the selected caption only. The raw time series is
    returned for VerbalTS metric reference, while `Y` is normalized for CFM
    training when `normalize=True`.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        window_length: int | None = None,
        normalize: bool = True,
        stats: dict[str, torch.Tensor] | None = None,
        seed: int = 0,
        caption_policy: str = "random",
        include_precomputed_embeddings: bool = False,
        precomputed_dim: int = 128,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if caption_policy not in {"random", "cyclic", "first"}:
            raise ValueError("caption_policy must be one of {'random', 'cyclic', 'first'}")
        self.root = Path(root)
        self.split = split
        self.seed = int(seed)
        self.caption_policy = caption_policy
        self.ts = _load_ts(self.root, split).astype(np.float32)
        self.captions = np.load(self.root / f"{split}_text_caps.npy", allow_pickle=True)
        self.attrs_idx = np.load(self.root / f"{split}_attrs_idx.npy", allow_pickle=False)
        if self.ts.ndim != 3:
            raise ValueError(f"{split}_ts.npy must have shape [N, L, C], got {self.ts.shape}")
        _validate_caption_array(self.captions, self.ts.shape[0], split)
        if self.attrs_idx.shape[0] != self.ts.shape[0]:
            raise ValueError(f"{split}_attrs_idx.npy row count does not match time series")

        self.raw_length = int(self.ts.shape[1])
        self.num_channels = int(self.ts.shape[2])
        self.window_length = int(window_length) if window_length is not None else self.raw_length
        if not 1 <= self.window_length <= self.raw_length:
            raise ValueError(f"window_length must be in [1, {self.raw_length}], got {self.window_length}")

        self.normalize = bool(normalize)
        self.stats = stats if stats is not None else compute_train_stats(self.root)
        if self.normalize:
            self.mean = self.stats["mean"].float()
            self.std = self.stats["std"].float()
        else:
            self.mean = torch.zeros(1, 1, self.num_channels)
            self.std = torch.ones(1, 1, self.num_channels)

        self.include_precomputed_embeddings = bool(include_precomputed_embeddings)
        self.precomputed_dim = int(precomputed_dim)
        self.epoch = 0
        self.caption_embeddings: np.ndarray | None = None
        if self.include_precomputed_embeddings:
            self.caption_embeddings = load_weather_caption_embeddings(self.root, expected_dim=self.precomputed_dim)[split]

    def __len__(self) -> int:
        return int(self.ts.shape[0])

    @property
    def sequence_length(self) -> int:
        return self.window_length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rng = np.random.default_rng(self.seed + int(index) + 1_000_003 * self.epoch)
        raw = torch.from_numpy(self.ts[index]).float()
        raw = self._crop(raw, rng)
        target = (raw - self.mean.squeeze(0)) / self.std.squeeze(0) if self.normalize else raw
        cap_idx = self._caption_index(index, rng)
        caption = str(self.captions[index, cap_idx])
        caption_candidates = [str(text) for text in self.captions[index].tolist()]
        item: dict[str, Any] = {
            "Y": target.float(),
            "ts": raw.float(),
            "caption": caption,
            "caption_candidates": caption_candidates,
            "caption_id": cap_idx,
            "slots": [caption],
            "attrs_idx": torch.from_numpy(self.attrs_idx[index]).long(),
        }
        if self.caption_embeddings is not None:
            item["caption_embedding"] = torch.from_numpy(self.caption_embeddings[index, cap_idx]).float()
        return item

    def _crop(self, series: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        if self.window_length == self.raw_length:
            return series
        max_start = self.raw_length - self.window_length
        start = int(rng.integers(0, max_start + 1))
        return series[start : start + self.window_length]

    def _caption_index(self, index: int, rng: np.random.Generator) -> int:
        if self.caption_policy == "first":
            return 0
        if self.caption_policy == "cyclic":
            return int(index) % int(self.captions.shape[1])
        return int(rng.integers(0, self.captions.shape[1]))


def collate_effect_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "B": torch.stack([s["B"] for s in samples]),
        "Y": torch.stack([s["Y"] for s in samples]),
        "mask": torch.stack([s["mask"] for s in samples]),
        "slots": [s["slots"] for s in samples],
        "full_text": [s["full_text"] for s in samples],
        "effect_type": [s["effect_type"] for s in samples],
        "spec": [s["spec"] for s in samples],
        "caption": [s["caption"] for s in samples],
    }
    if "slot_embeddings" in samples[0]:
        batch["slot_embeddings"] = torch.stack([s["slot_embeddings"] for s in samples])
    return batch


def collate_raw_caption_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "Y": torch.stack([s["Y"] for s in samples]),
        "ts": torch.stack([s["ts"] for s in samples]),
        "caption": [s["caption"] for s in samples],
        "caption_candidates": [s["caption_candidates"] for s in samples],
        "caption_id": torch.tensor([s["caption_id"] for s in samples], dtype=torch.long),
        "slots": [s["slots"] for s in samples],
        "attrs_idx": torch.stack([s["attrs_idx"] for s in samples]),
    }
    if "caption_embedding" in samples[0]:
        batch["caption_embeddings"] = torch.stack([s["caption_embedding"] for s in samples])
    return batch


def _sample_strength(effect_type: str, rng: np.random.Generator) -> float:
    if effect_type == "volatility_up":
        return float(rng.uniform(1.2, 1.9))
    if effect_type == "volatility_down":
        return float(rng.uniform(0.35, 0.8))
    if effect_type in {"trend_down", "drop"}:
        return float(rng.uniform(0.25, 1.25))
    return float(rng.uniform(0.25, 1.25))


def _load_ts(root: str | Path, split: str) -> np.ndarray:
    root = Path(root)
    path = root / f"{split}_ts.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def _validate_caption_array(captions: np.ndarray, num_samples: int, split: str) -> None:
    if captions.ndim != 2:
        raise ValueError(f"{split}_text_caps.npy must have shape [N, J], got {captions.shape}")
    if captions.shape[0] != num_samples:
        raise ValueError(f"{split}_text_caps.npy row count {captions.shape[0]} does not match time series {num_samples}")
    if captions.shape[1] < 1:
        raise ValueError(f"{split}_text_caps.npy must contain at least one caption per sample, got {captions.shape}")


def _load_meta(root: Path) -> dict[str, Any]:
    with (root / "meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _split_offsets(root: str | Path) -> dict[str, tuple[int, int]]:
    root = Path(root)
    sizes = {split: int(_load_ts(root, split).shape[0]) for split in SPLITS}
    train_end = sizes["train"]
    valid_end = train_end + sizes["valid"]
    return {
        "train": (0, train_end),
        "valid": (train_end, valid_end),
        "test": (valid_end, valid_end + sizes["test"]),
    }
