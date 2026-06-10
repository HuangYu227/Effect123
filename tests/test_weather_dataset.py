from __future__ import annotations

import numpy as np
import pytest
import torch

from effectcma_flow.data.weather_dataset import (
    WeatherRawCaptionDataset,
    WeatherSemiSyntheticDataset,
    collate_raw_caption_batch,
    compute_train_stats,
    load_weather_caption_embeddings,
)


def test_weather_loader_dtype_shape_and_train_stats(fake_weather_root):
    stats = compute_train_stats(fake_weather_root)
    ds = WeatherSemiSyntheticDataset(fake_weather_root, "valid", stats=stats, seed=123)
    item = ds[0]
    assert item["B"].dtype == torch.float32
    assert item["B"].shape == (12, 4)
    assert item["Y"].shape == (12, 4)
    assert item["mask"].shape == (12, 4)
    train = np.load(fake_weather_root / "train_ts.npy")
    assert torch.allclose(stats["mean"], torch.from_numpy(train.mean(axis=(0, 1), keepdims=True).astype(np.float32)))


def test_caption_embedding_reshape_and_split_alignment(fake_weather_root):
    splits = load_weather_caption_embeddings(fake_weather_root)
    assert splits["train"].shape == (5, 3, 128)
    assert splits["valid"].shape == (3, 3, 128)
    assert splits["test"].shape == (2, 3, 128)
    flat = np.load(fake_weather_root / "text_embeddings_128_all_caps.npy")
    assert np.allclose(splits["valid"][0, 0], flat[5 * 3])


def test_caption_embedding_loader_rejects_wrong_dim(fake_weather_root):
    np.save(fake_weather_root / "text_embeddings_128_all_caps.npy", np.zeros((30, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        load_weather_caption_embeddings(fake_weather_root)


def test_caption_embedding_loader_rejects_wrong_count(fake_weather_root):
    np.save(fake_weather_root / "text_embedding_caption_counts.npy", np.full((11,), 3, dtype=np.int64))
    with pytest.raises(ValueError, match="split total"):
        load_weather_caption_embeddings(fake_weather_root)


def test_precomputed_dataset_uses_synthetic_slot_embeddings(fake_weather_root):
    ds = WeatherSemiSyntheticDataset(fake_weather_root, "train", include_precomputed_embeddings=True, seed=123)
    item = ds[0]
    assert "slot_embeddings" in item
    assert "caption_embeddings" not in item
    assert item["slot_embeddings"].shape == (len(item["slots"]), 128)


def test_raw_caption_dataset_returns_text_only_training_fields(fake_weather_root):
    stats = compute_train_stats(fake_weather_root)
    ds = WeatherRawCaptionDataset(
        fake_weather_root,
        "valid",
        stats=stats,
        seed=123,
        caption_policy="cyclic",
        include_precomputed_embeddings=True,
    )
    item = ds[1]
    assert item["Y"].shape == (12, 4)
    assert item["ts"].shape == (12, 4)
    assert item["caption"] == "valid sample 1 caption 1"
    assert item["slots"] == [item["caption"]]
    assert item["caption_embedding"].shape == (128,)
    assert "B" not in item
    batch = collate_raw_caption_batch([item, ds[2]])
    assert batch["Y"].shape == (2, 12, 4)
    assert batch["caption_embeddings"].shape == (2, 128)


def test_raw_caption_dataset_accepts_single_caption_per_sample(fake_weather_root):
    for split in ("train", "valid", "test"):
        caps = np.load(fake_weather_root / f"{split}_text_caps.npy", allow_pickle=True)
        np.save(fake_weather_root / f"{split}_text_caps.npy", caps[:, :1])
    stats = compute_train_stats(fake_weather_root)
    ds = WeatherRawCaptionDataset(fake_weather_root, "train", stats=stats, seed=123, caption_policy="random")
    item = ds[3]
    assert item["caption"] == "train sample 3 caption 0"
    assert item["caption_id"] == 0
    batch = collate_raw_caption_batch([item])
    assert batch["caption"] == ["train sample 3 caption 0"]
