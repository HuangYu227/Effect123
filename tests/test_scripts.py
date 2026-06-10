from __future__ import annotations

import pytest
import torch

from effectcma_flow.training.checkpoint import checkpoint_eval_config, checkpoint_stats_or_none, validate_checkpoint_payload
from scripts.eval_weather import run_eval
from scripts.sample_weather import run_sample


def _cfg(fake_weather_root):
    return {
        "data": {
            "root": str(fake_weather_root),
            "window_length": None,
            "normalize": True,
            "caption_policy": "cyclic",
            "effect_types": ["spike"],
            "seed": 1,
        },
        "model": {
            "d_model": 16,
            "patch_len": 3,
            "num_operators": 3,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "operator_hidden": 8,
            "operator_t_dim": 4,
        },
        "text_encoder": {"mode": "hash", "hash_dim": 32, "precomputed_dim": 128},
        "train": {"batch_size": 2, "device": "cpu"},
        "sample": {"steps": 1},
        "task": {"mode": "text2ts"},
    }


def test_eval_requires_checkpoint_unless_explicit_random(fake_weather_root):
    with pytest.raises(ValueError, match="requires --checkpoint"):
        run_eval(_cfg(fake_weather_root), checkpoint=None, split="valid", max_batches=1)


def test_sample_requires_checkpoint_unless_explicit_random(fake_weather_root, tmp_path):
    with pytest.raises(ValueError, match="requires --checkpoint"):
        run_sample(_cfg(fake_weather_root), checkpoint=None, split="valid", index=0, output=str(tmp_path / "x.png"))


def test_eval_text2ts_random_init_smoke(fake_weather_root):
    metrics = run_eval(_cfg(fake_weather_root), checkpoint=None, split="valid", max_batches=1, allow_random_init=True)
    assert "mse" in metrics
    assert "mask_iou" not in metrics


def test_sample_text2ts_random_init_smoke(fake_weather_root, tmp_path):
    out = run_sample(
        _cfg(fake_weather_root),
        checkpoint=None,
        split="valid",
        index=0,
        output=str(tmp_path / "sample.png"),
        allow_random_init=True,
    )
    assert "caption" in out
    assert (tmp_path / "sample.png").exists()


def test_prompt_sample_text2ts_random_init_smoke(fake_weather_root, tmp_path):
    out = run_sample(
        _cfg(fake_weather_root),
        checkpoint=None,
        split="valid",
        index=0,
        output=str(tmp_path / "prompt.png"),
        npy_output=str(tmp_path / "prompt.npy"),
        caption="a cold windy weather sequence",
        allow_random_init=True,
    )
    assert out["shape"] == [12, 4]
    assert (tmp_path / "prompt.png").exists()
    assert (tmp_path / "prompt.npy").exists()


def test_checkpoint_text_mode_override_conflict_raises(fake_weather_root):
    payload = {"config": _cfg(fake_weather_root), "task_mode": "text2ts"}
    payload["config"]["text_encoder"]["mode"] = "hash"
    with pytest.raises(ValueError, match="Checkpoint was trained"):
        checkpoint_eval_config(_cfg(fake_weather_root), payload, text_encoder_override="hf")


def test_checkpoint_task_mode_override_conflict_raises(fake_weather_root):
    payload = {"config": _cfg(fake_weather_root), "task_mode": "text2ts"}
    with pytest.raises(ValueError, match="task.mode"):
        checkpoint_eval_config(_cfg(fake_weather_root), payload, task_mode_override="edit")


def test_checkpoint_eval_preserves_runtime_batch_override(fake_weather_root):
    payload = {"config": _cfg(fake_weather_root), "task_mode": "text2ts"}
    runtime = _cfg(fake_weather_root)
    runtime["train"]["batch_size"] = 7
    runtime["train"]["eval_batch_size"] = 3
    cfg = checkpoint_eval_config(runtime, payload)
    assert cfg["train"]["batch_size"] == 7
    assert cfg["train"]["eval_batch_size"] == 3


def test_checkpoint_stats_required():
    with pytest.raises(ValueError, match="missing 'stats'"):
        checkpoint_stats_or_none({"config": {}})
    stats = checkpoint_stats_or_none({"stats": {"mean": torch.zeros(1, 1, 2), "std": torch.ones(1, 1, 2)}})
    assert stats["mean"].dtype == torch.float32
    assert stats["mean"].device.type == "cpu"


def test_checkpoint_payload_requires_schema_version(fake_weather_root):
    payload = {"model": {}, "config": _cfg(fake_weather_root), "stats": {}, "step": 1}
    with pytest.raises(ValueError, match="schema_version"):
        validate_checkpoint_payload(payload)


def test_checkpoint_payload_requires_task_mode(fake_weather_root):
    payload = {
        "schema_version": 2,
        "model": {},
        "model_class": "TextToTSFlow",
        "config": _cfg(fake_weather_root),
        "stats": {"mean": torch.zeros(1, 1, 4), "std": torch.ones(1, 1, 4)},
        "step": 1,
    }
    with pytest.raises(ValueError, match="task_mode"):
        validate_checkpoint_payload(payload)
