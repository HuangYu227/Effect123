from __future__ import annotations

import torch

import effectcma_flow.models.text_encoder as text_encoder_module
from effectcma_flow.models.text_encoder import HashTextEncoder, PrecomputedTextEncoder, build_text_encoder


def test_hash_text_encoder_shape_and_determinism():
    enc = HashTextEncoder(raw_dim=64, d_model=16)
    slots = [["strong spike in the middle for channel 2"], ["weak trend in early segment for channel 1"]]
    z1, mask1 = enc(slots)
    z2, mask2 = enc(slots)
    assert z1.shape == (2, 1, 16)
    assert torch.allclose(z1, z2)
    assert torch.equal(mask1, mask2)


def test_hash_text_encoder_follows_module_dtype():
    enc = HashTextEncoder(raw_dim=64, d_model=16).half()
    z, mask = enc([["strong spike"]])
    assert z.dtype == torch.float16
    assert mask.dtype == torch.float16


def test_precomputed_text_encoder_shape():
    enc = PrecomputedTextEncoder(input_dim=128, d_model=16)
    embeddings = torch.randn(3, 2, 128)
    z, mask = enc({"embeddings": embeddings})
    assert z.shape == (3, 2, 16)
    assert mask.shape == (3, 2)


def test_precomputed_text_encoder_rejects_bad_mask_shape():
    enc = PrecomputedTextEncoder(input_dim=128, d_model=16)
    embeddings = torch.randn(3, 128)
    bad_mask = torch.ones(3, 128)
    try:
        enc({"embeddings": embeddings, "mask": bad_mask})
    except ValueError as exc:
        assert "mask shape" in str(exc)
    else:
        raise AssertionError("bad precomputed mask shape was accepted")


def test_precomputed_text_encoder_accepts_bd_with_b1_mask():
    enc = PrecomputedTextEncoder(input_dim=128, d_model=16)
    embeddings = torch.randn(3, 128)
    z, mask = enc({"embeddings": embeddings, "mask": torch.ones(3, 1)})
    assert z.shape == (3, 1, 16)
    assert mask.shape == (3, 1)


def test_hf_encoder_can_fallback_to_hash(monkeypatch):
    def fail_hf(*args, **kwargs):
        raise RuntimeError("mock hf load failure")

    monkeypatch.setattr(text_encoder_module, "HFTextEncoder", fail_hf)
    enc = build_text_encoder(
        {
            "mode": "hf",
            "hf_model_name": "mock-model",
            "local_files_only": True,
            "fallback_mode": "hash",
            "fallback_hash_dim": 32,
        },
        d_model=8,
    )
    assert isinstance(enc, HashTextEncoder)
