from __future__ import annotations

import torch
from torch import nn

from effectcma_flow.data.effects import EffectSpec, apply_effect
from effectcma_flow.models.effect_mapper import EffectMapper
from effectcma_flow.models.operator_bank import ResidualOperatorBank
from effectcma_flow.models.build import build_model
from effectcma_flow.models.ts_encoder import ChannelEncoder
from effectcma_flow.models.text_to_ts_flow import TextToTSFlow, _series_router_stats as text2ts_series_router_stats
from effectcma_flow.evaluation.sampler import euler_sample_text2ts
from effectcma_flow.training.train_step import cfm_train_step
from effectcma_flow.training.utils import text_condition_from_batch


def _config():
    return {
        "task": {"mode": "edit"},
        "model": {
            "d_model": 16,
            "patch_len": 3,
            "num_operators": 3,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "operator_hidden": 8,
            "operator_t_dim": 4,
        },
        "text_encoder": {"mode": "hash", "hash_dim": 32},
    }


def _text2ts_config():
    cfg = _config()
    cfg["task"] = {"mode": "text2ts"}
    return cfg


def _v3_text2ts_config():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "num_operators": 3,
            "mapper_bounded_field_gate": False,
            "mapper_flow_time_condition": False,
            "mapper_operator_router": "dual",
            "mapper_time_segment_scales": [1, 3],
            "operator_architecture": "structural",
            "operator_channel_heads": 2,
            "operator_context_mode": "none",
            "operator_context_film": False,
        }
    )
    return cfg


def _batch(batch_size=2, length=12, channels=4):
    base = torch.randn(batch_size, length, channels)
    ys, masks = [], []
    for i in range(batch_size):
        y, m = apply_effect(base[i], EffectSpec("spike", 2, 7, (1,), 0.8))
        ys.append(y)
        masks.append(m)
    return {
        "B": base,
        "Y": torch.stack(ys),
        "mask": torch.stack(masks),
        "slots": [["strong local spike in the middle segment for channel 1"] for _ in range(batch_size)],
    }


def _text_batch(batch_size=2, length=12, channels=4):
    return {
        "Y": torch.randn(batch_size, length, channels),
        "caption": [f"weather caption {i}" for i in range(batch_size)],
        "slots": [[f"weather caption {i}"] for i in range(batch_size)],
    }


def test_model_forward_finite():
    model = build_model(_config(), sequence_length=12, num_channels=4)
    batch = _batch()
    t = torch.rand(2)
    x_t = (1 - t[:, None, None]) * batch["B"] + t[:, None, None] * batch["Y"]
    out, aux = model(batch["B"], x_t, t, batch["slots"])
    assert out.shape == (2, 12, 4)
    assert torch.isfinite(out).all()
    assert aux["G"].shape[:3] == (2, 12, 4)
    assert aux["A_t_rank"].shape[:3] == (2, 1, 4)
    assert aux["alpha"].shape == (2, 1, 4)


def test_missing_task_mode_defaults_to_text2ts():
    cfg = _config()
    cfg.pop("task")
    model = build_model(cfg, sequence_length=12, num_channels=4)
    assert isinstance(model, TextToTSFlow)


def test_cfm_train_step_updates_trainable_params():
    model = build_model(_config(), sequence_length=12, num_channels=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.mapper.op_proto.detach().clone()
    result = cfm_train_step(model, _batch(), opt, text_encoder_mode="hash")
    assert torch.isfinite(result["loss"])
    assert not torch.allclose(before, model.mapper.op_proto.detach())


def test_text2ts_model_forward_and_train_step_updates_params():
    model = build_model(_text2ts_config(), sequence_length=12, num_channels=4)
    batch = _text_batch()
    t = torch.rand(2)
    x_t = torch.randn(2, 12, 4)
    out, aux = model(x_t, t, [[text] for text in batch["caption"]])
    assert out.shape == (2, 12, 4)
    assert torch.isfinite(out).all()
    assert aux["G"].shape[:3] == (2, 12, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.mapper.op_proto.detach().clone()
    result = cfm_train_step(model, batch, opt, text_encoder_mode="hash", task_mode="text2ts")
    assert torch.isfinite(result["loss"])
    assert not torch.allclose(before, model.mapper.op_proto.detach())


def test_text2ts_model_supports_non_divisible_patch_length():
    cfg = _text2ts_config()
    cfg["model"]["patch_len"] = 6
    model = build_model(cfg, sequence_length=13, num_channels=3)
    x_t = torch.randn(2, 13, 3)
    out, aux = model(x_t, torch.rand(2), [["caption"], ["caption"]])
    assert out.shape == (2, 13, 3)
    assert aux["G"].shape[:3] == (2, 13, 3)
    assert torch.isfinite(out).all()


class SpyZeroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_text_condition = None

    def forward(self, base, x_t, t, text_condition):
        self.seen_text_condition = text_condition
        batch = base.shape[0]
        device = base.device
        aux = {
            "A_o": torch.full((batch, 1, 2), 0.5, device=device),
        }
        return torch.zeros_like(base), aux


def test_cfm_loss_is_exact_and_model_does_not_receive_eval_metadata():
    batch = _batch(batch_size=1)
    batch["spec"] = [{"effect_type": "spike"}]
    batch["attrs_idx"] = torch.ones(1, 7)
    model = SpyZeroModel()
    result = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash")
    expected = torch.mean((batch["Y"] - batch["B"]) ** 2)
    assert torch.allclose(result["loss"], expected)
    assert model.seen_text_condition is batch["slots"]


def test_balanced_cfm_loss_reports_inside_outside_parts():
    batch = _batch(batch_size=1)
    model = SpyZeroModel()
    result = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", cfm_loss_mode="balanced")
    assert "loss_inside" in result
    assert "loss_outside" in result
    assert torch.allclose(result["loss"], 0.5 * (result["loss_inside"] + result["loss_outside"]))


def test_precomputed_text_condition_requires_slot_embeddings_not_captions():
    batch = {"caption_embeddings": torch.randn(2, 3, 128)}
    try:
        text_condition_from_batch(batch, "precomputed")
    except ValueError as exc:
        assert "slot_embeddings" in str(exc)
    else:
        raise AssertionError("caption embeddings were accepted as precomputed effect condition")


def test_precomputed_text_condition_mask_shapes():
    bd = {"slot_embeddings": torch.randn(2, 128)}
    bjd = {"slot_embeddings": torch.randn(2, 3, 128)}
    assert text_condition_from_batch(bd, "precomputed")["mask"].shape == (2, 1)
    assert text_condition_from_batch(bjd, "precomputed")["mask"].shape == (2, 3)


def test_precomputed_caption_condition_is_allowed_for_text2ts():
    batch = {"caption_embeddings": torch.randn(2, 128)}
    cond = text_condition_from_batch(batch, "precomputed", condition_key="caption")
    assert cond["embeddings"].shape == (2, 128)
    assert cond["mask"].shape == (2, 1)


def test_channel_encoder_patch_and_channel_mixing_shape():
    encoder = ChannelEncoder(18, 5, 20, patch_len=6, stride=3, temporal_layers=1, channel_layers=1, heads=4)
    out = encoder(torch.randn(3, 18, 5))
    assert out.shape == (3, 5, 20)
    assert torch.isfinite(out).all()


def test_effect_mapper_multirank_field_shapes():
    mapper = EffectMapper(d_model=16, num_operators=5, field_rank=3, slot_heads=4)
    slot_tokens = torch.randn(2, 2, 16)
    time_tokens = torch.randn(2, 4, 16)
    channel_tokens = torch.randn(2, 3, 16)
    g, aux = mapper(slot_tokens, time_tokens, channel_tokens, torch.ones(2, 2))
    assert g.shape == (2, 4, 3, 5)
    assert aux["A_o"].shape == (2, 2, 5)
    assert aux["A_o_rank"].shape == (2, 2, 3, 5)
    assert torch.isfinite(g).all()


def test_operator_bank_has_independent_temporal_experts():
    bank = ResidualOperatorBank(num_channels=4, num_operators=3, hidden=8, t_dim=4, depth=2)
    assert len(bank.experts) == 3
    assert bank.experts[0] is not bank.experts[1]
    x_t = torch.randn(2, 12, 4)
    out = bank(x_t, x_t * 0.5, torch.rand(2))
    assert out.shape == (2, 12, 4, 3)
    assert torch.isfinite(out).all()


def test_v3_structural_text2ts_forward_and_sampler():
    model = build_model(_v3_text2ts_config(), sequence_length=13, num_channels=3)
    x_t = torch.randn(2, 13, 3)
    text = [["steady rain and falling pressure"], ["clear sky and rising temperature"]]
    out, aux = model(x_t, torch.rand(2), text)
    assert out.shape == (2, 13, 3)
    assert torch.isfinite(out).all()
    assert aux["G"].shape == (2, 13, 3, 3)
    assert aux["V"].shape == (2, 13, 3, 3)
    assert aux["A_o_text"].shape[-1] == 3
    assert aux["A_o_series"].shape[-1] == 3
    sampled, sample_aux = euler_sample_text2ts(model, torch.zeros(2, 13, 3), text, steps=2)
    assert sampled.shape == (2, 13, 3)
    assert sample_aux["G"].shape[-1] == 3
    assert torch.isfinite(sampled).all()


def test_v3_dual_router_has_distinct_text_and_series_paths():
    model = build_model(_v3_text2ts_config(), sequence_length=12, num_channels=3)
    model.eval()
    t = torch.full((2,), 0.4)
    x_a = torch.randn(2, 12, 3)
    x_b = x_a * 0.1 + 2.0
    text_a = [["low-frequency seasonal weather"], ["low-frequency seasonal weather"]]
    text_b = [["sharp high-frequency noisy spikes"], ["sharp high-frequency noisy spikes"]]
    _, aux_a = model(x_a, t, text_a)
    _, aux_text_changed = model(x_a, t, text_b)
    _, aux_series_changed = model(x_b, t, text_a)
    assert not torch.allclose(aux_a["A_o_text"], aux_text_changed["A_o_text"])
    assert not torch.allclose(aux_a["G"], aux_text_changed["G"])
    assert torch.allclose(aux_a["V"], aux_text_changed["V"])
    assert not torch.allclose(aux_a["A_o_series"], aux_series_changed["A_o_series"])
    assert torch.allclose(aux_a["A_o_text"], aux_series_changed["A_o_text"])


def test_structural_operator_bank_requires_three_operators():
    cfg = _v3_text2ts_config()
    cfg["model"]["num_operators"] = 4
    try:
        build_model(cfg, sequence_length=12, num_channels=3)
    except ValueError as exc:
        assert "num_operators=3" in str(exc)
    else:
        raise AssertionError("structural operator bank accepted a non-3 operator count")


def test_edit_mode_can_use_dual_router_and_structural_bank():
    cfg = _config()
    cfg["model"].update(
        {
            "mapper_operator_router": "dual",
            "operator_architecture": "structural",
            "operator_channel_heads": 2,
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    batch = _batch()
    t = torch.rand(2)
    x_t = (1 - t[:, None, None]) * batch["B"] + t[:, None, None] * batch["Y"]
    out, aux = model(batch["B"], x_t, t, batch["slots"])
    assert out.shape == (2, 12, 4)
    assert aux["A_o_series"].shape == (2, 3)
    assert aux["text_context"].shape[0] == 2
    assert torch.isfinite(out).all()


def test_structural_bank_is_router_only_and_channel_expert_owns_channel_mixing():
    bank = ResidualOperatorBank(
        num_channels=3,
        num_operators=3,
        hidden=8,
        t_dim=4,
        architecture="structural",
        context_dim=16,
        context_mode="none",
        context_film=False,
        channel_heads=2,
    )
    assert bank.channel_mixer is None
    x_t = torch.randn(2, 12, 3)
    t = torch.rand(2)
    out_a = bank(x_t, None, t, context=torch.randn(2, 16))
    out_b = bank(x_t, None, t, context=torch.randn(2, 16))
    assert torch.allclose(out_a, out_b)
    assert out_a.shape == (2, 12, 3, 3)


def test_series_router_stats_support_half_precision_cpu_fft_path():
    x = torch.randn(2, 12, 3).half()
    stats = text2ts_series_router_stats(x)
    assert stats.shape == (2, 7)
    assert stats.dtype == torch.float16
    assert torch.isfinite(stats.float()).all()
