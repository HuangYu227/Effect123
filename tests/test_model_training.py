from __future__ import annotations

import pytest
import torch
from torch import nn

from effectcma_flow.data.effects import EffectSpec, apply_effect
from effectcma_flow.models.effect_mapper import EffectMapper
from effectcma_flow.models.cross_modal_bridge import CrossModalConditionBridge, _balanced_sigmoid_contrastive_loss
from effectcma_flow.models.operator_bank import ResidualOperatorBank
from effectcma_flow.models.build import build_model
from effectcma_flow.models.text_to_ts_flow import TextToTSFlow
from effectcma_flow.models.ts_encoder import ChannelEncoder
from effectcma_flow.training.train_step import cfm_train_step
from effectcma_flow.training.utils import text_condition_from_batch
from effectcma_flow.evaluation.sampler import sample_text2ts


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


def test_build_model_requires_explicit_task_mode():
    cfg = _config()
    cfg.pop("task")
    try:
        build_model(cfg, sequence_length=12, num_channels=4)
    except ValueError as exc:
        assert "task.mode" in str(exc)
    else:
        raise AssertionError("build_model accepted a config without task.mode")


def _text2ts_config():
    cfg = _config()
    cfg["task"] = {"mode": "text2ts"}
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
    assert torch.isfinite(result["pred_v_rms"])
    assert torch.isfinite(result["target_v_rms"])
    assert torch.isfinite(result["pred_target_rms_ratio"])
    assert torch.isfinite(result["velocity_cos"])
    assert not torch.allclose(before, model.mapper.op_proto.detach())


def test_clean_v6_rejects_routing_loss_weight():
    """Clean V6 must raise ValueError if routing_loss_weight != 0."""
    model = build_model(_text2ts_config(), sequence_length=12, num_channels=4)
    batch = _text_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        cfm_train_step(
            model,
            batch,
            opt,
            text_encoder_mode="hash",
            task_mode="text2ts",
            routing_loss_weight=0.02,
        )
    except ValueError as exc:
        assert "V6.1" in str(exc) or "routing" in str(exc).lower()
    else:
        raise AssertionError("Clean V6 should reject routing_loss_weight != 0")


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


def test_routing_loss_weight_nonzero_raises_clean_v6_error():
    """Clean V6 rejects any non-zero routing_loss_weight."""
    batch = _batch(batch_size=1)
    model = SpyZeroModel()
    try:
        cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", routing_loss_weight=1.0)
    except ValueError as exc:
        assert "V6.1" in str(exc) or "routing" in str(exc).lower()
    else:
        raise AssertionError("routing loss weight 1.0 should be rejected by Clean V6")


def test_balanced_cfm_loss_reports_inside_outside_parts():
    batch = _batch(batch_size=1)
    model = SpyZeroModel()
    result = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", cfm_loss_mode="balanced")
    assert "loss_inside" in result
    assert "loss_outside" in result
    assert torch.allclose(result["loss"], 0.5 * (result["loss_inside"] + result["loss_outside"]))


def test_balanced_cfm_accepts_singleton_channel_mask():
    batch = _batch(batch_size=1)
    batch["mask"] = batch["mask"][..., :1]
    model = SpyZeroModel()
    result = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", cfm_loss_mode="balanced")
    assert torch.isfinite(result["loss"])


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


def test_caption_condition_single_and_candidates_strategy():
    """Clean V6 supports 'single' and 'candidates' strategies; rejects 'semantic'."""
    batch = {
        "caption": ["temperature rises with repeated daily oscillation"],
        "caption_candidates": [["temperature rises with repeated daily oscillation", "humidity has local spikes"]],
    }
    default_single = text_condition_from_batch(batch, "hash", condition_key="caption")
    single = text_condition_from_batch(batch, "hash", condition_key="caption", caption_slot_strategy="single")
    candidates = text_condition_from_batch(
        batch,
        "hash",
        condition_key="caption",
        caption_slot_strategy="candidates",
        include_all_caption_candidates=True,
        max_caption_slots=8,
    )
    assert default_single == [["temperature rises with repeated daily oscillation"]]
    assert single == [["temperature rises with repeated daily oscillation"]]
    assert len(candidates[0]) > 1
    # semantic strategy must be rejected
    try:
        text_condition_from_batch(batch, "hash", condition_key="caption", caption_slot_strategy="semantic")
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("semantic strategy should be rejected in Clean V6")


def test_precomputed_caption_condition_ignores_slot_strategy():
    """Precomputed mode returns stored embeddings without inspecting caption_slot_strategy."""
    batch = {"caption_embeddings": torch.randn(2, 128)}
    cond = text_condition_from_batch(batch, "precomputed", condition_key="caption")
    assert cond["embeddings"].shape == (2, 128)
    assert cond["mask"].shape == (2, 1)


def test_channel_encoder_patch_and_channel_mixing_shape():
    encoder = ChannelEncoder(18, 5, 20, patch_len=6, stride=3, temporal_layers=1, channel_layers=1, heads=4)
    out = encoder(torch.randn(3, 18, 5))
    assert out.shape == (3, 5, 20)
    assert torch.isfinite(out).all()


def test_channel_encoder_pads_tail_instead_of_truncating():
    encoder = ChannelEncoder(13, 3, 12, patch_len=6, stride=6, temporal_layers=1, channel_layers=1, heads=3)
    assert encoder.num_patches == 3
    out = encoder(torch.randn(2, 13, 3))
    assert out.shape == (2, 3, 12)
    assert torch.isfinite(out).all()


class CountingTextEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.calls = 0
        self.proj = nn.Linear(1, d_model)

    def forward(self, text_condition):
        self.calls += 1
        batch = len(text_condition)
        raw = torch.ones(batch, 1, 1, device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        mask = torch.ones(batch, 1, device=self.proj.weight.device, dtype=self.proj.weight.dtype)
        return self.proj(raw), mask


def test_text2ts_prepared_condition_reuses_text_encoder_output():
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=3,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
    )
    x = torch.randn(2, 12, 3)
    t = torch.rand(2)
    prepared = model.prepare_condition([["caption"], ["caption"]], device=x.device, dtype=x.dtype)
    model(x, t, prepared)
    model(x, t, prepared)
    assert encoder.calls == 1


def test_rk4_sampler_prepares_text_condition_once():
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=3,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
    )
    shape = torch.zeros(2, 12, 3)
    pred, aux = sample_text2ts(model, shape, [["caption"], ["caption"]], solver="rk4", steps=2)
    assert pred.shape == shape.shape
    assert aux["sampler_solver"] == "rk4"
    assert encoder.calls == 1


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


def test_structural_operator_bank_exposes_adaptive_frequency_aux():
    bank = ResidualOperatorBank(
        num_channels=4,
        num_operators=3,
        hidden=8,
        t_dim=4,
        architecture="structural",
        context_dim=16,
    )
    x_t = torch.randn(2, 12, 4)
    out = bank(x_t, None, torch.rand(2), context=torch.randn(2, 16))
    assert out.shape == (2, 12, 4, 3)
    aux = bank.last_aux
    assert "frequency_frequency_band_centers" in aux
    assert "frequency_frequency_band_widths" in aux
    assert "frequency_frequency_band_gate" in aux
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert bank.experts[2].band_center_logits.grad is not None
    assert torch.isfinite(bank.experts[2].band_center_logits.grad).all()
    assert bank.experts[2].band_log_widths.grad is not None
    assert torch.isfinite(bank.experts[2].band_log_widths.grad).all()


def test_structural_operator_bank_accepts_expert_context():
    bank = ResidualOperatorBank(
        num_channels=4,
        num_operators=3,
        hidden=8,
        t_dim=4,
        architecture="structural",
        context_dim=16,
    )
    x_t = torch.randn(2, 12, 4)
    context = torch.randn(2, 16)
    expert_context = torch.randn(2, 3, 16)
    out = bank(x_t, None, torch.rand(2), context=context, expert_context=expert_context)
    assert out.shape == (2, 12, 4, 3)
    assert "expert_context_norm" in bank.last_aux
    assert torch.isfinite(out).all()


def test_structural_operator_bank_accepts_multiview_context():
    bank = ResidualOperatorBank(
        num_channels=4,
        num_operators=3,
        hidden=8,
        t_dim=4,
        architecture="structural",
        context_dim=16,
        multiview_context=True,
    )
    x_t = torch.randn(2, 12, 4)
    context = torch.randn(2, 16)
    expert_context = torch.randn(2, 3, 16)
    channel_context = torch.randn(2, 4, 16)
    out = bank(
        x_t,
        None,
        torch.rand(2),
        context=context,
        expert_context=expert_context,
        channel_context=channel_context,
    )
    assert out.shape == (2, 12, 4, 3)
    assert "expert_time_context_norm" in bank.last_aux
    assert "channel_context_norm" in bank.last_aux
    assert torch.isfinite(out).all()


def test_cross_modal_bridge_forward_shapes_and_mask():
    bridge = CrossModalConditionBridge(
        d_model=16,
        num_channels=4,
        sequence_length=12,
        num_experts=3,
        patch_size=5,
        num_heads=4,
        num_spectral_tokens=3,
    )
    slot_tokens = torch.randn(2, 3, 16)
    slot_mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    x_t = torch.randn(2, 12, 4)
    bridged_tokens, bridge_context, expert_context, channel_context, aux = bridge(
        slot_tokens=slot_tokens,
        slot_mask=slot_mask,
        x_t=x_t,
        t=torch.rand(2),
    )
    assert bridged_tokens.shape == (2, 3, 16)
    assert bridge_context.shape == (2, 16)
    assert expert_context.shape == (2, 3, 16)
    assert channel_context.shape == (2, 4, 16)
    assert torch.allclose(bridged_tokens[0, 2], torch.zeros_like(bridged_tokens[0, 2]), atol=1e-6)
    assert torch.allclose(bridged_tokens[1], torch.zeros_like(bridged_tokens[1]), atol=1e-6)
    assert "bridge_alignment_loss" in aux
    assert "bridge_channel_context_norm" in aux
    assert torch.isfinite(aux["bridge_alignment_loss"])
    assert aux["bridge_text_to_state_entropy"].requires_grad is False


def test_cross_modal_bridge_latent_query_focal_shapes():
    bridge = CrossModalConditionBridge(
        d_model=16,
        num_channels=4,
        sequence_length=12,
        num_experts=3,
        patch_size=4,
        num_heads=4,
        num_spectral_tokens=3,
        focal_mode="latent_query",
        num_stage_tokens=3,
    )
    slot_tokens = torch.randn(2, 7, 16)
    slot_mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    x_t = torch.randn(2, 12, 4)
    bridged_tokens, bridge_context, expert_context, channel_context, aux = bridge(
        slot_tokens=slot_tokens,
        slot_mask=slot_mask,
        x_t=x_t,
        t=torch.rand(2),
    )
    assert bridged_tokens.shape == (2, 7, 16)
    assert bridge_context.shape == (2, 16)
    assert expert_context.shape == (2, 3, 16)
    assert channel_context.shape == (2, 4, 16)
    assert "bridge_focal_channel_entropy_norm" in aux
    assert "bridge_scale_context_norm" in aux
    assert "bridge_stage_context_norm" in aux
    assert torch.isfinite(bridge_context).all()


def test_cross_modal_bridge_patch_merger_connector_shapes_and_gradients():
    bridge = CrossModalConditionBridge(
        d_model=16,
        num_channels=4,
        sequence_length=13,
        num_experts=3,
        patch_size=4,
        num_heads=4,
        state_connector="patch_merger",
        temporal_merge=3,
        channel_merge=1,
        token_budget=5,
        alignment_mode="siglip",
    )
    slot_tokens = torch.randn(2, 6, 16, requires_grad=True)
    slot_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    x_t = torch.randn(2, 13, 4, requires_grad=True)
    bridged_text, bridge_context, expert_context, channel_context, aux = bridge(
        slot_tokens=slot_tokens,
        slot_mask=slot_mask,
        x_t=x_t,
        t=torch.rand(2),
    )
    assert bridged_text.shape == (2, 6, 16)
    assert bridge_context.shape == (2, 16)
    assert expert_context.shape == (2, 3, 16)
    assert channel_context.shape == (2, 4, 16)
    assert "bridge_alignment_loss" in aux
    assert "bridge_connector_patch_merger" in aux
    assert aux["bridge_memory_token_count"].item() == 5.0
    assert aux["bridge_patch_token_count"].item() == 20.0
    assert torch.isnan(aux["bridge_text_to_state_entropy"])
    assert torch.isnan(aux["bridge_state_to_text_entropy"])
    loss = (
        bridged_text.square().mean()
        + bridge_context.square().mean()
        + expert_context.square().mean()
        + channel_context.square().mean()
        + aux["bridge_alignment_loss"]
    )
    loss.backward()
    connector = bridge.patch_merger_connector
    assert connector is not None
    assert connector.patch_merger_mlp[0].weight.grad is not None
    assert torch.isfinite(connector.patch_merger_mlp[0].weight.grad).all()
    assert slot_tokens.grad is not None and torch.isfinite(slot_tokens.grad).all()
    assert x_t.grad is not None and torch.isfinite(x_t.grad).all()


def test_balanced_sigmoid_contrastive_loss_does_not_dilute_positives():
    for batch in (2, 8):
        logits = torch.zeros(batch, batch)
        logits.diagonal().fill_(1.0)
        loss = _balanced_sigmoid_contrastive_loss(logits)
        expected = 0.5 * (-torch.nn.functional.logsigmoid(torch.tensor(1.0)) - torch.nn.functional.logsigmoid(torch.tensor(0.0)))
        assert torch.allclose(loss, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# V6.1 Latent Regime Adapter + Global Operator Gate tests
# ---------------------------------------------------------------------------

from effectcma_flow.models.latent_regime_adapter import (
    LatentRegimeConditionAdapter,
    regime_orthogonal_loss,
)
from effectcma_flow.models.global_operator_gate import GlobalOperatorGate


def test_regime_adapter_forward_shapes():
    """LatentRegimeConditionAdapter produces correct output shapes."""
    adapter = LatentRegimeConditionAdapter(
        d_model=16,
        num_channels=4,
        num_regimes=4,
        append_regime_token=True,
    )
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    slot_tokens = torch.randn(2, 3, 16)
    slot_mask = torch.ones(2, 3)
    text_context = torch.randn(2, 16)
    st, sm, ctx, aux = adapter(x_t=x_t, t=t, slot_tokens=slot_tokens, slot_mask=slot_mask, text_context=text_context)
    # append_regime_token=True adds one token
    assert st.shape == (2, 4, 16)
    assert sm.shape == (2, 4)
    assert ctx.shape == (2, 16)
    assert "regime_prob" in aux
    assert "regime_ortho_loss" in aux
    assert aux["regime_prob"].shape == (2, 4)
    assert torch.isfinite(st).all()
    assert torch.isfinite(ctx).all()


def test_regime_adapter_no_append_token():
    """With append_regime_token=False, slot tokens are unchanged."""
    adapter = LatentRegimeConditionAdapter(
        d_model=16,
        num_channels=4,
        num_regimes=4,
        append_regime_token=False,
    )
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    slot_tokens = torch.randn(2, 3, 16)
    slot_mask = torch.ones(2, 3)
    text_context = torch.randn(2, 16)
    st, sm, ctx, aux = adapter(x_t=x_t, t=t, slot_tokens=slot_tokens, slot_mask=slot_mask, text_context=text_context)
    assert st.shape == (2, 3, 16)  # unchanged
    assert sm.shape == (2, 3)


def test_regime_adapter_uniform_posterior_outputs_uniform():
    """Uniform posterior mode keeps the adapter path but removes dynamic routing."""
    adapter = LatentRegimeConditionAdapter(
        d_model=16,
        num_channels=4,
        num_regimes=4,
        posterior_mode="uniform",
        append_regime_token=False,
    )
    x_t = torch.randn(3, 12, 4)
    t = torch.rand(3)
    slot_tokens = torch.randn(3, 2, 16)
    slot_mask = torch.ones(3, 2)
    text_context = torch.randn(3, 16)
    st, sm, ctx, aux = adapter(x_t=x_t, t=t, slot_tokens=slot_tokens, slot_mask=slot_mask, text_context=text_context)
    expected = torch.full_like(aux["regime_prob"], 0.25)
    assert st.shape == (3, 2, 16)
    assert sm.shape == (3, 2)
    assert ctx.shape == (3, 16)
    assert torch.allclose(aux["regime_prob"], expected)
    assert torch.allclose(aux["regime_usage"], torch.full_like(aux["regime_usage"], 0.25))
    assert torch.allclose(aux["regime_entropy_norm"], torch.tensor(1.0), atol=1e-6)
    assert aux["regime_posterior_uniform"].item() == 1.0


def test_regime_orthogonal_loss():
    """Orthogonal loss is zero for orthonormal bank, positive otherwise."""
    # Orthonormal bank
    bank = torch.eye(4)
    assert torch.allclose(regime_orthogonal_loss(bank), torch.tensor(0.0), atol=1e-7)
    # Single prototype
    single = torch.randn(1, 4)
    assert regime_orthogonal_loss(single).item() == 0.0
    # Non-orthogonal bank should give positive loss
    bank2 = torch.randn(4, 4)
    loss = regime_orthogonal_loss(bank2)
    assert loss.item() > 0.0


def test_global_operator_gate_forward_shapes():
    """GlobalOperatorGate produces correct gate shapes."""
    gate = GlobalOperatorGate(d_model=16, num_channels=4, num_operators=3)
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    text_context = torch.randn(2, 16)
    gate_prob, g, aux = gate(x_t=x_t, t=t, text_context=text_context, velocity_shape=(2, 12, 4, 3))
    assert gate_prob.shape == (2, 3)
    assert g.shape == (2, 12, 4, 3)
    assert "A_o" in aux
    assert "operator_gate_entropy" in aux
    assert torch.isfinite(gate_prob).all()
    assert torch.isfinite(g).all()


def test_global_operator_gate_attention_router_uses_tokens():
    """Attention router uses operator queries over state/text/time memory."""
    gate = GlobalOperatorGate(
        d_model=16,
        num_channels=4,
        num_operators=3,
        router_type="attention",
        num_heads=4,
    )
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    text_context = torch.randn(2, 16)
    slot_tokens = torch.randn(2, 5, 16)
    slot_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]])
    gate_prob, g, aux = gate(
        x_t=x_t,
        t=t,
        text_context=text_context,
        slot_tokens=slot_tokens,
        slot_mask=slot_mask,
        velocity_shape=(2, 12, 4, 3),
    )
    assert gate_prob.shape == (2, 3)
    assert g.shape == (2, 12, 4, 3)
    assert "operator_gate_attention_entropy_norm" in aux
    assert "operator_gate_attention_text_mass" in aux
    assert aux["operator_gate_attention_text_mass"].item() > 0.0
    assert torch.isfinite(gate_prob).all()


def test_global_operator_gate_without_velocity_shape():
    """GlobalOperatorGate works without expanding to velocity_shape."""
    gate = GlobalOperatorGate(d_model=16, num_channels=4, num_operators=3)
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    text_context = torch.randn(2, 16)
    gate_prob, g, aux = gate(x_t=x_t, t=t, text_context=text_context)
    assert gate_prob.shape == (2, 3)
    assert g is None


def test_text2ts_flow_with_regime_adapter_and_global_gate():
    """TextToTSFlow with V6.1 regime adapter + global operator gate."""
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=4,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
        use_latent_regime_adapter=True,
        num_regimes=4,
        router_mode="global_operator",
    )
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    out, aux = model(x_t, t, [["caption"], ["caption"]])
    assert out.shape == (2, 12, 4)
    assert torch.isfinite(out).all()
    assert "regime_prob" in aux
    assert "A_o" in aux
    assert "G" in aux
    assert "V" in aux
    # regime adapter appended a token, so encoder was called
    assert encoder.calls == 1


def test_build_model_passes_regime_posterior_mode():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "use_latent_regime_adapter": True,
            "num_regimes": 4,
            "regime_posterior_mode": "uniform",
            "router_mode": "global_operator",
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    assert model.regime_adapter is not None
    assert model.regime_adapter.posterior_mode == "uniform"


def test_build_model_passes_cross_modal_bridge_config():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "use_cross_modal_bridge": True,
            "bridge_num_heads": 2,
            "bridge_patch_size": 5,
            "bridge_num_spectral_tokens": 2,
            "bridge_focal_mode": "latent_query",
            "bridge_num_stage_tokens": 4,
            "operator_multiview_context": True,
            "router_mode": "global_operator",
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    assert model.cross_modal_bridge is not None
    assert model.cross_modal_bridge.patch_size == 5
    assert model.cross_modal_bridge.num_spectral_tokens == 2
    assert model.cross_modal_bridge.focal_mode == "latent_query"
    assert model.cross_modal_bridge.num_stage_tokens == 4
    assert model.operator_bank.multiview_context is True


def test_build_model_passes_patch_merger_bridge_config():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "operator_multiview_context": True,
            "router_mode": "global_operator",
            "use_cross_modal_bridge": True,
            "bridge_state_connector": "patch_merger",
            "bridge_temporal_merge": 3,
            "bridge_channel_merge": 1,
            "bridge_token_budget": 5,
            "bridge_alignment_mode": "siglip",
        }
    )
    model = build_model(cfg, sequence_length=13, num_channels=4)
    assert model.cross_modal_bridge is not None
    assert model.cross_modal_bridge.state_connector == "patch_merger"
    connector = model.cross_modal_bridge.patch_merger_connector
    assert connector is not None
    assert connector.temporal_merge == 3
    assert connector.channel_merge == 1
    assert connector.token_budget == 5
    assert connector.alignment_mode == "siglip"
    # P2: the patch-merger connector owns the forward, so the legacy bridge body
    # must NOT be built (no inert "built but bypassed" parameters).
    assert model.cross_modal_bridge._has_legacy_body is False
    assert not hasattr(model.cross_modal_bridge, "time_patch_proj")
    assert not hasattr(model.cross_modal_bridge, "state_to_context")


def test_legacy_bridge_still_builds_its_body():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "router_mode": "global_operator",
            "use_cross_modal_bridge": True,
            "bridge_state_connector": "legacy",
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    assert model.cross_modal_bridge._has_legacy_body is True
    assert hasattr(model.cross_modal_bridge, "time_patch_proj")
    assert model.cross_modal_bridge.patch_merger_connector is None


def test_budget_pool_active_flag_reports_correctly():
    """bridge_budget_pool_active must be 1.0 only when patch tokens exceed budget."""
    base = {
        "operator_architecture": "structural",
        "router_mode": "global_operator",
        "use_cross_modal_bridge": True,
        "bridge_state_connector": "patch_merger",
        "bridge_temporal_merge": 2,
        "bridge_channel_merge": 1,
    }
    # length 12, temporal_merge 2, channels 4 -> 6*4 = 24 patch tokens.
    fires_cfg = _text2ts_config()
    fires_cfg["model"].update({**base, "bridge_token_budget": 8})
    model = build_model(fires_cfg, sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=2, length=12, channels=4)
    out = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", task_mode="text2ts")
    assert out["bridge_patch_token_count"].item() == 24.0
    assert out["bridge_budget_pool_active"].item() == 1.0

    inert_cfg = _text2ts_config()
    inert_cfg["model"].update({**base, "bridge_token_budget": 96})
    model2 = build_model(inert_cfg, sequence_length=12, num_channels=4)
    out2 = cfm_train_step(model2, batch, optimizer=None, text_encoder_mode="hash", task_mode="text2ts")
    assert out2["bridge_budget_pool_active"].item() == 0.0


def test_text2ts_flow_with_cross_modal_bridge_and_global_gate():
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=4,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
        operator_architecture="structural",
        operator_multiview_context=True,
        router_mode="global_operator",
        use_cross_modal_bridge=True,
        bridge_focal_mode="latent_query",
        bridge_num_heads=4,
        bridge_patch_size=5,
    )
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    out, aux = model(x_t, t, [["caption"], ["caption"]])
    assert out.shape == (2, 12, 4)
    assert torch.isfinite(out).all()
    assert "bridge_alignment_loss" in aux
    assert "bridge_text_to_state_entropy" in aux
    assert "operator_aux" in aux
    assert "expert_context_norm" in aux["operator_aux"]
    assert "channel_context_norm" in aux["operator_aux"]
    assert "expert_time_context_norm" in aux["operator_aux"]
    assert "A_o" in aux


def test_text2ts_flow_with_latent_query_bridge_train_step():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "operator_multiview_context": True,
            "router_mode": "global_operator",
            "operator_gate_router": "attention",
            "use_cross_modal_bridge": True,
            "bridge_focal_mode": "latent_query",
            "bridge_num_stage_tokens": 3,
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=2, length=12, channels=4)
    result = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", task_mode="text2ts")
    assert torch.isfinite(result["loss"])
    assert "bridge_focal_channel_entropy_norm" in result
    assert "bridge_memory_token_count" in result
    assert "operator_gate_attention_entropy_norm" in result


def test_text2ts_flow_with_patch_merger_bridge_train_step():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "operator_multiview_context": True,
            "router_mode": "global_operator",
            "operator_gate_router": "attention",
            "use_cross_modal_bridge": True,
            "bridge_state_connector": "patch_merger",
            "bridge_temporal_merge": 3,
            "bridge_token_budget": 5,
            "bridge_alignment_mode": "siglip",
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=2, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        bridge_alignment_weight=1e-3,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["loss_bridge_alignment"])
    assert "bridge_connector_patch_merger" in result
    expected = result["loss_cfm"] + 1e-3 * result["loss_bridge_alignment"]
    assert torch.allclose(result["loss"], expected, atol=1e-6)


def test_text2ts_flow_regime_adapter_only():
    """TextToTSFlow with regime adapter but legacy mapper routing."""
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=4,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
        use_latent_regime_adapter=True,
        num_regimes=4,
        router_mode="legacy",
    )
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    out, aux = model(x_t, t, [["caption"], ["caption"]])
    assert out.shape == (2, 12, 4)
    assert torch.isfinite(out).all()
    assert "regime_prob" in aux
    assert "A_t" in aux  # legacy mapper still produces A_t
    assert "G" in aux


def test_v61_train_step_includes_regime_ortho_loss():
    """V6.1 train_step adds regime_ortho_weight * regime_ortho_loss."""
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=4,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
        use_latent_regime_adapter=True,
        num_regimes=4,
        router_mode="global_operator",
    )
    batch = _text_batch(batch_size=2, length=12, channels=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = cfm_train_step(
        model, batch, opt,
        text_encoder_mode="hash",
        task_mode="text2ts",
        regime_ortho_weight=1e-4,
    )
    assert torch.isfinite(result["loss"])
    assert "loss_regime_ortho" in result
    assert "regime_entropy" in result
    assert "operator_gate_entropy" in result


def test_v61_train_step_regime_ortho_zero_weight():
    """With regime_ortho_weight=0, no ortho loss is added."""
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=4,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
        use_latent_regime_adapter=True,
        num_regimes=4,
        router_mode="global_operator",
    )
    batch = _text_batch(batch_size=2, length=12, channels=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = cfm_train_step(
        model, batch, opt,
        text_encoder_mode="hash",
        task_mode="text2ts",
        regime_ortho_weight=0.0,
    )
    assert result["loss_regime_ortho"].item() == 0.0


def test_v62_train_step_includes_bridge_alignment_loss():
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=4,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
        operator_architecture="structural",
        router_mode="global_operator",
        use_cross_modal_bridge=True,
        bridge_num_heads=4,
    )
    batch = _text_batch(batch_size=2, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        bridge_alignment_weight=1e-3,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["loss_bridge_alignment"])
    assert result["loss_bridge_alignment"].item() >= 0.0
    assert "bridge_alignment_loss" in result
    expected = result["loss_cfm"] + 1e-3 * result["loss_bridge_alignment"]
    assert torch.allclose(result["loss"], expected, atol=1e-6)


def test_text2ts_caption_negative_ranking_loss_is_optional_and_finite():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "router_mode": "global_operator",
            "use_cross_modal_bridge": True,
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=3, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        caption_ranking_weight=1e-3,
        caption_ranking_margin=0.05,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["loss_caption_ranking"])
    assert torch.isfinite(result["caption_neg_mse"])
    assert 0.0 <= result["caption_ranking_acc"].item() <= 1.0
    expected = result["loss_cfm"] + 1e-3 * result["loss_caption_ranking"]
    assert torch.allclose(result["loss"], expected, atol=1e-6)


def test_text2ts_caption_negative_ranking_batch_one_is_zero():
    model = build_model(_text2ts_config(), sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=1, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        caption_ranking_weight=1e-3,
    )
    assert result["loss_caption_ranking"].item() == 0.0
    assert torch.allclose(result["loss"], result["loss_cfm"])


def test_v62_candidates_slots_reach_bridge_diagnostics():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "router_mode": "global_operator",
            "use_cross_modal_bridge": True,
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=2, length=12, channels=4)
    batch["caption_candidates"] = [
        [f"weather caption {idx} candidate {slot}" for slot in range(6)]
        for idx in range(2)
    ]
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        caption_slot_strategy="candidates",
        max_caption_slots=4,
        include_all_caption_candidates=True,
    )
    assert torch.isfinite(result["loss"])
    assert torch.allclose(result["text_slot_count"], torch.tensor(4.0))
    assert result["text_slot_count_min"].item() == 4.0
    assert result["text_slot_count_max"].item() == 4.0
    assert result["bridge_state_to_text_entropy"].item() > 0.0


def test_v61_uniform_regime_train_step_exposes_uniform_diagnostics():
    encoder = CountingTextEncoder(d_model=16)
    model = TextToTSFlow(
        sequence_length=12,
        num_channels=4,
        patch_len=3,
        d_model=16,
        num_operators=3,
        text_encoder=encoder,
        transformer_layers=1,
        transformer_heads=4,
        operator_hidden=8,
        operator_t_dim=4,
        use_latent_regime_adapter=True,
        num_regimes=4,
        regime_posterior_mode="uniform",
        router_mode="global_operator",
    )
    batch = _text_batch(batch_size=2, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        regime_ortho_weight=1e-4,
    )
    assert torch.isfinite(result["loss"])
    assert torch.allclose(result["aux"]["regime_prob"], torch.full_like(result["aux"]["regime_prob"], 0.25))
    assert torch.allclose(result["regime_usage"], torch.full_like(result["regime_usage"], 0.25))
    assert torch.allclose(result["regime_entropy_norm"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(result["regime_max_prob"], torch.tensor(0.25))


def test_v61_backward_compat_no_regime_no_global_gate():
    """Default config (no regime, legacy router) still works."""
    cfg = _text2ts_config()
    model = build_model(cfg, sequence_length=12, num_channels=4)
    assert model.regime_adapter is None
    assert model.global_operator_gate is None
    batch = _text_batch()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = cfm_train_step(model, batch, opt, text_encoder_mode="hash", task_mode="text2ts")
    assert torch.isfinite(result["loss"])
    assert result["loss_regime_ortho"].item() == 0.0


def test_spectral_loss_disabled_by_default_is_zero_and_noop():
    model = build_model(_text2ts_config(), sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=2, length=12, channels=4)
    result = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", task_mode="text2ts")
    assert result["loss_spectral"].item() == 0.0
    assert torch.allclose(result["loss"], result["loss_cfm"])


def test_spectral_loss_text2ts_is_finite_and_added():
    model = build_model(_text2ts_config(), sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=2, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        spectral_loss_weight=0.1,
    )
    assert torch.isfinite(result["loss_spectral"])
    assert result["loss_spectral"].item() > 0.0
    expected = result["loss_cfm"] + 0.1 * result["loss_spectral"]
    assert torch.allclose(result["loss"], expected, atol=1e-6)


def test_spectral_loss_edit_mode_respects_mask():
    model = build_model(_config(), sequence_length=12, num_channels=4)
    batch = _batch(batch_size=2, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="edit",
        spectral_loss_weight=0.1,
    )
    assert torch.isfinite(result["loss_spectral"])
    expected = result["loss_cfm"] + 0.1 * result["loss_spectral"]
    assert torch.allclose(result["loss"], expected, atol=1e-6)


def test_spectral_loss_helper_zero_on_perfect_prediction():
    from effectcma_flow.training.train_step import _spectral_magnitude_loss

    target = torch.randn(3, 16, 4)
    loss = _spectral_magnitude_loss(target.clone(), target, mask=None, log_magnitude=True)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_spectral_loss_helper_x0_recovery_identity():
    """x_hat0 = x_t + (1 - t) * target_v reconstructs the target exactly."""
    from effectcma_flow.training.train_step import _spectral_magnitude_loss

    source = torch.randn(2, 16, 3)
    target = torch.randn(2, 16, 3)
    t = torch.rand(2)
    x_t = (1.0 - t[:, None, None]) * source + t[:, None, None] * target
    target_v = target - source
    pred_x0 = x_t + (1.0 - t[:, None, None]) * target_v
    assert torch.allclose(pred_x0, target, atol=1e-5)
    loss = _spectral_magnitude_loss(pred_x0, target, mask=None, log_magnitude=False)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_operator_balance_loss_is_optional_finite_and_added():
    cfg = _text2ts_config()
    cfg["model"].update({"operator_architecture": "structural", "router_mode": "global_operator"})
    model = build_model(cfg, sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=4, length=12, channels=4)
    result = cfm_train_step(
        model,
        batch,
        optimizer=None,
        text_encoder_mode="hash",
        task_mode="text2ts",
        operator_balance_weight=0.01,
    )
    assert torch.isfinite(result["loss_operator_balance"])
    # KL(usage || uniform) >= 0 always.
    assert result["loss_operator_balance"].item() >= -1e-6
    expected = result["loss_cfm"] + 0.01 * result["loss_operator_balance"]
    assert torch.allclose(result["loss"], expected, atol=1e-6)


def test_operator_balance_loss_zero_when_disabled():
    cfg = _text2ts_config()
    cfg["model"].update({"operator_architecture": "structural", "router_mode": "global_operator"})
    model = build_model(cfg, sequence_length=12, num_channels=4)
    batch = _text_batch(batch_size=4, length=12, channels=4)
    result = cfm_train_step(model, batch, optimizer=None, text_encoder_mode="hash", task_mode="text2ts")
    assert result["loss_operator_balance"].item() == 0.0


def test_operator_balance_loss_grows_under_collapse():
    """KL term must be ~0 for a uniform gate and large for a collapsed gate."""
    import torch.nn.functional as F

    from effectcma_flow.models.global_operator_gate import GlobalOperatorGate

    gate = GlobalOperatorGate(d_model=16, num_channels=4, num_operators=3)

    def balance(g):
        usage = g.mean(dim=0)
        uniform = g.new_full((g.shape[-1],), 1.0 / g.shape[-1])
        return (usage.clamp_min(1e-8) * (usage.clamp_min(1e-8) / uniform).log()).sum()

    uniform_gate = torch.full((8, 3), 1.0 / 3.0)
    collapsed_gate = F.one_hot(torch.zeros(8, dtype=torch.long), num_classes=3).float()
    assert balance(uniform_gate).item() == pytest.approx(0.0, abs=1e-6)
    assert balance(collapsed_gate).item() > 1.0


def test_dense_clean_alignment_builds_and_trains():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "router_mode": "global_operator",
            "use_cross_modal_bridge": True,
            "bridge_state_connector": "patch_merger",
            "bridge_alignment_mode": "siglip",
            "bridge_alignment_dense": True,
            "bridge_alignment_regions": 6,
            "bridge_alignment_target": "clean",
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    connector = model.cross_modal_bridge.patch_merger_connector
    assert connector.alignment_dense is True
    assert connector.alignment_target == "clean"
    assert connector.text_region_pool.num_regions == 6
    batch = _text_batch(batch_size=4, length=12, channels=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = cfm_train_step(
        model,
        batch,
        opt,
        text_encoder_mode="hash",
        task_mode="text2ts",
        bridge_alignment_weight=0.01,
    )
    assert torch.isfinite(result["loss"])
    assert result["loss_bridge_alignment"].item() != 0.0
    assert float(result["aux"]["bridge_alignment_dense"]) == 1.0
    # Region pools must actually receive gradient (dense path is wired, not dead).
    for pool in (connector.text_region_pool, connector.state_region_pool):
        assert pool.queries.grad is not None
        assert pool.queries.grad.abs().sum().item() > 0.0


def test_dense_alignment_falls_back_to_state_without_target():
    """At sampling time (no target), clean-target dense alignment must not crash."""
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "router_mode": "global_operator",
            "use_cross_modal_bridge": True,
            "bridge_state_connector": "patch_merger",
            "bridge_alignment_dense": True,
            "bridge_alignment_target": "clean",
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    x_t = torch.randn(2, 12, 4)
    t = torch.rand(2)
    # No target kwarg -> dense alignment should pool the fused state instead.
    pred, aux = model(x_t, t, [["sunny dry day"], ["cold front incoming"]])
    assert pred.shape == x_t.shape
    assert torch.isfinite(aux["bridge_alignment_loss"]).all()


def test_spectral_loss_gradient_reaches_frequency_expert():
    cfg = _text2ts_config()
    cfg["model"].update(
        {
            "operator_architecture": "structural",
            "router_mode": "global_operator",
        }
    )
    model = build_model(cfg, sequence_length=12, num_channels=4)
    freq_expert = model.operator_bank.experts[2]  # FrequencyBandExpert
    assert type(freq_expert).__name__ == "FrequencyBandExpert"
    batch = _text_batch(batch_size=2, length=12, channels=4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    result = cfm_train_step(
        model,
        batch,
        opt,
        text_encoder_mode="hash",
        task_mode="text2ts",
        spectral_loss_weight=1.0,
    )
    assert torch.isfinite(result["loss"])
    grads = [p.grad for p in freq_expert.parameters() if p.grad is not None]
    assert grads, "frequency expert received no gradient"
    assert any(g.abs().sum().item() > 0 for g in grads)
