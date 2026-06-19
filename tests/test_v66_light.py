from __future__ import annotations

import math

import torch

from effectcma_flow.models import SpectralPromptGenerator, build_model
from effectcma_flow.models.spectral_prompt import merge_spectral_expert_context
from effectcma_flow.training import cfm_train_step, normalized_multi_resolution_fft_loss


def test_spectral_prompt_shapes_and_zero_init_delta():
    torch.manual_seed(0)
    generator = SpectralPromptGenerator(
        d_model=16,
        num_bands=3,
        num_experts=3,
        num_heads=4,
        gate_temperature=0.7,
    )
    slot_tokens = torch.randn(2, 5, 16)
    slot_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    out = generator(slot_tokens, slot_mask)

    assert out.band_tokens.shape == (2, 3, 16)
    assert out.band_gates.shape == (2, 3)
    assert out.expert_delta.shape == (2, 3, 16)
    assert torch.allclose(out.band_gates.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(out.band_tokens).all()
    assert torch.isfinite(out.expert_delta).all()
    assert torch.allclose(out.expert_delta, torch.zeros_like(out.expert_delta), atol=1e-7)
    assert "spectral_prompt_entropy_norm" in out.aux

    expert_context = torch.randn(2, 3, 16)
    merged = merge_spectral_expert_context(expert_context, out.expert_delta)
    assert torch.allclose(merged, expert_context, atol=1e-7)


def test_normalized_mrfft_loss_mask_and_reduction():
    torch.manual_seed(1)
    target = torch.randn(2, 17, 3)
    pred = target + 0.05 * torch.randn_like(target)
    mask = torch.ones(2, 17)
    mask[0, -3:] = 0

    per_sample = normalized_multi_resolution_fft_loss(
        pred,
        target,
        fft_sizes=(8, 16, 32),
        mask=mask,
        reduction="none",
    )
    scalar = normalized_multi_resolution_fft_loss(
        pred,
        target,
        fft_sizes=(8, 16, 32),
        mask=mask,
        reduction="mean",
    )
    zero = normalized_multi_resolution_fft_loss(
        target,
        target,
        fft_sizes=(8, 16, 32),
        mask=mask,
    )

    assert per_sample.shape == (2,)
    assert torch.isfinite(per_sample).all()
    assert torch.isfinite(scalar)
    assert torch.allclose(scalar, per_sample.mean(), atol=1e-7)
    assert float(zero) < 1e-7


def test_v66_light_text2ts_path_trains_with_spectral_prompt():
    torch.manual_seed(2)
    cfg = _v66_light_test_config()
    model = build_model(cfg, sequence_length=16, num_channels=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "Y": torch.randn(2, 16, 3),
        "caption": [
            "steady weather trend with a mild daily oscillation",
            "sharp high frequency change followed by a slow drift",
        ],
    }

    t = torch.rand(2)
    x_t = torch.randn(2, 16, 3)
    pred_v, aux = model(x_t, t, [[text] for text in batch["caption"]], target=batch["Y"])
    assert pred_v.shape == (2, 16, 3)
    assert torch.isfinite(pred_v).all()
    assert "spectral_prompt_gate_low" in aux
    assert math.isfinite(float(aux["spectral_prompt_entropy_norm"].detach()))

    result = cfm_train_step(
        model,
        batch,
        opt,
        text_encoder_mode="hash",
        task_mode="text2ts",
        spectral_loss_weight=0.03,
        spectral_loss_type="normalized_mrfft",
        spectral_fft_sizes=(8, 16, 32),
        spectral_distance="l1",
        bridge_alignment_weight=0.01,
        operator_balance_weight=0.01,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["loss_spectral"])
    assert torch.isfinite(result["spectral_prompt_entropy_norm"])
    assert torch.isfinite(result["spectral_prompt_gate_low"])


def test_tsp_bridge_clean_alignment_and_optional_regularizers():
    torch.manual_seed(3)
    cfg = _tsp_v2_test_config()
    model = build_model(cfg, sequence_length=16, num_channels=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = {
        "Y": torch.randn(2, 16, 3),
        "caption": [
            "daily periodic weather with a local abrupt change",
            "slow seasonal drift mixed with high frequency fluctuation",
        ],
    }

    t = torch.rand(2)
    x_t = torch.randn(2, 16, 3)
    pred_v, aux = model(x_t, t, [[text] for text in batch["caption"]], target=batch["Y"])
    assert pred_v.shape == (2, 16, 3)
    assert torch.isfinite(pred_v).all()
    assert int(aux["tsp_connector_active"].item()) == 1
    assert int(aux["tsp_budget_token_count"].item()) == 16
    assert int(aux["bridge_alignment_tsp_clean_encoded"].item()) == 1
    assert torch.isfinite(aux["tsp_scale_entropy_loss"])
    assert torch.isfinite(aux["tsp_scale_balance_loss"])

    result = cfm_train_step(
        model,
        batch,
        opt,
        text_encoder_mode="hash",
        task_mode="text2ts",
        bridge_alignment_weight=0.01,
        tsp_scale_entropy_weight=0.001,
        tsp_scale_balance_weight=0.001,
    )
    assert torch.isfinite(result["loss"])
    assert torch.isfinite(result["loss_tsp_scale_entropy"])
    assert torch.isfinite(result["loss_tsp_scale_balance"])
    assert int(result["bridge_alignment_tsp_clean_encoded"].item()) == 1


def _v66_light_test_config() -> dict:
    return {
        "task": {"mode": "text2ts"},
        "model": {
            "d_model": 16,
            "patch_len": 4,
            "num_operators": 3,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "operator_architecture": "structural",
            "operator_hidden": 12,
            "operator_t_dim": 4,
            "operator_depth": 2,
            "operator_dropout": 0.0,
            "operator_context_film": True,
            "operator_multiview_context": True,
            "operator_channel_heads": 1,
            "operator_frequency_band_mode": "gaussian",
            "router_mode": "global_operator",
            "operator_gate_router": "attention",
            "operator_gate_heads": 4,
            "use_cross_modal_bridge": True,
            "bridge_num_heads": 4,
            "bridge_patch_size": 4,
            "bridge_state_connector": "patch_merger",
            "bridge_temporal_merge": 2,
            "bridge_channel_merge": 1,
            "bridge_token_budget": 24,
            "bridge_alignment_mode": "siglip",
            "bridge_alignment_dense": True,
            "bridge_alignment_target": "clean",
            "bridge_text_agg_tokens": 2,
            "use_spectral_prompt": True,
            "spectral_prompt_bands": 3,
            "spectral_prompt_heads": 4,
            "spectral_prompt_dropout": 0.0,
            "spectral_prompt_gate_temperature": 0.7,
        },
        "text_encoder": {
            "mode": "hash",
            "hash_dim": 32,
        },
    }


def _tsp_v2_test_config() -> dict:
    cfg = _v66_light_test_config()
    cfg["model"].update(
        {
            "bridge_state_connector": "temporal_pyramid_v2",
            "bridge_token_budget": 16,
            "temporal_pyramid_patch_lens": [4, 8],
            "temporal_pyramid_token_budget": 16,
            "temporal_pyramid_anchor_tokens": 2,
            "temporal_pyramid_cross_scale_layers": 1,
            "temporal_pyramid_dropout": 0.0,
            "temporal_pyramid_use_topdown": True,
            "temporal_pyramid_use_bottomup": True,
            "temporal_pyramid_use_text_routing": True,
            "temporal_pyramid_temporal_bias_tau": 0.25,
            "temporal_pyramid_gate_temperature": 0.7,
        }
    )
    return cfg
