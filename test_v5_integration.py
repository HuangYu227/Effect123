from __future__ import annotations

import sys
import traceback
from typing import Any

import torch

from effectcma_flow.models.build import build_model
from effectcma_flow.training.train_step import cfm_train_step
from effectcma_flow.evaluation.sampler import sample_text2ts
from effectcma_flow.training.diagnostics import (
    print_core_flow_report,
    gate_health_status,
    assert_text_encoder_is_semantic,
)

SEQ_LEN = 48
NUM_CHANNELS = 3
PATCH_LEN = 6
D_MODEL = 32

PASS_COUNT = 0
FAIL_COUNT = 0


def _report(name, passed, detail=""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    suffix = " -- " + detail if detail else ""
    print("  [" + tag + "] " + name + suffix)


def cfg_v1_baseline():
    return {
        "task": {"mode": "text2ts"},
        "model": {
            "d_model": D_MODEL,
            "patch_len": PATCH_LEN,
            "num_operators": 7,
            "transformer_layers": 2,
            "transformer_heads": 4,
            "operator_hidden": 64,
            "operator_t_dim": 16,
            "operator_depth": 2,
            "operator_kernel_size": 3,
            "operator_architecture": "homogeneous",
            "mapper_operator_router": "text",
            "mapper_field_rank": 4,
            "mapper_slot_layers": 1,
            "mapper_bounded_field_gate": True,
            "mapper_gate_rescale": "auto",
            "mapper_flow_time_condition": True,
        },
        "text_encoder": {"mode": "hash", "hash_dim": 64},
    }


def cfg_v5_full():
    return {
        "task": {"mode": "text2ts"},
        "model": {
            "d_model": D_MODEL,
            "patch_len": PATCH_LEN,
            "num_operators": 3,
            "transformer_layers": 2,
            "transformer_heads": 4,
            "operator_hidden": 64,
            "operator_t_dim": 16,
            "operator_depth": 2,
            "operator_kernel_size": 3,
            "operator_architecture": "structural",
            "operator_channel_heads": 4,
            "mapper_operator_router": "dual",
            "mapper_field_rank": 4,
            "mapper_slot_layers": 1,
            "mapper_bounded_field_gate": True,
            "mapper_gate_rescale": "auto",
            "mapper_flow_time_condition": True,
        },
        "text_encoder": {"mode": "hash", "hash_dim": 64},
    }


def mock_batch(batch_size=2, length=SEQ_LEN, channels=NUM_CHANNELS):
    return {
        "Y": torch.randn(batch_size, length, channels),
        "caption": ["sunny weather", "rainy day"][:batch_size],
    }


def run_pipeline(label, config, batch):
    sep = "=" * 60
    print()
    print(sep)
    print("  Pipeline: " + label)
    print(sep)
    seq_len = batch["Y"].shape[1]
    num_ch = batch["Y"].shape[2]

    # 1. build_model
    try:
        model = build_model(config, sequence_length=seq_len, num_channels=num_ch)
        model.train()
        _report("build_model", True)
    except Exception as exc:
        _report("build_model", False, str(exc))
        traceback.print_exc()
        return

    # 2. cfm_train_step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    try:
        result = cfm_train_step(
            model, batch, opt,
            text_encoder_mode="hash", task_mode="text2ts",
        )
        loss = result["loss"]
        loss_finite = bool(torch.isfinite(loss).item())
        _report("cfm_train_step loss finite", loss_finite, "loss=%.6f" % loss.item())
    except Exception as exc:
        _report("cfm_train_step", False, str(exc))
        traceback.print_exc()
        return

    # 3. gradients exist
    try:
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters() if p.requires_grad
        )
        _report("loss has gradients", has_grad)
    except Exception as exc:
        _report("loss has gradients", False, str(exc))

    # 4. aux dict keys
    aux = result.get("aux", {})
    for key in ["A_o", "A_t", "A_c", "gate_rescale_factor", "gate_cell_mass_mean"]:
        _report("aux[" + repr(key) + "] present", key in aux)

    # 5. structural-specific aux
    is_structural = config.get("model", {}).get("operator_architecture") == "structural"
    if is_structural:
        op_aux = aux.get("operator_aux", {})
        has_op_aux = len(op_aux) > 0
        _report("structural operator_aux non-empty", has_op_aux, "keys=" + str(list(op_aux.keys())[:6]))
        for prefix in ["time_", "channel_", "frequency_"]:
            found = any(k.startswith(prefix) for k in op_aux)
            _report("operator_aux has " + repr(prefix + "*") + " keys", found)
    else:
        oa = aux.get("operator_aux", {})
        _report("homogeneous: no operator_aux expected", len(oa) == 0)

    # 6. sample_text2ts with all 3 solvers
    model.eval()
    shape_like = batch["Y"]
    text_cond = [[str(c)] for c in batch["caption"]]
    for solver in ("euler", "midpoint", "rk4"):
        try:
            with torch.no_grad():
                x_out, sample_aux = sample_text2ts(
                    model, shape_like, text_cond,
                    solver=solver, steps=4,
                )
            shape_ok = x_out.shape == shape_like.shape
            finite = bool(torch.isfinite(x_out).all().item())
            solver_tag = sample_aux.get("sampler_solver")
            _report("sample_text2ts(" + solver + ") shape", shape_ok, "got " + str(tuple(x_out.shape)))
            _report("sample_text2ts(" + solver + ") finite", finite)
            _report("sample_text2ts(" + solver + ") solver_tag", solver_tag == solver, "tag=" + str(solver_tag))
        except Exception as exc:
            _report("sample_text2ts(" + solver + ")", False, str(exc))
            traceback.print_exc()

    # 7. diagnostics
    try:
        pred_rms = result["pred_v"].square().mean().sqrt()
        tgt_rms = result["target_v"].square().mean().sqrt().clamp_min(1e-8)
        logs = {
            "loss": result["loss"],
            "gate_rescale_factor": aux.get("gate_rescale_factor"),
            "gate_raw_cell_mass_mean": aux.get("gate_raw_cell_mass_mean"),
            "gate_cell_mass_mean": aux.get("gate_cell_mass_mean"),
            "operator_gate_entropy": result.get("operator_gate_entropy"),
            "time_gate_entropy": result.get("time_gate_entropy"),
            "channel_gate_entropy": result.get("channel_gate_entropy"),
            "pred_target_rms_ratio": (pred_rms / tgt_rms).item(),
        }
        print("  -- print_core_flow_report --")
        print_core_flow_report(logs)
        _report("print_core_flow_report", True)
    except Exception as exc:
        _report("print_core_flow_report", False, str(exc))

    try:
        status = gate_health_status(logs)
        _report("gate_health_status", True, status)
    except Exception as exc:
        _report("gate_health_status", False, str(exc))

    # assert_text_encoder_is_semantic should RAISE for hash encoder
    try:
        assert_text_encoder_is_semantic(model)
        _report("assert_text_encoder_is_semantic raises for hash", False, "did not raise")
    except RuntimeError:
        _report("assert_text_encoder_is_semantic raises for hash", True)



def run_edge_cases():
    sep = "=" * 60
    print()
    print(sep)
    print("  Edge cases")
    print(sep)

    # Edge 1: batch=1, length=12, channels=1, homogeneous
    config = cfg_v1_baseline()
    batch = {"Y": torch.randn(1, 12, 1), "caption": ["foggy morning"]}
    try:
        model = build_model(config, sequence_length=12, num_channels=1)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        result = cfm_train_step(model, batch, opt, text_encoder_mode="hash", task_mode="text2ts")
        ok = bool(torch.isfinite(result["loss"]).item())
        _report("edge batch=1,len=12,ch=1 homogeneous", ok, "loss=%.6f" % result["loss"].item())
    except Exception as exc:
        _report("edge batch=1,len=12,ch=1 homogeneous", False, str(exc))
        traceback.print_exc()

    # Edge 2: batch=1, length=12, channels=1, structural
    config2 = cfg_v5_full()
    batch2 = {"Y": torch.randn(1, 12, 1), "caption": ["foggy morning"]}
    try:
        model2 = build_model(config2, sequence_length=12, num_channels=1)
        model2.train()
        opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        result2 = cfm_train_step(model2, batch2, opt2, text_encoder_mode="hash", task_mode="text2ts")
        ok2 = bool(torch.isfinite(result2["loss"]).item())
        _report("edge batch=1,len=12,ch=1 structural", ok2, "loss=%.6f" % result2["loss"].item())
    except Exception as exc:
        _report("edge batch=1,len=12,ch=1 structural", False, str(exc))
        traceback.print_exc()

    # Edge 3: single-caption sample
    config3 = cfg_v1_baseline()
    batch3 = {"Y": torch.randn(1, 48, 3), "caption": ["only one caption"]}
    try:
        model3 = build_model(config3, sequence_length=48, num_channels=3)
        model3.eval()
        with torch.no_grad():
            x_out, _ = sample_text2ts(
                model3, batch3["Y"],
                [[str(c)] for c in batch3["caption"]],
                solver="euler", steps=2,
            )
        _report("edge single-caption sample", x_out.shape == (1, 48, 3), "shape=" + str(tuple(x_out.shape)))
    except Exception as exc:
        _report("edge single-caption sample", False, str(exc))
        traceback.print_exc()

    # Edge 4: return_trajectory
    try:
        model4 = build_model(cfg_v1_baseline(), sequence_length=48, num_channels=3)
        model4.eval()
        with torch.no_grad():
            _, traj_aux = sample_text2ts(
                model4, torch.randn(2, 48, 3),
                [["sunny"], ["rainy"]],
                solver="rk4", steps=3, return_trajectory=True,
            )
        has_traj = "trajectory" in traj_aux
        traj_len = len(traj_aux.get("trajectory", []))
        _report("return_trajectory=True", has_traj and traj_len == 3, "traj_len=%d" % traj_len)
    except Exception as exc:
        _report("return_trajectory=True", False, str(exc))
        traceback.print_exc()



def main():
    sep = "=" * 60
    print(sep)
    print("  EffectCMA v5 Integration Test")
    print(sep)

    run_pipeline("v1 baseline (homogeneous / router=text)", cfg_v1_baseline(), mock_batch())
    run_pipeline("v5 full (structural / router=dual)", cfg_v5_full(), mock_batch())
    run_edge_cases()

    total = PASS_COUNT + FAIL_COUNT
    print()
    print(sep)
    print("  SUMMARY: %d/%d passed, %d/%d failed" % (PASS_COUNT, total, FAIL_COUNT, total))
    print(sep)

    if FAIL_COUNT > 0:
        print()
        print("RESULT: FAIL")
        sys.exit(1)
    else:
        print()
        print("RESULT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
