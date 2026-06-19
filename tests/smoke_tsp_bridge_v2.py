from __future__ import annotations

import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

torch.set_num_threads(1)

from effectcma_flow.models.temporal_pyramid_bridge_v2 import TemporalSemanticPyramidConnectorV2


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise AssertionError(f"{name} contains non-finite values")


def run_case(batch: int, length: int, channels: int, d_model: int, patch_lens: tuple[int, ...], budget: int) -> None:
    torch.manual_seed(7)
    module = TemporalSemanticPyramidConnectorV2(
        sequence_length=length,
        num_channels=channels,
        d_model=d_model,
        patch_lens=patch_lens,
        token_budget=budget,
        anchor_tokens=2,
        cross_scale_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    x_t = torch.randn(batch, length, channels)
    t = torch.rand(batch)
    slot_tokens = torch.randn(batch, 4, d_model)
    slot_mask = torch.ones(batch, 4)
    slot_mask[0, -1] = 0.0
    state_tokens, aux = module(x_t=x_t, t=t, slot_tokens=slot_tokens, slot_mask=slot_mask)
    assert state_tokens.shape == (batch, budget, d_model), state_tokens.shape
    assert_finite("state_tokens", state_tokens)
    required = [
        "tsp_connector_active",
        "tsp_raw_token_count",
        "tsp_budget_token_count",
        "tsp_anchor_token_count",
        "tsp_scale_gate_entropy_norm",
        "tsp_global_gate_s0",
        "tsp_local_gate_s0",
    ]
    missing = [k for k in required if k not in aux]
    if missing:
        raise AssertionError(f"missing aux keys: {missing}")
    loss = state_tokens.square().mean()
    loss.backward()
    total_grad = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total_grad += float(p.grad.detach().abs().sum())
    if not math.isfinite(total_grad) or total_grad <= 0.0:
        raise AssertionError(f"bad total_grad={total_grad}")
    print(f"ok: B={batch}, L={length}, C={channels}, D={d_model}, patch={patch_lens}, budget={budget}, raw={aux['tsp_raw_token_count'].item():.0f}")


def main() -> None:
    # Fast CPU smoke cases. Full training uses L=128 and larger D on GPU.
    run_case(batch=1, length=64, channels=2, d_model=32, patch_lens=(4, 8, 16), budget=24)
    run_case(batch=1, length=64, channels=4, d_model=32, patch_lens=(4, 12, 32), budget=24)
    print("TSP-Bridge++ smoke test passed.")


if __name__ == "__main__":
    main()
