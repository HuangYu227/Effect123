from __future__ import annotations

import pytest
import torch

from effectcma_flow.data.effects import EffectSpec, apply_effect


@pytest.mark.parametrize("effect_type", ["level_shift", "trend_up", "trend_down", "volatility_up", "volatility_down", "spike", "drop"])
def test_effects_are_scoped(effect_type):
    base = torch.randn(12, 4)
    strength = 1.5 if effect_type == "volatility_up" else 0.5
    spec = EffectSpec(effect_type=effect_type, start=3, end=8, channels=(1, 3), strength=strength)
    target, mask = apply_effect(base, spec)
    residual = target - base
    assert mask.sum().item() == 10
    assert torch.allclose(residual[mask == 0], torch.zeros_like(residual[mask == 0]), atol=1e-6)
    assert torch.isfinite(target).all()


def test_invalid_spec_rejected():
    base = torch.zeros(12, 4)
    with pytest.raises(ValueError):
        apply_effect(base, EffectSpec("spike", start=8, end=3, channels=(1,), strength=1.0))
    with pytest.raises(ValueError):
        apply_effect(base, EffectSpec("spike", start=1, end=3, channels=(9,), strength=1.0))

