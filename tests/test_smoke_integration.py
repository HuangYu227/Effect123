"""Smoke integration test: operator bank + routing losses with synthetic data.

Verifies that the structural operator bank and routing regularization
work end-to-end without runtime errors.
"""
from __future__ import annotations

import torch

from effectcma_flow.models.operator_bank import ResidualOperatorBank
from effectcma_flow.training.routing_losses import (
    RoutingLossTargets,
    RoutingLossWeights,
    compute_routing_regularization,
)


def test_structural_operator_bank_forward_and_routing_losses():
    """Structural operator bank produces output and aux; routing losses consume aux."""
    bank = ResidualOperatorBank(
        num_channels=3,
        num_operators=3,
        hidden=32,
        t_dim=16,
        architecture="structural",
        channel_heads=2,
    )
    x_t = torch.randn(2, 22, 3)
    base = torch.randn(2, 22, 3)
    t = torch.rand(2)
    out = bank(x_t, base, t)
    assert out.shape == (2, 22, 3, 3), f"Expected (2,22,3,3), got {out.shape}"
    assert torch.isfinite(out).all(), "Operator bank output contains non-finite values"
    aux_keys = list(bank.last_aux.keys())
    assert len(aux_keys) > 0, "Operator bank produced no aux keys"


def test_routing_losses_with_full_aux():
    """Routing losses compute correctly given all expected aux keys."""
    aux = {
        "A_o": torch.softmax(torch.randn(2, 22, 3), dim=-1),
        "A_t": torch.softmax(torch.randn(2, 22), dim=-1),
        "A_c": torch.softmax(torch.randn(2, 3), dim=-1),
        "G": torch.randn(2, 22, 3),
    }
    loss, parts = compute_routing_regularization(aux)
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.item() >= 0.0, f"Routing loss should be non-negative, got {loss.item()}"
    expected_keys = {
        "router_op_entropy",
        "router_op_entropy_norm",
        "router_op_confidence",
        "router_op_balance",
        "router_time_entropy",
        "router_time_entropy_norm",
        "router_time_confidence",
        "router_channel_entropy",
        "router_channel_entropy_norm",
        "router_channel_confidence",
        "router_field_mass",
        "router_loss",
    }
    assert expected_keys.issubset(set(parts.keys())), (
        f"Missing loss part keys: {expected_keys - set(parts.keys())}"
    )


def test_routing_losses_with_empty_aux():
    """Routing losses gracefully handle empty aux dict (returns zero loss)."""
    loss, parts = compute_routing_regularization({})
    assert loss.ndim == 0
    assert loss.item() == 0.0, f"Empty aux loss should be 0, got {loss.item()}"
    assert "router_loss" in parts


def test_routing_losses_backward():
    """Routing losses produce gradients that flow back to aux tensors."""
    # softmax produces non-leaf tensors; use retain_grad so .grad is populated
    a_o_raw = torch.randn(2, 22, 3, requires_grad=True)
    a_t_raw = torch.randn(2, 22, requires_grad=True)
    a_c_raw = torch.randn(2, 3, requires_grad=True)
    a_o = torch.softmax(a_o_raw, dim=-1)
    a_t = torch.softmax(a_t_raw, dim=-1)
    a_c = torch.softmax(a_c_raw, dim=-1)
    a_o.retain_grad()
    a_t.retain_grad()
    a_c.retain_grad()
    g = torch.randn(2, 22, 3, requires_grad=True)
    aux = {"A_o": a_o, "A_t": a_t, "A_c": a_c, "G": g}
    # Use a non-None field_mass target so G participates in the loss
    targets = RoutingLossTargets(field_mass=1.0)
    loss, _ = compute_routing_regularization(aux, targets=targets)
    loss.backward()
    assert a_o.grad is not None, "A_o has no gradient"
    assert a_t.grad is not None, "A_t has no gradient"
    assert a_c.grad is not None, "A_c has no gradient"
    assert g.grad is not None, "G has no gradient"


def test_operator_bank_to_routing_loss_end_to_end():
    """Operator bank output feeds into routing losses without shape errors."""
    bank = ResidualOperatorBank(
        num_channels=3,
        num_operators=3,
        hidden=16,
        t_dim=8,
        architecture="structural",
        channel_heads=2,
    )
    x_t = torch.randn(2, 16, 3)
    base = torch.randn(2, 16, 3)
    t = torch.rand(2)
    out = bank(x_t, base, t)
    # Construct synthetic routing aux from bank output shape
    # A_o: (batch, time, channels, num_operators) -> take mean over time/channels -> (batch, num_operators)
    a_o = torch.softmax(out.mean(dim=(1, 2)), dim=-1)
    # A_t: (batch, time) -> softmax over time
    a_t = torch.softmax(torch.randn(2, 16), dim=-1)
    # A_c: (batch, channels) -> softmax over channels
    a_c = torch.softmax(torch.randn(2, 3), dim=-1)
    # G: same shape as out mean over operators -> (batch, time, channels)
    g = out.mean(dim=-1)
    aux = {"A_o": a_o, "A_t": a_t, "A_c": a_c, "G": g}
    loss, parts = compute_routing_regularization(aux)
    assert torch.isfinite(loss), f"End-to-end routing loss not finite: {loss}"
    assert loss.item() >= 0.0
