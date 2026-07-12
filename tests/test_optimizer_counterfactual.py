from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch.optim import Adagrad

from tools.optimizer_counterfactual import optimizer_counterfactual


class ScalarModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def test_exact_adagrad_counterfactual_preserves_original_state() -> None:
    model = ScalarModel()
    optimizer = Adagrad(model.parameters(), lr=1e-3)
    # Build a non-zero accumulator.  At Adagrad's first zero-state step,
    # same-sign scalar gradients are both normalized to the same update and a
    # genuine edge can have exactly zero marginal parameter effect.
    optimizer.zero_grad(set_to_none=True)
    warm_loss = (model.weight - 0.2).square()
    warm_loss.backward()
    optimizer.step()
    model_before = copy.deepcopy(model.state_dict())
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    def base_loss(current: nn.Module) -> torch.Tensor:
        return (current.weight - 0.2).square()

    def with_edge(current: nn.Module) -> torch.Tensor:
        return base_loss(current) + 0.5 * (current.weight + 1.0).square()

    def probe(current: nn.Module) -> torch.Tensor:
        return (current.weight - 0.5).square()

    result = optimizer_counterfactual(
        model,
        optimizer,
        with_edge_loss=with_edge,
        without_edge_loss=base_loss,
        probe_loss=probe,
    )

    assert torch.equal(model.state_dict()["weight"], model_before["weight"])
    assert optimizer.state_dict() == optimizer_before
    assert result.marginal_update_norm > 0
    assert result.sign_agreement
    assert result.actual_probe_harm * result.first_order_probe_harm > 0


def test_counterfactual_requires_complete_scalar_objectives() -> None:
    model = ScalarModel()
    optimizer = Adagrad(model.parameters(), lr=1e-3)

    def vector_loss(current: nn.Module) -> torch.Tensor:
        return torch.stack([current.weight, current.weight])

    try:
        optimizer_counterfactual(
            model,
            optimizer,
            with_edge_loss=vector_loss,
            without_edge_loss=lambda current: current.weight.square(),
            probe_loss=lambda current: current.weight.square(),
        )
    except ValueError as exc:
        assert "scalar" in str(exc)
    else:
        raise AssertionError("non-scalar loss closure must fail")
