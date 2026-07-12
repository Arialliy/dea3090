"""Exact two-branch optimizer counterfactuals for Phase-A influence audits."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.optim import Optimizer


LossClosure = Callable[[nn.Module], torch.Tensor]


@dataclass(frozen=True)
class OptimizerCounterfactualResult:
    with_edge_train_loss: float
    without_edge_train_loss: float
    probe_loss_with_edge: float
    probe_loss_without_edge: float
    actual_probe_harm: float
    first_order_probe_harm: float
    marginal_update_norm: float
    sign_agreement: bool


def _clone_branch(
    model: nn.Module,
    optimizer: Optimizer,
) -> tuple[nn.Module, Optimizer]:
    # Copying the pair in one operation preserves optimizer parameter
    # references to the cloned model and copies accumulators/step counters.
    branch_model, branch_optimizer = copy.deepcopy((model, optimizer))
    return branch_model, branch_optimizer


def _take_step(
    model: nn.Module,
    optimizer: Optimizer,
    loss_closure: LossClosure,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    loss = loss_closure(model)
    if loss.ndim != 0:
        raise ValueError("loss closure must return a scalar tensor")
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def optimizer_counterfactual(
    model: nn.Module,
    optimizer: Optimizer,
    *,
    with_edge_loss: LossClosure,
    without_edge_loss: LossClosure,
    probe_loss: LossClosure,
) -> OptimizerCounterfactualResult:
    """Compare exact optimizer steps with and without one supervision edge.

    Both training closures must return the *complete* objective for their
    branch.  Their only intended difference is the audited instance-head edge.
    This captures Adagrad's gradient-dependent accumulator update and the
    interaction of the edge gradient with every other objective.
    """

    base_parameters = dict(model.named_parameters())
    probe = probe_loss(model)
    if probe.ndim != 0:
        raise ValueError("probe loss closure must return a scalar tensor")
    probe_gradients = torch.autograd.grad(
        probe,
        tuple(base_parameters.values()),
        allow_unused=True,
        create_graph=False,
    )

    with_model, with_optimizer = _clone_branch(model, optimizer)
    without_model, without_optimizer = _clone_branch(model, optimizer)
    with_train = _take_step(with_model, with_optimizer, with_edge_loss)
    without_train = _take_step(
        without_model,
        without_optimizer,
        without_edge_loss,
    )

    with_parameters = dict(with_model.named_parameters())
    without_parameters = dict(without_model.named_parameters())
    if with_parameters.keys() != base_parameters.keys() or (
        without_parameters.keys() != base_parameters.keys()
    ):
        raise RuntimeError("counterfactual branches changed named-parameter keys")

    first_order = probe.detach().new_zeros(())
    squared_norm = probe.detach().new_zeros(())
    for (name, _base), gradient in zip(base_parameters.items(), probe_gradients):
        marginal_update = (
            with_parameters[name].detach() - without_parameters[name].detach()
        )
        squared_norm = squared_norm + marginal_update.square().sum()
        if gradient is not None:
            first_order = first_order + (gradient.detach() * marginal_update).sum()

    with torch.no_grad():
        probe_with = probe_loss(with_model)
        probe_without = probe_loss(without_model)
    actual = probe_with - probe_without
    first_value = float(first_order.cpu())
    actual_value = float(actual.cpu())
    sign_agreement = (
        first_value == 0.0
        and actual_value == 0.0
        or first_value * actual_value > 0.0
    )
    return OptimizerCounterfactualResult(
        with_edge_train_loss=with_train,
        without_edge_train_loss=without_train,
        probe_loss_with_edge=float(probe_with.cpu()),
        probe_loss_without_edge=float(probe_without.cpu()),
        actual_probe_harm=actual_value,
        first_order_probe_harm=first_value,
        marginal_update_norm=float(torch.sqrt(squared_norm).cpu()),
        sign_agreement=bool(sign_agreement),
    )


__all__ = ["OptimizerCounterfactualResult", "optimizer_counterfactual"]
