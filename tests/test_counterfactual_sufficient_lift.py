from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet
from model.counterfactual_sufficient_lift import (
    CounterfactualSufficientLiftMSHNet,
    CounterfactualSufficientPool2d,
)


def test_lift_retains_max_and_exact_deleted_maximum() -> None:
    value = torch.tensor(
        [[[[1.0, 4.0], [3.0, 2.0]]]], requires_grad=True
    )
    lifted, state = CounterfactualSufficientPool2d()(
        value, return_state=True
    )
    assert lifted.shape == (1, 2, 1, 1)
    assert state["factual_maximum"].item() == 4.0
    assert state["deleted_maximum"].item() == 3.0
    assert state["exclusive_residual"].item() == 1.0
    torch.testing.assert_close(
        state["factual_maximum"] - state["exclusive_residual"],
        state["deleted_maximum"],
    )
    lifted.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_lift_does_not_suppress_single_site_evidence() -> None:
    value = torch.zeros(1, 1, 2, 2)
    value[0, 0, 0, 1] = 5.0
    lifted = CounterfactualSufficientPool2d()(value)
    assert lifted[0, 0, 0, 0] == 5.0
    assert lifted[0, 1, 0, 0] == 5.0


def test_zero_residual_domain_exactly_embeds_baseline_forward() -> None:
    torch.manual_seed(71)
    baseline = MSHNet(3).eval()
    torch.manual_seed(71)
    lifted = CounterfactualSufficientLiftMSHNet(3, active_stages=(0,)).eval()
    image = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        base_sides, base_prediction = baseline(image, True)
        lift_sides, lift_prediction = lifted(image, True)
    for base, candidate in zip(base_sides, lift_sides, strict=True):
        torch.testing.assert_close(candidate, base, rtol=0, atol=2e-5)
    torch.testing.assert_close(lift_prediction, base_prediction, rtol=0, atol=2e-5)


def test_lift_construction_preserves_canonical_rng_stream() -> None:
    torch.manual_seed(711)
    MSHNet(3)
    canonical_next = torch.rand(8)
    torch.manual_seed(711)
    CounterfactualSufficientLiftMSHNet(3, active_stages=(0,))
    lifted_next = torch.rand(8)
    assert torch.equal(lifted_next, canonical_next)


def test_canonical_checkpoint_embedding_preserves_trained_forward() -> None:
    torch.manual_seed(712)
    baseline = MSHNet(3).eval()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    candidate = CounterfactualSufficientLiftMSHNet(3, active_stages=(0,)).eval()
    candidate.load_canonical_state_dict(baseline.state_dict())
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _, baseline_prediction = baseline(image, True)
        _, candidate_prediction = candidate(image, True)
    torch.testing.assert_close(
        candidate_prediction, baseline_prediction, rtol=0, atol=2e-5
    )


def test_residual_kernels_receive_finite_nonzero_gradient() -> None:
    torch.manual_seed(72)
    model = CounterfactualSufficientLiftMSHNet(3, active_stages=(0,))
    image = torch.randn(2, 3, 32, 32)
    _, prediction = model(image, True)
    prediction.square().mean().backward()
    block = model.encoder_1[0]
    split = block.conv1.weight.shape[1] // 2
    residual_gradient = block.conv1.weight.grad[:, split:]
    assert torch.isfinite(residual_gradient).all()
    assert torch.count_nonzero(residual_gradient) > 0
    shortcut_gradient = block.shortcut[0].weight.grad[:, split:]
    assert torch.isfinite(shortcut_gradient).all()
    assert torch.count_nonzero(shortcut_gradient) > 0


def test_csl_changes_only_selected_downstream_input_domains() -> None:
    baseline = MSHNet(3)
    model = CounterfactualSufficientLiftMSHNet(3, active_stages=(0, 2))
    assert model.encoder_1[0].conv1.in_channels == 2 * baseline.encoder_1[0].conv1.in_channels
    assert model.encoder_2[0].conv1.in_channels == baseline.encoder_2[0].conv1.in_channels
    assert model.encoder_3[0].conv1.in_channels == 2 * baseline.encoder_3[0].conv1.in_channels
    assert model.middle_layer[0].conv1.in_channels == baseline.middle_layer[0].conv1.in_channels
