from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet
from model.scale_normalized_tangent_lift import (
    ScaleNormalizedTangentLiftMSHNet,
    ScaleNormalizedTangentPool2d,
)


def test_constant_activation_has_zero_tangent_and_native_max() -> None:
    value = torch.full((1, 3, 4, 4), 2.0)
    lifted, state = ScaleNormalizedTangentPool2d()(value, return_state=True)
    torch.testing.assert_close(lifted[:, :3], torch.full((1, 3, 2, 2), 2.0))
    assert torch.count_nonzero(lifted[:, 3:]) == 0
    assert torch.count_nonzero(state["relative_slope"]) == 0


def test_relative_slope_is_bounded_and_positive_scale_invariant() -> None:
    torch.manual_seed(91)
    value = torch.relu(torch.randn(1, 4, 8, 8))
    _, state1 = ScaleNormalizedTangentPool2d()(value, return_state=True)
    _, state2 = ScaleNormalizedTangentPool2d()(7.0 * value, return_state=True)
    assert torch.all((state1["relative_slope"] >= 0) & (state1["relative_slope"] <= 1))
    torch.testing.assert_close(
        state1["relative_slope"], state2["relative_slope"], rtol=1e-5, atol=1e-6
    )


def test_zero_tangent_kernels_embed_canonical_mshnet() -> None:
    torch.manual_seed(92)
    baseline = MSHNet(3).eval()
    torch.manual_seed(92)
    candidate = ScaleNormalizedTangentLiftMSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        base_sides, base_prediction = baseline(image, True)
        candidate_sides, candidate_prediction = candidate(image, True)
    for base, lifted in zip(base_sides, candidate_sides, strict=True):
        torch.testing.assert_close(lifted, base, rtol=0, atol=2e-5)
    torch.testing.assert_close(candidate_prediction, base_prediction, rtol=0, atol=2e-5)


def test_tangent_coordinate_receives_nonzero_finite_gradients() -> None:
    torch.manual_seed(93)
    model = ScaleNormalizedTangentLiftMSHNet(3)
    _, prediction = model(torch.randn(2, 3, 32, 32), True)
    prediction.square().mean().backward()
    assert torch.isfinite(model.encoder_1[0].conv1.weight.grad).all()
    assert torch.count_nonzero(model.encoder_1[0].conv1.weight.grad[:, -1:]) > 0
    assert torch.count_nonzero(model.encoder_1[0].shortcut[0].weight.grad[:, -1:]) > 0


def test_canonical_checkpoint_embedding_and_rng_preservation() -> None:
    torch.manual_seed(94)
    baseline = MSHNet(3).eval()
    canonical_next = torch.rand(4)
    torch.manual_seed(94)
    candidate = ScaleNormalizedTangentLiftMSHNet(3).eval()
    candidate_next = torch.rand(4)
    assert torch.equal(candidate_next, canonical_next)
    candidate.load_canonical_state_dict(baseline.state_dict())
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _, base_prediction = baseline(image, True)
        _, candidate_prediction = candidate(image, True)
    torch.testing.assert_close(candidate_prediction, base_prediction, rtol=0, atol=2e-5)
