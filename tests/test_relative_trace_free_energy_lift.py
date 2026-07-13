from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet
from model.relative_trace_free_energy_lift import (
    RelativeTraceFreeEnergyLiftMSHNet,
    RelativeTraceFreeEnergyPool2d,
    relative_trace_free_energy_jet,
)


def test_rtfe_coordinate_is_bounded_scale_invariant_and_rotation_equivariant() -> None:
    torch.manual_seed(111)
    value = torch.relu(torch.randn(1, 6, 16, 16)) + 0.1
    coordinate = relative_trace_free_energy_jet(value)["coordinate"]
    scaled = relative_trace_free_energy_jet(5.0 * value)["coordinate"]
    rotated = relative_trace_free_energy_jet(
        torch.rot90(value, 1, (-2, -1))
    )["coordinate"]
    assert torch.all((coordinate >= 0) & (coordinate <= 1))
    torch.testing.assert_close(coordinate, scaled, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        torch.rot90(coordinate, 1, (-2, -1)), rotated, rtol=2e-5, atol=2e-6
    )


def test_constant_field_has_zero_coordinate_and_native_maximum() -> None:
    value = torch.full((1, 4, 8, 8), 3.0)
    lifted, state = RelativeTraceFreeEnergyPool2d()(value, return_state=True)
    torch.testing.assert_close(lifted[:, :4], torch.full((1, 4, 4, 4), 3.0))
    assert torch.count_nonzero(state["coordinate"]) == 0
    assert torch.count_nonzero(lifted[:, 4:]) == 0


def test_rtfe_zero_kernel_model_embeds_baseline_and_rng() -> None:
    torch.manual_seed(112)
    baseline = MSHNet(3).eval()
    canonical_next = torch.rand(5)
    torch.manual_seed(112)
    candidate = RelativeTraceFreeEnergyLiftMSHNet(3).eval()
    candidate_next = torch.rand(5)
    assert torch.equal(candidate_next, canonical_next)
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        base_sides, base_prediction = baseline(image, True)
        candidate_sides, candidate_prediction = candidate(image, True)
    for base, lifted in zip(base_sides, candidate_sides, strict=True):
        torch.testing.assert_close(lifted, base, rtol=0, atol=2e-5)
    torch.testing.assert_close(candidate_prediction, base_prediction, rtol=0, atol=2e-5)


def test_rtfe_coordinate_kernels_receive_finite_nonzero_gradient() -> None:
    torch.manual_seed(113)
    model = RelativeTraceFreeEnergyLiftMSHNet(3)
    _, prediction = model(torch.randn(2, 3, 32, 32), True)
    prediction.square().mean().backward()
    assert torch.isfinite(model.encoder_1[0].conv1.weight.grad).all()
    assert torch.count_nonzero(model.encoder_1[0].conv1.weight.grad[:, -1:]) > 0
    assert torch.count_nonzero(model.encoder_1[0].shortcut[0].weight.grad[:, -1:]) > 0


def test_rtfe_checkpoint_embedding_and_parameter_delta() -> None:
    baseline = MSHNet(3).eval()
    candidate = RelativeTraceFreeEnergyLiftMSHNet(3).eval()
    candidate.load_canonical_state_dict(baseline.state_dict())
    assert sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    ) == 320
