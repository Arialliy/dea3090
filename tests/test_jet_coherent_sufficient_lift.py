from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet
from model.jet_coherent_sufficient_lift import (
    JetCoherentSufficientLiftMSHNet,
    JetCoherentSufficientPool2d,
    jet_coherence,
)


def test_constant_field_has_zero_jet_and_native_maximum() -> None:
    value = torch.full((1, 3, 6, 6), 2.0)
    lifted, state = JetCoherentSufficientPool2d()(value, return_state=True)
    torch.testing.assert_close(lifted[:, :3], torch.full((1, 3, 3, 3), 2.0))
    assert torch.count_nonzero(lifted[:, 3:]) == 0
    assert torch.count_nonzero(state["coherence"]) == 0


def test_jet_is_positive_scale_invariant_and_rotation_equivariant() -> None:
    torch.manual_seed(101)
    value = torch.relu(torch.randn(1, 5, 12, 12)) + 0.1
    base = jet_coherence(value)["coherence"]
    scaled = jet_coherence(3.0 * value)["coherence"]
    rotated = jet_coherence(torch.rot90(value, 1, (-2, -1)))["coherence"]
    torch.testing.assert_close(base, scaled, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        torch.rot90(base, 1, (-2, -1)), rotated, rtol=2e-5, atol=2e-6
    )


def test_zero_jet_kernels_embed_canonical_forward_and_rng() -> None:
    torch.manual_seed(102)
    baseline = MSHNet(3).eval()
    canonical_next = torch.rand(5)
    torch.manual_seed(102)
    candidate = JetCoherentSufficientLiftMSHNet(3).eval()
    candidate_next = torch.rand(5)
    assert torch.equal(candidate_next, canonical_next)
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        base_sides, base_prediction = baseline(image, True)
        candidate_sides, candidate_prediction = candidate(image, True)
    for base, lifted in zip(base_sides, candidate_sides, strict=True):
        torch.testing.assert_close(lifted, base, rtol=0, atol=2e-5)
    torch.testing.assert_close(candidate_prediction, base_prediction, rtol=0, atol=2e-5)


def test_owned_jet_kernels_receive_finite_nonzero_gradients() -> None:
    torch.manual_seed(103)
    model = JetCoherentSufficientLiftMSHNet(3)
    _, prediction = model(torch.randn(2, 3, 32, 32), True)
    prediction.square().mean().backward()
    split = model.encoder_1[0].conv1.weight.shape[1] // 2
    assert torch.isfinite(model.encoder_1[0].conv1.weight.grad).all()
    assert torch.count_nonzero(model.encoder_1[0].conv1.weight.grad[:, split:]) > 0
    assert torch.count_nonzero(model.encoder_1[0].shortcut[0].weight.grad[:, split:]) > 0


def test_trained_checkpoint_embedding_and_parameter_delta() -> None:
    torch.manual_seed(104)
    baseline = MSHNet(3).eval()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    candidate = JetCoherentSufficientLiftMSHNet(3).eval()
    candidate.load_canonical_state_dict(baseline.state_dict())
    assert sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    ) == 5120
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        _, base_prediction = baseline(image, True)
        _, candidate_prediction = candidate(image, True)
    torch.testing.assert_close(candidate_prediction, base_prediction, rtol=0, atol=2e-5)
