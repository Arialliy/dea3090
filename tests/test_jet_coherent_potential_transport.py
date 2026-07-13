from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet
from model.jet_coherent_potential_transport import (
    JetCoherentPotentialTransportMSHNet,
)


def test_zero_potential_exactly_embeds_canonical_model_and_rng() -> None:
    torch.manual_seed(121)
    baseline = MSHNet(3).eval()
    canonical_next = torch.rand(4)
    torch.manual_seed(121)
    candidate = JetCoherentPotentialTransportMSHNet(3).eval()
    candidate_next = torch.rand(4)
    assert torch.equal(candidate_next, canonical_next)
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        base_sides, base_prediction = baseline(image, True)
        candidate_sides, candidate_prediction = candidate(image, True)
    for base, lifted in zip(base_sides, candidate_sides, strict=True):
        assert torch.equal(lifted, base)
    assert torch.equal(candidate_prediction, base_prediction)


def test_potential_receives_nonzero_finite_gradient_without_suppressing_native_path() -> None:
    torch.manual_seed(122)
    model = JetCoherentPotentialTransportMSHNet(3)
    state = model(
        torch.randn(2, 3, 32, 32), True, return_transport_state=True
    )
    assert torch.equal(state["e0_native"], state["e0_transported"])
    state["pred"].square().mean().backward()
    assert model.jet_potential.grad is not None
    assert torch.isfinite(model.jet_potential.grad).all()
    assert torch.count_nonzero(model.jet_potential.grad) > 0


def test_checkpoint_embedding_and_parameter_delta() -> None:
    baseline = MSHNet(3).eval()
    candidate = JetCoherentPotentialTransportMSHNet(3).eval()
    candidate.load_canonical_state_dict(baseline.state_dict())
    assert sum(p.numel() for p in candidate.parameters()) - sum(
        p.numel() for p in baseline.parameters()
    ) == 16
    assert torch.count_nonzero(candidate.jet_potential) == 0
