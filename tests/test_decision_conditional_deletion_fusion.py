from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.decision_conditional_deletion_fusion import (
    DCDFMSHNet,
    decision_conditional_deletion_fusion,
)


def test_fragile_positive_is_corrected_most_strongly() -> None:
    affine = torch.tensor([[[[4.0]], [[4.0]], [[-4.0]]]])
    robust = torch.tensor([[[[-4.0]], [[4.0]], [[-5.0]]]])
    state = decision_conditional_deletion_fusion(affine, robust)

    corrections = state["deletion_correction"].flatten()
    assert corrections[0] > corrections[1]
    assert corrections[0] > corrections[2]
    assert state["pred"][0, 0, 0, 0] < affine[0, 0, 0, 0]


def test_output_stays_between_robust_and_affine_when_fragile() -> None:
    affine = torch.tensor([[[[2.0]]]])
    robust = torch.tensor([[[[-1.0]]]])
    pred = decision_conditional_deletion_fusion(affine, robust)["pred"]
    assert robust.item() < pred.item() < affine.item()


def test_no_gap_means_exact_affine_identity() -> None:
    affine = torch.tensor([[[[-1.0]], [[2.0]]]])
    robust = affine + 1.0
    state = decision_conditional_deletion_fusion(affine, robust)
    assert torch.equal(state["fragility_gap"], torch.zeros_like(affine))
    assert torch.equal(state["pred"], affine)


def test_dcdf_keeps_parameter_and_checkpoint_identity() -> None:
    canonical = DeterministicMSHNet(3)
    dcdf = DCDFMSHNet(3)
    assert tuple(canonical.state_dict()) == tuple(dcdf.state_dict())
    assert sum(p.numel() for p in canonical.parameters()) == sum(
        p.numel() for p in dcdf.parameters()
    )
    dcdf.load_state_dict(canonical.state_dict(), strict=True)


def test_dcdf_real_forward_and_backward_are_finite() -> None:
    torch.manual_seed(31)
    model = DCDFMSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    state = model(image, True, return_deletion_state=True)

    assert state["pred"].shape == (1, 1, 32, 32)
    assert torch.all(state["deletion_correction"] >= 0)
    state["pred"].square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert model.final.weight.grad is not None
    assert torch.isfinite(model.final.weight.grad).all()


def test_dcdf_cold_path_matches_canonical_bit_exactly() -> None:
    torch.manual_seed(32)
    canonical = DeterministicMSHNet(3).eval()
    dcdf = DCDFMSHNet(3).eval()
    dcdf.load_state_dict(canonical.state_dict(), strict=True)
    image = torch.randn(1, 3, 32, 32)
    _, canonical_pred = canonical(image, False)
    _, dcdf_pred = dcdf(image, False)
    assert torch.equal(canonical_pred, dcdf_pred)
