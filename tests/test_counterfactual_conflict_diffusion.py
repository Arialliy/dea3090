from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.counterfactual_conflict_diffusion import (
    CCFDMSHNet,
    counterfactual_conflict_diffusion,
)
from model.dea_shared_discrepancy_stencil import SharedDiscrepancyStencil


def test_zero_initialization_embeds_affine_baseline_exactly() -> None:
    stencil = SharedDiscrepancyStencil()
    affine = torch.randn(2, 1, 16, 16)
    robust = torch.randn_like(affine)
    state = counterfactual_conflict_diffusion(affine, robust, stencil)
    assert torch.equal(state["pred"], affine)
    assert torch.equal(
        state["conflict_correction"], torch.zeros_like(affine)
    )


def test_conflict_correction_has_exact_zero_spatial_mean() -> None:
    torch.manual_seed(41)
    stencil = SharedDiscrepancyStencil(initial_weights=[0.1] * 8)
    affine = torch.randn(2, 1, 16, 16)
    robust = torch.randn_like(affine)
    correction = counterfactual_conflict_diffusion(
        affine, robust, stencil
    )["conflict_correction"]
    torch.testing.assert_close(
        correction.mean(dim=(-2, -1)),
        torch.zeros_like(correction.mean(dim=(-2, -1))),
        atol=1e-7,
        rtol=0.0,
    )


def test_zero_stencil_has_nonzero_first_order_learning_direction() -> None:
    torch.manual_seed(42)
    stencil = SharedDiscrepancyStencil()
    affine = torch.randn(1, 1, 16, 16)
    robust = torch.randn_like(affine)
    target = torch.randn_like(affine)
    pred = counterfactual_conflict_diffusion(
        affine, robust, stencil
    )["pred"]
    (pred - target).square().mean().backward()
    assert stencil.theta.grad is not None
    assert torch.isfinite(stencil.theta.grad).all()
    assert stencil.theta.grad.abs().sum() > 0


def test_ccfd_adds_exactly_eight_parameters_after_frozen_front() -> None:
    canonical = DeterministicMSHNet(3)
    ccfd = CCFDMSHNet(3)
    canonical_params = sum(p.numel() for p in canonical.parameters())
    ccfd_params = sum(p.numel() for p in ccfd.parameters())
    assert ccfd_params - canonical_params == 8
    missing, unexpected = ccfd.load_state_dict(
        canonical.state_dict(), strict=False
    )
    assert missing == ["conflict_stencil.theta"]
    assert unexpected == []


def test_ccfd_real_forward_and_backward_are_finite() -> None:
    torch.manual_seed(43)
    model = CCFDMSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    state = model(image, True, return_conflict_state=True)
    assert state["pred"].shape == (1, 1, 32, 32)
    torch.testing.assert_close(
        state["conflict_correction"].mean(dim=(-2, -1)),
        torch.zeros(1, 1),
        atol=1e-7,
        rtol=0.0,
    )
    state["pred"].square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert model.conflict_stencil.theta.grad is not None


def test_ccfd_cold_path_is_canonical_when_stencil_is_zero() -> None:
    torch.manual_seed(44)
    canonical = DeterministicMSHNet(3).eval()
    ccfd = CCFDMSHNet(3).eval()
    missing, unexpected = ccfd.load_state_dict(
        canonical.state_dict(), strict=False
    )
    assert missing == ["conflict_stencil.theta"] and unexpected == []
    image = torch.randn(1, 3, 32, 32)
    _, canonical_pred = canonical(image, False)
    _, ccfd_pred = ccfd(image, False)
    assert torch.equal(canonical_pred, ccfd_pred)
