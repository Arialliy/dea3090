from __future__ import annotations

import torch
from torch import nn

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.deletion_stable_fusion import (
    DSFMSHNet,
    deletion_stable_fusion,
    normalized_soft_min,
)


def test_normalized_soft_min_is_exact_for_equal_coalitions() -> None:
    value = torch.randn(2, 1, 8, 8).repeat(1, 4, 1, 1)
    torch.testing.assert_close(normalized_soft_min(value, dim=1), value[:, :1])


def test_normalized_soft_min_focuses_gradient_on_worst_deletion() -> None:
    deletion = torch.tensor([[[[2.0]], [[-1.0]], [[0.5]], [[1.0]]]], requires_grad=True)
    pred = normalized_soft_min(deletion, dim=1)
    pred.backward()
    expected = torch.softmax(-deletion.detach(), dim=1)

    torch.testing.assert_close(deletion.grad, expected)
    torch.testing.assert_close(
        deletion.grad.sum(dim=1, keepdim=True),
        torch.ones_like(deletion.grad[:, :1]),
    )
    assert deletion.grad[0, 1, 0, 0] == deletion.grad.max()


def test_single_source_dominance_is_discounted() -> None:
    fusion = nn.Conv2d(4, 1, 1, bias=False)
    with torch.no_grad():
        fusion.weight.fill_(1.0)
    scale_logits = torch.zeros(1, 4, 8, 8)
    scale_logits[:, 0] = 8.0
    state = deletion_stable_fusion(scale_logits, fusion)

    assert torch.all(state["pred"] < state["affine_pred"])
    assert torch.all(state["deletion_responsibility"][:, 0] > 0.99)


def test_dsf_keeps_parameter_and_checkpoint_identity() -> None:
    canonical = DeterministicMSHNet(3)
    dsf = DSFMSHNet(3)
    assert tuple(canonical.state_dict()) == tuple(dsf.state_dict())
    assert sum(p.numel() for p in canonical.parameters()) == sum(
        p.numel() for p in dsf.parameters()
    )
    dsf.load_state_dict(canonical.state_dict(), strict=True)


def test_dsf_real_forward_state_and_backward_are_finite() -> None:
    torch.manual_seed(21)
    model = DSFMSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32, requires_grad=True)
    state = model(image, True, return_deletion_state=True)

    assert state["deletion_logits"].shape == (1, 4, 32, 32)
    torch.testing.assert_close(
        state["deletion_responsibility"].sum(dim=1, keepdim=True),
        torch.ones_like(state["pred"]),
    )
    state["pred"].square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert model.final.weight.grad is not None
    assert torch.isfinite(model.final.weight.grad).all()


def test_dsf_cold_path_matches_canonical_bit_exactly() -> None:
    torch.manual_seed(22)
    canonical = DeterministicMSHNet(3).eval()
    dsf = DSFMSHNet(3).eval()
    dsf.load_state_dict(canonical.state_dict(), strict=True)
    image = torch.randn(1, 3, 32, 32)

    canonical_sides, canonical_pred = canonical(image, False)
    dsf_sides, dsf_pred = dsf(image, False)
    assert canonical_sides == dsf_sides == []
    assert torch.equal(canonical_pred, dsf_pred)
