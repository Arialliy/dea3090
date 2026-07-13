from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.orthogonal_scale_ownership import (
    OSOMSHNet,
    block_average_projection,
    orthogonal_ownership_bands,
)


def test_block_average_projection_is_idempotent_and_nested() -> None:
    torch.manual_seed(1)
    value = torch.randn(2, 3, 32, 40)
    p1 = block_average_projection(value, 2)
    p2 = block_average_projection(value, 4)
    p3 = block_average_projection(value, 8)

    torch.testing.assert_close(block_average_projection(p1, 2), p1)
    torch.testing.assert_close(block_average_projection(p2, 4), p2)
    torch.testing.assert_close(block_average_projection(p3, 8), p3)
    torch.testing.assert_close(block_average_projection(p1, 4), p2)
    torch.testing.assert_close(block_average_projection(p2, 8), p3)


def test_ownership_bands_form_exact_partition_of_identity() -> None:
    torch.manual_seed(2)
    value = torch.randn(2, 1, 32, 40)
    repeated_sources = value.repeat(1, 4, 1, 1)
    bands = orthogonal_ownership_bands(repeated_sources)

    torch.testing.assert_close(bands.sum(dim=1, keepdim=True), value)


def test_ownership_bands_are_pairwise_orthogonal() -> None:
    torch.manual_seed(3)
    sources = torch.randn(2, 4, 32, 40, dtype=torch.float64)
    bands = orthogonal_ownership_bands(sources)

    for left in range(4):
        for right in range(left + 1, 4):
            inner = (bands[:, left] * bands[:, right]).sum()
            torch.testing.assert_close(
                inner,
                inner.new_zeros(()),
                atol=1e-10,
                rtol=0.0,
            )


def test_oso_preserves_canonical_parameter_and_state_dict_identity() -> None:
    canonical = DeterministicMSHNet(3)
    oso = OSOMSHNet(3)

    assert tuple(canonical.state_dict()) == tuple(oso.state_dict())
    assert sum(p.numel() for p in canonical.parameters()) == sum(
        p.numel() for p in oso.parameters()
    )
    oso.load_state_dict(canonical.state_dict(), strict=True)


def test_oso_forward_reconstructs_owned_contributions_and_backpropagates() -> None:
    torch.manual_seed(4)
    model = OSOMSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32, requires_grad=True)

    state = model(image, True, return_ownership_state=True)
    assert state["ownership_contributions"].shape == (1, 4, 32, 32)
    torch.testing.assert_close(state["pred"], state["reconstructed"])
    loss = state["pred"].square().mean()
    loss.backward()

    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert model.final.weight.grad is not None
    assert torch.isfinite(model.final.weight.grad).all()


def test_oso_cold_path_is_bit_exact_to_canonical() -> None:
    torch.manual_seed(5)
    canonical = DeterministicMSHNet(3).eval()
    oso = OSOMSHNet(3).eval()
    oso.load_state_dict(canonical.state_dict(), strict=True)
    image = torch.randn(1, 3, 32, 32)

    canonical_sides, canonical_pred = canonical(image, False)
    oso_sides, oso_pred = oso(image, False)

    assert canonical_sides == oso_sides == []
    assert torch.equal(canonical_pred, oso_pred)


def test_oso_fails_closed_for_unaligned_spatial_shape() -> None:
    sources = torch.randn(1, 4, 30, 32)
    try:
        orthogonal_ownership_bands(sources)
    except ValueError as exc:
        assert "divisible by 8" in str(exc)
    else:
        raise AssertionError("unaligned OSO input must fail closed")
