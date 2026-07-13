from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet
from model.birth_constrained_scale_filtration import (
    BirthConstrainedScaleFiltrationMSHNet,
    birth_constrained_scale_filtration,
    geodesic_dilation,
    strict_local_maxima,
)


def test_strict_maxima_reject_flat_plateaus() -> None:
    value = torch.zeros(1, 1, 5, 5)
    assert strict_local_maxima(value).sum() == 0
    value[0, 0, 2, 2] = 1.0
    maxima = strict_local_maxima(value)
    assert maxima.sum() == 1
    assert maxima[0, 0, 2, 2] == 1


def test_strict_maxima_ignores_outside_image_at_negative_border() -> None:
    value = torch.full((1, 1, 3, 3), -4.0)
    value[0, 0, 0, 0] = -3.0
    maxima = strict_local_maxima(value)
    assert maxima.sum() == 1
    assert maxima[0, 0, 0, 0] == 1


def test_geodesic_dilation_never_exceeds_its_mask() -> None:
    marker = torch.zeros(1, 1, 7, 7)
    marker[0, 0, 3, 3] = 1.0
    mask = torch.ones_like(marker)
    reconstruction = geodesic_dilation(marker, mask, iterations=2)
    assert torch.all(reconstruction <= mask)
    assert reconstruction[0, 0, 3, 3] == 1
    assert reconstruction[0, 0, 3, 5] == 1
    assert reconstruction[0, 0, 0, 0] == 0


def test_zero_coarse_contributions_leave_the_fine_prefix_exact() -> None:
    torch.manual_seed(11)
    contributions = torch.zeros(2, 4, 9, 9)
    contributions[:, 0] = torch.randn(2, 9, 9)
    bias = torch.tensor([0.3])
    output = birth_constrained_scale_filtration(contributions, bias)
    torch.testing.assert_close(
        output,
        contributions[:, 0:1] + 0.3,
        rtol=0,
        atol=0,
    )


def test_positive_coarse_evidence_cannot_raise_a_remote_flat_region() -> None:
    contributions = torch.zeros(1, 4, 9, 9)
    contributions[0, 0, 2, 2] = 2.0
    contributions[0, 1, 7, 7] = 5.0
    output = birth_constrained_scale_filtration(contributions, None)
    assert output[0, 0, 7, 7] == 0
    assert output[0, 0, 2, 2] == 2


def test_bcsf_keeps_canonical_parameters_and_backpropagates() -> None:
    baseline = MSHNet(3)
    model = BirthConstrainedScaleFiltrationMSHNet(3)
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=True)
    assert missing == []
    assert unexpected == []
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )
    image = torch.randn(1, 3, 32, 32)
    masks, prediction = model(image, True)
    assert len(masks) == 4
    prediction.mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
