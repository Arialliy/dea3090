from __future__ import annotations

import torch

from model.MSHNet import MSHNet


def test_fusion_homotopy_has_exact_endpoints() -> None:
    torch.manual_seed(7)
    model = MSHNet(3).eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        masks_zero, prediction_zero = model(image, True, fusion_alpha=0.0)
        masks_one, prediction_one = model(image, True, fusion_alpha=1.0)
        masks_full, prediction_full = model(image, True)

    torch.testing.assert_close(prediction_zero, masks_zero[0], rtol=0, atol=0)
    torch.testing.assert_close(prediction_one, prediction_full, rtol=0, atol=0)
    for actual, expected in zip(masks_one, masks_full):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_fusion_homotopy_rejects_invalid_alpha() -> None:
    model = MSHNet(3).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        try:
            model(image, True, fusion_alpha=1.1)
        except ValueError as exc:
            assert "fusion_alpha" in str(exc)
        else:
            raise AssertionError("invalid fusion alpha should be rejected")
