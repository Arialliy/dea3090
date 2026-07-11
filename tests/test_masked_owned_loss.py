from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.masked_owned_loss import MaskedOwnedScaleIoULoss  # noqa: E402


def test_masked_owned_loss_returns_per_sample_values() -> None:
    criterion = MaskedOwnedScaleIoULoss()
    logits = torch.zeros((2, 1, 4, 4))
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits)

    loss = criterion(logits, target, valid)

    assert loss.shape == (2,)
    assert torch.isfinite(loss).all()


def test_masked_owned_loss_rejects_shape_mismatch() -> None:
    criterion = MaskedOwnedScaleIoULoss()
    logits = torch.zeros((1, 1, 4, 4))
    target = torch.zeros((1, 1, 2, 2))
    valid = torch.ones_like(logits)

    with pytest.raises(ValueError, match="shapes must match"):
        criterion(logits, target, valid)


def test_invalid_pixels_have_zero_gradient() -> None:
    criterion = MaskedOwnedScaleIoULoss()
    logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 1, 1] = 1.0
    valid = torch.ones_like(logits)
    valid[:, :, 1, 1] = 0.0

    loss = criterion(logits, target, valid).mean()
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0, 1, 1] == 0
