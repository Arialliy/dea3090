from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.loss import SLSIoULoss  # noqa: E402
from model.partial_sls_loss import PartialSLSIoULoss  # noqa: E402


def test_all_valid_is_bitwise_canonical_loss_and_gradient() -> None:
    torch.manual_seed(3)
    logits_a = torch.randn((2, 1, 8, 8), requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    target = (torch.rand_like(logits_a) > 0.8).float()
    valid = torch.ones_like(target)

    canonical = SLSIoULoss()(logits_a, target, warm_epoch=1, epoch=3)
    partial = PartialSLSIoULoss()(
        logits_b,
        target,
        valid,
        warm_epoch=1,
        epoch=3,
    )
    canonical.backward()
    partial.backward()

    assert torch.equal(canonical, partial)
    assert torch.equal(logits_a.grad, logits_b.grad)


def test_invalid_pixel_has_zero_gradient() -> None:
    logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, :, 1, 1] = 1
    valid = torch.ones_like(logits)
    valid[:, :, 0, 0] = 0

    loss = PartialSLSIoULoss()(logits, target, valid, warm_epoch=0, epoch=1)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 0] == 0


def test_partial_empty_positive_term_has_zero_prediction_gradient() -> None:
    logits = torch.zeros((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits)
    valid[:, :, 1, 1] = 0

    loss = PartialSLSIoULoss()(logits, target, valid, warm_epoch=0, epoch=1)
    loss.backward()

    assert loss == 1
    assert logits.grad is not None
    assert logits.grad.abs().sum() == 0
