from __future__ import annotations

import torch
import torch.nn as nn

from main import Trainer
from model.MSHNet import MSHNet
from model.counterfactual_responsibility import (
    build_safe_background,
    counterfactual_responsibility_suppression,
)
from model.loss import SLSIoULoss
from model.scale_coalition_supervision import leave_one_scale_out_coalitions


def test_only_decision_flip_scale_receives_gradient() -> None:
    z_full = torch.tensor([[[[1.0, 1.0], [-1.0, 1.0]]]])
    contributions = torch.tensor(
        [[
            [[2.0, 0.2], [3.0, -2.0]],
            [[0.1, 2.0], [0.0, 0.1]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]],
        requires_grad=True,
    )
    target = torch.zeros_like(z_full)

    loss, logs = counterfactual_responsibility_suppression(
        z_full, contributions, target, safe_kernel=1
    )
    loss.backward()

    expected_mask = torch.zeros_like(contributions, dtype=torch.bool)
    expected_mask[0, 0, 0, 0] = True
    expected_mask[0, 1, 0, 1] = True
    assert torch.equal(contributions.grad != 0, expected_mask)
    assert logs["responsible_count"] == 2


def test_target_neighbourhood_is_never_suppressed() -> None:
    z_full = torch.ones(1, 1, 7, 7)
    contributions = torch.zeros(1, 4, 7, 7, requires_grad=True)
    contributions.data[:, 0] = 2.0
    target = torch.zeros_like(z_full)
    target[:, :, 3, 3] = 1.0

    loss, logs = counterfactual_responsibility_suppression(
        z_full, contributions, target, safe_kernel=3
    )
    loss.backward()

    assert contributions.grad[0, 0, 3, 3] == 0
    assert contributions.grad[0, 0, 2:5, 2:5].abs().sum() == 0
    assert logs["responsible_count"] == 40


def test_no_flip_has_exact_zero_loss_and_gradient() -> None:
    z_full = torch.ones(2, 1, 4, 4)
    contributions = torch.zeros(2, 4, 4, 4, requires_grad=True)
    target = torch.zeros_like(z_full)

    loss, logs = counterfactual_responsibility_suppression(
        z_full, contributions, target, safe_kernel=1
    )
    loss.backward()

    assert loss.item() == 0.0
    torch.testing.assert_close(contributions.grad, torch.zeros_like(contributions))
    assert logs["responsible_count"] == 0


def test_safe_background_validates_kernel() -> None:
    target = torch.zeros(1, 1, 4, 4)
    try:
        build_safe_background(target, kernel_size=2)
    except ValueError as error:
        assert "positive odd" in str(error)
    else:
        raise AssertionError("even safe kernel must be rejected")


def test_detached_scale_evidence_calibrates_only_fusion_weights() -> None:
    masks = tuple(
        torch.ones(1, 1, 4, 4, requires_grad=True) for _ in range(4)
    )
    fusion = nn.Conv2d(4, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        fusion.weight.zero_()
        fusion.weight[0, 0, 0, 0] = 2.0
    z_full = torch.ones(1, 1, 4, 4)
    coalition = leave_one_scale_out_coalitions(
        tuple(mask.detach() for mask in masks), z_full, fusion
    )

    loss, logs = counterfactual_responsibility_suppression(
        z_full,
        coalition["contributions"],
        torch.zeros_like(z_full),
        safe_kernel=1,
    )
    loss.backward()

    assert logs["responsible_count"] == 16
    assert fusion.weight.grad is not None
    assert fusion.weight.grad.abs().sum() > 0
    assert all(mask.grad is None for mask in masks)


def test_crs_before_start_is_exact_canonical_mshnet_loss() -> None:
    torch.manual_seed(37)
    model = MSHNet(3)
    masks, z_full = model(torch.randn(2, 3, 32, 32), True)
    labels = torch.zeros_like(z_full)
    labels[:, :, 12:18, 12:18] = 1.0
    trainer = Trainer.__new__(Trainer)
    trainer.args = type(
        "Args",
        (),
        {
            "deep_supervision": "crs_flip_suppression",
            "crs_lambda": 0.05,
            "crs_start_epoch": 20,
            "crs_ramp_epochs": 20,
            "crs_safe_kernel": 15,
            "crs_detach_scale_evidence": False,
        },
    )()
    trainer.model = model
    trainer.loss_fun = SLSIoULoss()
    trainer.down = nn.MaxPool2d(2, 2)
    trainer.warm_epoch = 5
    trainer.last_deep_supervision_log = {}
    trainer.last_tgds_components = None

    actual = trainer.compute_deep_supervision_loss(
        z_full, masks, labels, instance_map=None, epoch=19
    )
    expected_sum = trainer.loss_fun(z_full, labels, 5, 19)
    target = labels
    for index, mask in enumerate(masks):
        if index > 0:
            target = trainer.down(target)
        expected_sum = expected_sum + trainer.loss_fun(mask, target, 5, 19)

    torch.testing.assert_close(actual, expected_sum / 5.0, rtol=0.0, atol=0.0)
    assert trainer.last_deep_supervision_log["crs_identity"] == 1.0
