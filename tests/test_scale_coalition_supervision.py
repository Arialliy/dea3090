from __future__ import annotations

import torch
import torch.nn as nn

from main import Trainer
from model.MSHNet import MSHNet
from model.loss import SLSIoULoss
from model.scale_coalition_supervision import (
    assemble_mshnet_scale_logits,
    direct_zero_channel_coalitions,
    leave_one_scale_out_coalitions,
    nested_scale_filtration,
)


def make_masks() -> list[torch.Tensor]:
    torch.manual_seed(19)
    return [
        torch.randn(2, 1, 16, 16, requires_grad=True),
        torch.randn(2, 1, 8, 8, requires_grad=True),
        torch.randn(2, 1, 4, 4, requires_grad=True),
        torch.randn(2, 1, 2, 2, requires_grad=True),
    ]


def test_coalition_reconstruction_matches_native_fusion() -> None:
    masks = make_masks()
    fusion = nn.Conv2d(4, 1, kernel_size=3, padding=1)
    scale_logits = assemble_mshnet_scale_logits(masks)
    z_full = fusion(scale_logits)

    result = leave_one_scale_out_coalitions(masks, z_full, fusion)

    torch.testing.assert_close(result["scale_logits"], scale_logits)
    torch.testing.assert_close(result["reconstructed"], z_full)
    for deleted_scale in range(4):
        intervened = scale_logits.clone()
        intervened[:, deleted_scale] = 0.0
        expected = fusion(intervened)
        torch.testing.assert_close(
            result["coalition_logits"][:, deleted_scale : deleted_scale + 1],
            expected,
        )


def test_deleted_scale_has_no_gradient_through_its_coalition() -> None:
    masks = make_masks()
    fusion = nn.Conv2d(4, 1, kernel_size=3, padding=1, bias=True)
    z_full = fusion(assemble_mshnet_scale_logits(masks))
    result = leave_one_scale_out_coalitions(masks, z_full, fusion)

    deleted_scale = 2
    coalition = result["coalition_logits"][
        :, deleted_scale : deleted_scale + 1
    ]
    gradients = torch.autograd.grad(coalition.sum(), masks, allow_unused=True)

    assert gradients[deleted_scale] is not None
    torch.testing.assert_close(
        gradients[deleted_scale],
        torch.zeros_like(gradients[deleted_scale]),
    )
    for scale, gradient in enumerate(gradients):
        if scale != deleted_scale:
            assert gradient is not None
            assert gradient.abs().sum() > 0


def test_helper_adds_no_trainable_parameters() -> None:
    masks = make_masks()
    fusion = nn.Conv2d(4, 1, kernel_size=3, padding=1)
    before = tuple(fusion.parameters())
    z_full = fusion(assemble_mshnet_scale_logits(masks))

    leave_one_scale_out_coalitions(masks, z_full, fusion)

    assert tuple(fusion.parameters()) == before


def test_direct_zero_channel_audit_matches_native_manual_deletion() -> None:
    masks = make_masks()
    fusion = nn.Conv2d(4, 1, kernel_size=3, padding=1)
    scale_logits = assemble_mshnet_scale_logits(masks)

    direct = direct_zero_channel_coalitions(scale_logits, fusion)

    assert direct.shape == (2, 4, 16, 16)
    for scale in range(4):
        retained = scale_logits.clone()
        retained[:, scale] = 0.0
        torch.testing.assert_close(
            direct[:, scale : scale + 1], fusion(retained), rtol=0.0, atol=0.0
        )


def test_filtration_states_are_exact_nested_native_fusions() -> None:
    masks = make_masks()
    fusion = nn.Conv2d(4, 1, kernel_size=3, padding=1)
    scale_logits = assemble_mshnet_scale_logits(masks)
    z_full = fusion(scale_logits)

    result = nested_scale_filtration(masks, z_full, fusion)

    for terminal_scale in range(4):
        intervened = scale_logits.clone()
        intervened[:, terminal_scale + 1 :] = 0.0
        expected = fusion(intervened) if terminal_scale < 3 else z_full
        torch.testing.assert_close(
            result["filtration_logits"][
                :, terminal_scale : terminal_scale + 1
            ],
            expected,
        )


def test_trainer_uses_one_final_and_four_coalition_objectives() -> None:
    torch.manual_seed(23)
    model = MSHNet(3)
    images = torch.randn(2, 3, 32, 32)
    masks, z_full = model(images, True)
    labels = torch.zeros_like(z_full)
    labels[:, :, 12:18, 14:20] = 1.0

    trainer = Trainer.__new__(Trainer)
    trainer.args = type(
        "Args", (), {"deep_supervision": "cscs_leave_one_out"}
    )()
    trainer.model = model
    trainer.loss_fun = SLSIoULoss()
    trainer.warm_epoch = 5
    trainer.last_deep_supervision_log = {}
    trainer.last_tgds_components = None

    actual = trainer.compute_deep_supervision_loss(
        z_full,
        masks,
        labels,
        instance_map=None,
        epoch=6,
    )
    coalition = leave_one_scale_out_coalitions(masks, z_full, model.final)
    terms = [trainer.loss_fun(z_full, labels, 5, 6)]
    terms.extend(
        trainer.loss_fun(
            coalition["coalition_logits"][:, index : index + 1],
            labels,
            5,
            6,
        )
        for index in range(4)
    )
    expected = torch.stack(terms).mean()

    torch.testing.assert_close(actual, expected)
    assert trainer.last_deep_supervision_log[
        "canonical_objective_count"
    ] == 5.0
    assert trainer.last_deep_supervision_log[
        "coalition_reconstruction_error"
    ] < 1e-4


def test_anchor_filtration_preserves_five_objective_budget() -> None:
    torch.manual_seed(29)
    model = MSHNet(3)
    images = torch.randn(2, 3, 32, 32)
    masks, z_full = model(images, True)
    labels = torch.zeros_like(z_full)
    labels[:, :, 10:17, 13:19] = 1.0

    trainer = Trainer.__new__(Trainer)
    trainer.args = type(
        "Args", (), {"deep_supervision": "asfs_anchor_filtration"}
    )()
    trainer.model = model
    trainer.loss_fun = SLSIoULoss()
    trainer.down = nn.MaxPool2d(2, 2)
    trainer.warm_epoch = 5
    trainer.last_deep_supervision_log = {}
    trainer.last_tgds_components = None

    actual = trainer.compute_deep_supervision_loss(
        z_full, masks, labels, instance_map=None, epoch=6
    )
    filtration = nested_scale_filtration(masks, z_full, model.final)
    expected_terms = (
        trainer.loss_fun(masks[0], labels, 5, 6),
        trainer.loss_fun(masks[1], trainer.down(labels), 5, 6),
        trainer.loss_fun(filtration["filtration_logits"][:, 1:2], labels, 5, 6),
        trainer.loss_fun(filtration["filtration_logits"][:, 2:3], labels, 5, 6),
        trainer.loss_fun(z_full, labels, 5, 6),
    )

    torch.testing.assert_close(actual, torch.stack(expected_terms).mean())
    assert trainer.last_deep_supervision_log[
        "canonical_objective_count"
    ] == 5.0


def test_rdfs_role_discovery_phase_is_exact_canonical_objective() -> None:
    torch.manual_seed(31)
    model = MSHNet(3)
    images = torch.randn(2, 3, 32, 32)
    masks, z_full = model(images, True)
    labels = torch.zeros_like(z_full)
    labels[:, :, 11:18, 9:16] = 1.0

    trainer = Trainer.__new__(Trainer)
    trainer.args = type(
        "Args",
        (),
        {
            "deep_supervision": "rdfs_continuation",
            "rdfs_start_epoch": 20,
            "rdfs_ramp_epochs": 20,
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
    canonical_sum = trainer.loss_fun(z_full, labels, 5, 19)
    target = labels
    for index, mask in enumerate(masks):
        if index > 0:
            target = trainer.down(target)
        canonical_sum = canonical_sum + trainer.loss_fun(mask, target, 5, 19)
    expected = canonical_sum / 5.0

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert trainer.last_deep_supervision_log["rdfs_alpha"] == 0.0


def test_rdfs_continuation_alpha_reaches_anchor_filtration() -> None:
    trainer = Trainer.__new__(Trainer)
    trainer.args = type(
        "Args",
        (),
        {
            "deep_supervision": "rdfs_continuation",
            "rdfs_start_epoch": 20,
            "rdfs_ramp_epochs": 10,
        },
    )()

    assert trainer.get_rdfs_alpha(19) == 0.0
    assert trainer.get_rdfs_alpha(20) == 0.0
    assert trainer.get_rdfs_alpha(25) == 0.5
    assert trainer.get_rdfs_alpha(30) == 1.0
