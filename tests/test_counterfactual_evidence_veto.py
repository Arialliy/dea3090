from __future__ import annotations

import torch
from torch import nn

from model.MSHNet import MSHNet
from model.dea_counterfactual_veto import (
    CounterfactualEvidenceVeto,
    FineScaleVetoMSHNet,
    SharedCEVMSHNet,
)


def _load_baseline_state(
    baseline: MSHNet,
    controlled: FineScaleVetoMSHNet | SharedCEVMSHNet,
) -> None:
    incompatible = controlled.load_state_dict(baseline.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "veto_head.veto_predictor.weight",
        "veto_head.veto_predictor.bias",
    }
    assert incompatible.unexpected_keys == []


def test_zero_strength_is_strict_mshnet_identity_after_loading_same_state() -> None:
    torch.manual_seed(101)
    baseline = MSHNet(3).eval()
    controlled = SharedCEVMSHNet(3, veto_strength=0.0).eval()
    _load_baseline_state(baseline, controlled)
    x = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        baseline_masks, baseline_pred = baseline(x, True)
        output = controlled(x, True, return_dict=True)

    assert torch.equal(output["z_base"], baseline_pred)
    assert torch.equal(output["pred"], baseline_pred)
    for controlled_mask, baseline_mask in zip(output["masks"], baseline_masks):
        assert torch.equal(controlled_mask, baseline_mask)
    assert torch.count_nonzero(output["cev"]["vetoes"]) == 0
    assert torch.count_nonzero(output["cev"]["delta"]) == 0


def test_veto_direction_is_determined_by_contribution_sign() -> None:
    veto = CounterfactualEvidenceVeto(
        context_channels=1,
        active_scales=(0,),
        kernel_size=1,
        initial_bias=0.0,
    )
    z_base = torch.zeros(1, 1, 1, 2)
    contributions = torch.zeros(1, 4, 1, 2)
    contributions[0, 0, 0, 0] = 2.0
    contributions[0, 0, 0, 1] = -4.0

    output = veto(
        z_base=z_base,
        contributions=contributions,
        decoder_feature_0=torch.zeros(1, 1, 1, 2),
    )

    assert output["vetoes"][0, 0, 0, 0].item() == 0.5
    assert output["pred"][0, 0, 0, 0] < z_base[0, 0, 0, 0]
    assert output["pred"][0, 0, 0, 1] > z_base[0, 0, 0, 1]
    assert output["pred"][0, 0, 0, 0].item() == -1.0
    assert output["pred"][0, 0, 0, 1].item() == 2.0
    assert torch.count_nonzero(output["vetoes"][:, 1:]) == 0


def test_delta_contains_only_vetoed_exact_scale_contributions() -> None:
    torch.manual_seed(103)
    veto = CounterfactualEvidenceVeto(
        context_channels=3,
        active_scales=(0, 1, 2, 3),
        kernel_size=3,
        initial_bias=0.0,
    )
    with torch.no_grad():
        veto.veto_predictor.weight.normal_(mean=0.0, std=0.1)
    z_base = torch.randn(2, 1, 7, 9)
    contributions = torch.randn(2, 4, 7, 9)

    output = veto(
        z_base=z_base,
        contributions=contributions,
        decoder_feature_0=torch.randn(2, 3, 7, 9),
        veto_strength=0.7,
    )
    expected_delta = -(output["vetoes"] * contributions).sum(
        dim=1, keepdim=True
    )

    assert torch.equal(output["delta"], expected_delta)
    assert torch.equal(output["pred"], z_base + expected_delta)
    assert torch.equal(output["without_scale"], z_base - contributions)


def test_fine_and_shared_controls_have_the_intended_scale_contract() -> None:
    fine = FineScaleVetoMSHNet(3)
    shared = SharedCEVMSHNet(3)

    assert fine.active_scales == (0,)
    assert shared.active_scales == (0, 1, 2, 3)
    assert sum(parameter.numel() for parameter in fine.veto_head.parameters()) == 883
    assert sum(parameter.numel() for parameter in shared.veto_head.parameters()) == 883
    assert sum(
        isinstance(module, nn.Conv2d)
        for module in shared.veto_head.modules()
    ) == 1
    assert shared.veto_head.veto_predictor is next(
        module
        for module in shared.veto_head.modules()
        if isinstance(module, nn.Conv2d)
    )


def test_frozen_model_trains_only_veto_head_and_keeps_baseline_bn_in_eval() -> None:
    model = FineScaleVetoMSHNet(3, freeze_baseline=True)
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert trainable == {
        "veto_head.veto_predictor.weight",
        "veto_head.veto_predictor.bias",
    }

    model.train()
    baseline_batch_norms = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
        and not name.startswith("veto_head")
    ]
    assert baseline_batch_norms
    assert model.veto_head.training
    assert model.veto_head.veto_predictor.training
    assert all(not module.training for module in baseline_batch_norms)
    assert not model.encoder_0.training
    assert not model.decoder_0.training
    assert not model.final.training


def test_full_forward_shapes_are_finite() -> None:
    torch.manual_seed(107)
    model = FineScaleVetoMSHNet(3).eval()

    with torch.no_grad():
        output = model(
            torch.randn(1, 3, 32, 48),
            True,
            return_dict=True,
            veto_strength=1.0,
        )

    assert [tuple(mask.shape) for mask in output["masks"]] == [
        (1, 1, 32, 48),
        (1, 1, 16, 24),
        (1, 1, 8, 12),
        (1, 1, 4, 6),
    ]
    assert output["pred"].shape == (1, 1, 32, 48)
    assert output["z_base"].shape == (1, 1, 32, 48)
    assert output["scale_logits_full"].shape == (1, 4, 32, 48)
    assert output["decoder_feature_0"].shape == (1, 16, 32, 48)
    assert output["cev"]["contributions"].shape == (1, 4, 32, 48)
    assert output["cev"]["without_scale"].shape == (1, 4, 32, 48)
    assert output["cev"]["vetoes"].shape == (1, 4, 32, 48)
    assert output["cev"]["delta"].shape == (1, 1, 32, 48)
    assert torch.count_nonzero(output["cev"]["vetoes"][:, 1:]) == 0

    tensors = [
        output["pred"],
        output["z_base"],
        output["scale_logits_full"],
        output["decoder_feature_0"],
        output["cev"]["contributions"],
        output["cev"]["without_scale"],
        output["cev"]["vetoes"],
        output["cev"]["delta"],
        *output["masks"],
    ]
    assert all(torch.isfinite(tensor).all() for tensor in tensors)
