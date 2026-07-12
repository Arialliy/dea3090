from __future__ import annotations

import torch

from model.MSHNet import MSHNet as WorkbenchMSHNet
from model.baselines.mshnet_deterministic import (
    ChannelAttention as DeterministicChannelAttention,
)
from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.baselines.mshnet_official import (
    ChannelAttention as OfficialChannelAttention,
)
from model.baselines.mshnet_official import MSHNet as OfficialMSHNet


def test_official_and_deterministic_variants_have_identical_state_schema() -> None:
    torch.manual_seed(7)
    official = OfficialMSHNet(3)
    deterministic = DeterministicMSHNet(3)

    assert official.state_dict().keys() == deterministic.state_dict().keys()
    assert sum(p.numel() for p in official.parameters()) == sum(
        p.numel() for p in deterministic.parameters()
    )
    deterministic.load_state_dict(official.state_dict(), strict=True)


def test_workbench_checkpoint_strips_to_clean_deterministic_model_exactly() -> None:
    torch.manual_seed(11)
    workbench = WorkbenchMSHNet(3).eval()
    clean = DeterministicMSHNet(3).eval()
    clean_keys = set(clean.state_dict())
    filtered = {
        key: value
        for key, value in workbench.state_dict().items()
        if key in clean_keys
    }
    clean.load_state_dict(filtered, strict=True)

    extras = set(workbench.state_dict()) - clean_keys
    assert extras
    assert all(key.startswith("decidability_head.") for key in extras)
    assert sum(p.numel() for p in workbench.parameters()) - sum(
        p.numel() for p in clean.parameters()
    ) == 521

    images = torch.randn(1, 3, 32, 32)
    with torch.inference_mode():
        workbench_masks, workbench_output = workbench(images, True)
        clean_masks, clean_output = clean(images, True)
    for actual, expected in zip(clean_masks, workbench_masks):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(clean_output, workbench_output, rtol=0.0, atol=0.0)


def test_official_and_deterministic_forward_match_but_tie_backward_differs() -> None:
    official = OfficialChannelAttention(16)
    deterministic = DeterministicChannelAttention(16)
    deterministic.load_state_dict(official.state_dict(), strict=True)
    with torch.no_grad():
        # Keep the sigmoid away from saturation so the different max-pool
        # tie subgradients remain observable at the input.
        official.fc1.weight.fill_(0.01)
        official.fc2.weight.fill_(0.01)
        deterministic.load_state_dict(official.state_dict(), strict=True)

    official_input = torch.ones(1, 16, 2, 2, requires_grad=True)
    deterministic_input = official_input.detach().clone().requires_grad_(True)
    official_output = official(official_input)
    deterministic_output = deterministic(deterministic_input)
    torch.testing.assert_close(
        official_output, deterministic_output, rtol=0.0, atol=0.0
    )

    official_output.sum().backward()
    deterministic_output.sum().backward()
    assert not torch.equal(official_input.grad, deterministic_input.grad)


def test_official_and_deterministic_full_forward_are_equal() -> None:
    torch.manual_seed(17)
    official = OfficialMSHNet(3).eval()
    deterministic = DeterministicMSHNet(3).eval()
    deterministic.load_state_dict(official.state_dict(), strict=True)
    images = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        official_masks, official_output = official(images, True)
        deterministic_masks, deterministic_output = deterministic(images, True)
    for actual, expected in zip(deterministic_masks, official_masks):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        deterministic_output, official_output, rtol=0.0, atol=0.0
    )
