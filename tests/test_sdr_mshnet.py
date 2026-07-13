from __future__ import annotations

import torch

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.sdr_mshnet import SDRMSHNet


def test_sdr_mshnet_is_parameter_identical_and_forward_identical() -> None:
    torch.manual_seed(7)
    baseline = DeterministicMSHNet(3).eval()
    model = SDRMSHNet(3).eval()
    model.load_state_dict(baseline.state_dict(), strict=True)

    assert tuple(model.state_dict()) == tuple(baseline.state_dict())
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in baseline.parameters()
    )

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        baseline_sides, baseline_pred = baseline(x, True)
        model_sides, model_pred = model(x, True)
        state = model(x, True, return_responsibility_state=True)

    for actual, expected in zip(model_sides, baseline_sides):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model_pred, baseline_pred, rtol=0, atol=0)
    torch.testing.assert_close(state["pred"], baseline_pred, rtol=0, atol=0)
    torch.testing.assert_close(
        state["reconstructed"], baseline_pred, rtol=1e-5, atol=5e-5
    )
    torch.testing.assert_close(
        state["deletion_logits"],
        state["pred"] - state["contributions"],
        rtol=0,
        atol=0,
    )


def test_sdr_mshnet_complete_training_view_backpropagates() -> None:
    model = SDRMSHNet(3).train()
    x = torch.randn(2, 3, 32, 32)
    target = torch.zeros(2, 1, 32, 32)

    state = model(x, True, return_responsibility_state=True)
    loss, logs = model.responsibility_objective(
        state, target, safe_kernel=1
    )
    (state["pred"].mean() + loss).backward()

    assert torch.isfinite(loss)
    assert "responsible_count" in logs
    assert model.final.weight.grad is not None
    assert torch.isfinite(model.final.weight.grad).all()


def test_responsibility_state_rejects_single_scale_warmup_path() -> None:
    model = SDRMSHNet(3).eval()
    x = torch.randn(1, 3, 32, 32)

    try:
        model(x, False, return_responsibility_state=True)
    except ValueError as error:
        assert "four-scale warm path" in str(error)
    else:
        raise AssertionError("single-scale path must not fabricate SDR state")


def test_sdr_mshnet_exposes_conservative_routing_objective() -> None:
    model = SDRMSHNet(3).train()
    state = model(
        torch.randn(2, 3, 32, 32),
        True,
        return_responsibility_state=True,
    )
    target = torch.zeros(2, 1, 32, 32)

    loss, logs = model.responsibility_routing_objective(
        state, target, safe_kernel=1
    )

    assert torch.isfinite(loss)
    assert logs["responsibility_conserving_routing"] == 1


def test_sdr_mshnet_exposes_final_density_risk_objective() -> None:
    model = SDRMSHNet(3).train()
    state = model(
        torch.randn(2, 3, 32, 32),
        True,
        return_responsibility_state=True,
    )
    target = torch.zeros(2, 1, 32, 32)

    loss, logs = model.responsibility_density_objective(
        state, target, safe_kernel=1
    )

    assert torch.isfinite(loss)
    assert logs["responsibility_density_risk"] == 1
