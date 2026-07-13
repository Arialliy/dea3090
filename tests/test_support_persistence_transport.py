import torch

from model.baselines.mshnet_deterministic import MSHNet
from model.support_persistence_transport import (
    SupportPersistenceMSHNet,
    SupportPersistencePool2d,
)
from utils.order_statistic_pool import leave_one_peak_pool2d, support_persistence_pool2d


def test_support_persistence_pool_is_bounded_by_order_statistics() -> None:
    torch.manual_seed(8)
    x = torch.randn(2, 7, 16, 12, requires_grad=True)
    pool = SupportPersistencePool2d()
    output, state = pool(x, return_state=True)
    maximum = leave_one_peak_pool2d(x, alpha=0.0)
    deleted = leave_one_peak_pool2d(x, alpha=1.0)
    assert torch.all(output <= maximum + 1e-7)
    assert torch.all(output >= deleted - 1e-7)
    assert torch.all((state["survival_gate"] >= 0) & (state["survival_gate"] <= 1))
    output.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_model_and_audit_implement_the_same_parameter_free_equation() -> None:
    torch.manual_seed(81)
    x = torch.randn(2, 7, 16, 12)
    model_output = SupportPersistencePool2d()(x)
    audit_output = support_persistence_pool2d(x)
    torch.testing.assert_close(model_output, audit_output, rtol=0, atol=0)


def test_spt0_changes_only_the_first_native_resampling_rule() -> None:
    baseline = MSHNet(3)
    model = SupportPersistenceMSHNet(3, active_stages=(0,))
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert missing == []
    assert unexpected == []
    assert model.active_stages == (0,)
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    assert model_parameters == baseline_parameters


def test_spt0_real_forward_and_backward_are_finite() -> None:
    torch.manual_seed(9)
    model = SupportPersistenceMSHNet(3, active_stages=(0,))
    image = torch.randn(1, 3, 32, 32)
    masks, prediction = model(image, True)
    assert len(masks) == 4
    assert prediction.shape == (1, 1, 32, 32)
    prediction.mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
