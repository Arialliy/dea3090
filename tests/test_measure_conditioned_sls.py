from __future__ import annotations

import torch

from model.loss import SLSIoULoss
from model.measure_conditioned_sls import MeasureConditionedSLSIoULoss


def test_all_non_null_batch_is_bitwise_canonical() -> None:
    torch.manual_seed(3)
    prediction_a = torch.randn(2, 1, 16, 16, requires_grad=True)
    prediction_b = prediction_a.detach().clone().requires_grad_(True)
    target = torch.zeros_like(prediction_a)
    target[0, 0, 2:5, 3:6] = 1
    target[1, 0, 10:12, 8:11] = 1

    expected = SLSIoULoss()(prediction_a, target, warm_epoch=5, epoch=8)
    actual = MeasureConditionedSLSIoULoss()(
        prediction_b,
        target,
        warm_epoch=5,
        epoch=8,
    )
    assert torch.equal(actual, expected)
    expected.backward()
    actual.backward()
    assert torch.equal(prediction_a.grad, prediction_b.grad)


def test_all_null_batch_penalizes_only_predicted_foreground() -> None:
    prediction = torch.tensor(
        [[[[1.0, -1.0], [0.5, -0.5]]]],
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    loss = MeasureConditionedSLSIoULoss()(
        prediction,
        target,
        warm_epoch=5,
        epoch=8,
    )
    torch.testing.assert_close(loss, torch.tensor((1.0 + 0.25) / 4.0))
    loss.backward()
    torch.testing.assert_close(
        prediction.grad,
        torch.tensor([[[[0.5, 0.0], [0.25, 0.0]]]]),
    )


def test_mixed_batch_preserves_positive_sample_and_replaces_only_null() -> None:
    torch.manual_seed(11)
    prediction = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, 0, 2:4, 3:5] = 1
    loss = MeasureConditionedSLSIoULoss()(
        prediction,
        target,
        warm_epoch=5,
        epoch=8,
    )
    positive = SLSIoULoss()(
        prediction[:1],
        target[:1],
        warm_epoch=5,
        epoch=8,
    )
    null = torch.relu(prediction[1:]).square().mean()
    torch.testing.assert_close(loss, (positive + null) / 2, atol=1e-6, rtol=1e-6)


def test_rejects_shape_mismatch() -> None:
    loss = MeasureConditionedSLSIoULoss()
    try:
        loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 2, 2), 5, 8)
    except ValueError as exc:
        assert "shapes" in str(exc)
    else:
        raise AssertionError("shape mismatch should be rejected")


def test_null_abstention_has_zero_loss_and_gradient() -> None:
    prediction = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = MeasureConditionedSLSIoULoss(null_mode="abstain")(
        prediction,
        target,
        warm_epoch=5,
        epoch=8,
    )
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_rejects_unknown_null_mode() -> None:
    try:
        MeasureConditionedSLSIoULoss(null_mode="unknown")
    except ValueError as exc:
        assert "null_mode" in str(exc)
    else:
        raise AssertionError("unknown null mode should be rejected")
