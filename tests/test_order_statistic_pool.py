import pytest
import torch
import torch.nn.functional as F

from utils.order_statistic_pool import (
    channel_consensus_pool2d,
    counterfactual_self_support_pool2d,
    leave_one_channel_influence_pool2d,
    leave_one_peak_pool2d,
    support_persistence_pool2d,
)


def test_alpha_zero_is_exact_max_pooling() -> None:
    x = torch.tensor([[[[1.0, 4.0], [3.0, 2.0]]]])
    torch.testing.assert_close(
        leave_one_peak_pool2d(x, alpha=0.0), F.max_pool2d(x, 2, 2), rtol=0, atol=0
    )


def test_alpha_one_removes_single_largest_site_per_cell() -> None:
    x = torch.tensor([[[[1.0, 9.0, 5.0, 7.0], [3.0, 2.0, 8.0, 6.0]]]])
    expected = torch.tensor([[[[3.0, 7.0]]]])
    torch.testing.assert_close(leave_one_peak_pool2d(x, alpha=1.0), expected)


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        leave_one_peak_pool2d(torch.zeros(1, 1, 3, 4), alpha=0.5)
    with pytest.raises(ValueError):
        leave_one_peak_pool2d(torch.zeros(1, 1, 4, 4), alpha=1.1)


def test_channel_consensus_stays_between_second_and_first_order_statistics() -> None:
    torch.manual_seed(4)
    x = torch.randn(2, 5, 8, 10)
    result = channel_consensus_pool2d(x)
    maximum = leave_one_peak_pool2d(x, alpha=0.0)
    second = leave_one_peak_pool2d(x, alpha=1.0)
    assert torch.all(result <= maximum + 1e-7)
    assert torch.all(result >= second - 1e-7)


def test_support_persistence_never_removes_more_than_half_exclusive_evidence() -> None:
    torch.manual_seed(5)
    x = torch.randn(2, 5, 8, 10)
    result = support_persistence_pool2d(x)
    maximum = leave_one_peak_pool2d(x, alpha=0.0)
    second = leave_one_peak_pool2d(x, alpha=1.0)
    midpoint = 0.5 * (maximum + second)
    assert torch.all(result <= maximum + 1e-7)
    assert torch.all(result >= midpoint - 1e-7)


def test_leave_one_channel_influence_has_an_exact_inverse_channel_bound() -> None:
    torch.manual_seed(6)
    x = torch.randn(2, 8, 8, 10, requires_grad=True)
    result, state = leave_one_channel_influence_pool2d(x, return_state=True)
    maximum = leave_one_peak_pool2d(x, alpha=0.0)
    second = leave_one_peak_pool2d(x, alpha=1.0)
    lower_bound = maximum - (maximum - second) / x.shape[1]
    assert torch.all(result <= maximum + 1e-7)
    assert torch.all(result >= lower_bound - 1e-7)
    assert torch.all(state["self_influence"] >= 0)
    assert torch.all(state["self_influence"] <= 1.0 / x.shape[1] + 1e-7)
    result.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_leave_one_channel_influence_vanishes_for_identical_channel_votes() -> None:
    cell = torch.tensor([[[[1.0, 4.0], [3.0, 2.0]]]])
    x = cell.repeat(1, 5, 1, 1)
    result, state = leave_one_channel_influence_pool2d(x, return_state=True)
    torch.testing.assert_close(state["self_influence"], torch.zeros_like(state["self_influence"]), atol=1e-7, rtol=0)
    torch.testing.assert_close(result, F.max_pool2d(x, 2, 2), atol=1e-7, rtol=0)


def test_counterfactual_self_support_is_a_bounded_fraction_without_parameters() -> None:
    torch.manual_seed(7)
    x = torch.randn(2, 8, 8, 10, requires_grad=True)
    result, state = counterfactual_self_support_pool2d(x, return_state=True)
    maximum = leave_one_peak_pool2d(x, alpha=0.0)
    second = leave_one_peak_pool2d(x, alpha=1.0)
    assert torch.all(result <= maximum + 1e-7)
    assert torch.all(result >= second - 1e-7)
    ratio = state["counterfactual_self_support"]
    assert torch.all((ratio >= 0) & (ratio <= 1))
    result.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_counterfactual_self_support_is_zero_for_identical_channel_votes() -> None:
    cell = torch.tensor([[[[1.0, 4.0], [3.0, 2.0]]]])
    x = cell.repeat(1, 5, 1, 1)
    result, state = counterfactual_self_support_pool2d(x, return_state=True)
    torch.testing.assert_close(
        state["counterfactual_self_support"],
        torch.zeros_like(state["counterfactual_self_support"]),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(result, F.max_pool2d(x, 2, 2), atol=1e-7, rtol=0)
