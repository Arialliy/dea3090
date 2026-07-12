from __future__ import annotations

import torch

from model.task_gradient_supervision import (
    combine_task_and_auxiliary_gradients,
    gradient_inner_product,
    project_auxiliary_gradient,
)


def test_aligned_auxiliary_gradient_is_unchanged() -> None:
    task = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0]))
    auxiliary = (torch.tensor([2.0, 1.0]), torch.tensor([1.0, 3.0]))
    projected, stats = project_auxiliary_gradient(task, auxiliary)
    for actual, expected in zip(projected, auxiliary):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert stats["conflict"].item() == 0.0


def test_conflicting_gradient_is_minimally_projected_to_boundary() -> None:
    task = (torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))
    auxiliary = (torch.tensor([-2.0, 3.0]), torch.tensor([-2.0, 0.0]))
    projected, stats = project_auxiliary_gradient(task, auxiliary)

    torch.testing.assert_close(
        gradient_inner_product(task, projected),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        projected[0], torch.tensor([-2.0 / 3.0, 3.0]), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        projected[1], torch.tensor([-2.0 / 3.0, 4.0 / 3.0]), atol=1e-6, rtol=0
    )
    assert stats["conflict"].item() == 1.0
    assert stats["removed_norm"].item() > 0.0


def test_zero_task_gradient_leaves_auxiliary_gradient_unchanged() -> None:
    task = (torch.zeros(2), None)
    auxiliary = (torch.randn(2), torch.randn(3))
    projected, _stats = project_auxiliary_gradient(task, auxiliary)
    torch.testing.assert_close(projected[0], auxiliary[0], rtol=0, atol=0)
    torch.testing.assert_close(projected[1], auxiliary[1], rtol=0, atol=0)


def test_unused_auxiliary_parameter_remains_unused() -> None:
    task = (torch.ones(2), torch.ones(1))
    auxiliary = (torch.ones(2), None)
    projected, _stats = project_auxiliary_gradient(task, auxiliary)
    assert projected[1] is None


def test_final_only_parameter_does_not_dilute_shared_projection() -> None:
    task = (torch.tensor([1.0]), torch.tensor([1000.0]))
    auxiliary = (torch.tensor([-2.0]), None)
    projected, _stats = project_auxiliary_gradient(task, auxiliary)
    torch.testing.assert_close(projected[0], torch.tensor([0.0]))
    assert projected[1] is None
    torch.testing.assert_close(
        gradient_inner_product(task, projected),
        torch.tensor(0.0),
        atol=1e-6,
        rtol=0,
    )


def test_combination_preserves_canonical_denominator() -> None:
    task = (torch.tensor([5.0]), None)
    auxiliary = [
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([4.0]), None),
    ]
    combined = combine_task_and_auxiliary_gradients(task, auxiliary, 3.0)
    torch.testing.assert_close(combined[0], torch.tensor([10.0 / 3.0]))
    torch.testing.assert_close(combined[1], torch.tensor([2.0 / 3.0]))


def test_rejects_invalid_gradient_shapes_and_denominator() -> None:
    try:
        project_auxiliary_gradient((torch.ones(1),), (), epsilon=1e-12)
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("mismatched gradients should be rejected")

    try:
        combine_task_and_auxiliary_gradients((torch.ones(1),), [], 0.0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("nonpositive denominator should be rejected")
