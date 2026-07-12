"""Parameter-space constrained deep supervision.

Each auxiliary gradient is minimally projected onto the half-space whose
inner product with the final-task gradient is non-negative.  The operation is
asymmetric: the final task is the constraint, not another peer objective.
There are no trainable parameters and the inference graph is unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


Gradient = torch.Tensor | None


def gradient_inner_product(
    left: Sequence[Gradient],
    right: Sequence[Gradient],
) -> torch.Tensor:
    """Return the global inner product over matching parameter gradients."""

    if len(left) != len(right):
        raise ValueError("gradient sequences must have the same length")
    reference = next(
        (value for value in (*left, *right) if value is not None),
        None,
    )
    if reference is None:
        return torch.tensor(0.0)
    result = reference.new_zeros(())
    for left_value, right_value in zip(left, right):
        if left_value is not None and right_value is not None:
            result = result + (left_value * right_value).sum()
    return result


def project_auxiliary_gradient(
    task_gradient: Sequence[Gradient],
    auxiliary_gradient: Sequence[Gradient],
    epsilon: float = 1e-12,
) -> tuple[tuple[Gradient, ...], dict[str, torch.Tensor]]:
    """Minimally remove an auxiliary component opposing the final task.

    The returned gradient solves

        min_g 0.5 ||g - g_aux||^2  subject to <g, g_task> >= 0.

    ``None`` entries preserve autograd's unused-parameter semantics.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if len(task_gradient) != len(auxiliary_gradient):
        raise ValueError("gradient sequences must have the same length")

    inner_product = gradient_inner_product(task_gradient, auxiliary_gradient)
    # The side objective can only move parameters on its autograd support.
    # Project in that reachable shared-parameter subspace; including final-only
    # parameters in the denominator would under-correct the conflict.
    reachable_task_gradient = tuple(
        task_value if auxiliary_value is not None else None
        for task_value, auxiliary_value in zip(task_gradient, auxiliary_gradient)
    )
    task_norm_squared = gradient_inner_product(
        reachable_task_gradient,
        reachable_task_gradient,
    )
    auxiliary_norm_squared = gradient_inner_product(
        auxiliary_gradient,
        auxiliary_gradient,
    )
    coefficient = torch.minimum(
        inner_product,
        torch.zeros_like(inner_product),
    ) / task_norm_squared.clamp_min(epsilon)

    projected: list[Gradient] = []
    for task_value, auxiliary_value in zip(
        reachable_task_gradient,
        auxiliary_gradient,
    ):
        if auxiliary_value is None:
            projected.append(None)
        elif task_value is None:
            projected.append(auxiliary_value)
        else:
            projected.append(auxiliary_value - coefficient * task_value)

    denominator = (
        task_norm_squared.sqrt() * auxiliary_norm_squared.sqrt()
    ).clamp_min(epsilon)
    statistics = {
        "cosine": (inner_product / denominator).detach(),
        "conflict": (inner_product < 0).to(inner_product.dtype).detach(),
        "task_norm": task_norm_squared.sqrt().detach(),
        "auxiliary_norm": auxiliary_norm_squared.sqrt().detach(),
        "removed_norm": (
            coefficient.abs() * task_norm_squared.sqrt()
        ).detach(),
    }
    return tuple(projected), statistics


def combine_task_and_auxiliary_gradients(
    task_gradient: Sequence[Gradient],
    auxiliary_gradients: Sequence[Sequence[Gradient]],
    denominator: float,
) -> tuple[Gradient, ...]:
    """Combine gradients using the canonical objective denominator."""

    if denominator <= 0:
        raise ValueError("denominator must be positive")
    length = len(task_gradient)
    if any(len(values) != length for values in auxiliary_gradients):
        raise ValueError("all gradient sequences must have the same length")

    combined: list[Gradient] = []
    for index, task_value in enumerate(task_gradient):
        values = [
            auxiliary[index]
            for auxiliary in auxiliary_gradients
            if auxiliary[index] is not None
        ]
        if task_value is None and not values:
            combined.append(None)
            continue
        if task_value is None:
            total = torch.zeros_like(values[0])
        else:
            total = task_value.clone()
        for value in values:
            total = total + value
        combined.append(total / denominator)
    return tuple(combined)


__all__ = [
    "combine_task_and_auxiliary_gradients",
    "gradient_inner_product",
    "project_auxiliary_gradient",
]
