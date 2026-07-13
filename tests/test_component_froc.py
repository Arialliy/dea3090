from __future__ import annotations

import numpy as np
import pytest
import torch

from utils.component_froc import ComponentFROC


def _example() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.full((2, 1, 16, 16), -10.0)
    labels = torch.zeros_like(logits)
    labels[0, 0, 4, 4] = 1.0
    labels[1, 0, 8, 8] = 1.0
    logits[0, 0, 4, 4] = 10.0
    logits[1, 0, 8, 8] = 10.0
    logits[1, 0, 13, 13] = 10.0
    return logits, labels


def test_component_froc_counts_instances_and_false_components() -> None:
    logits, labels = _example()
    metric = ComponentFROC(thresholds=(0.25, 0.5, 1.0))
    metric.update(logits, labels)
    curve = metric.get_curve()

    assert curve.num_images == 2
    assert curve.num_targets == 2
    assert curve.detection_probability.tolist() == [1.0, 1.0, 0.0]
    assert curve.false_positive_components_per_image.tolist() == [0.5, 0.5, 0.0]


def test_component_budget_envelope_uses_best_feasible_threshold() -> None:
    logits, labels = _example()
    metric = ComponentFROC(thresholds=(0.25, 0.5, 1.0))
    metric.update(logits, labels)
    low, high = metric.at_budgets((0.1, 0.5))

    assert low.detection_probability == 0.0
    assert low.achieved_fppi == 0.0
    assert low.threshold == 1.0
    assert high.detection_probability == 1.0
    assert high.achieved_fppi == 0.5
    assert high.threshold == 0.25
    assert metric.mean_low_budget_detection((0.1, 0.5)) == 0.5


def test_component_froc_batching_is_accumulation_invariant() -> None:
    logits, labels = _example()
    batched = ComponentFROC(thresholds=(0.1, 0.5, 0.9, 1.0))
    separate = ComponentFROC(thresholds=(0.1, 0.5, 0.9, 1.0))
    batched.update(logits, labels)
    for index in range(2):
        separate.update(logits[index : index + 1], labels[index : index + 1])

    a, b = batched.get_curve(), separate.get_curve()
    assert np.array_equal(a.detection_probability, b.detection_probability)
    assert np.array_equal(
        a.false_positive_components_per_image,
        b.false_positive_components_per_image,
    )


def test_component_froc_rejects_invalid_grids_and_shapes() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ComponentFROC(thresholds=(0.5, 0.5))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ComponentFROC(thresholds=(-0.1, 1.0))
    metric = ComponentFROC(thresholds=(0.5, 1.0))
    with pytest.raises(ValueError, match="equal shape"):
        metric.update(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 7, 8))


def test_logit_space_preserves_ranking_after_float32_sigmoid_would_saturate() -> None:
    logits = torch.full((1, 1, 8, 8), -100.0)
    labels = torch.zeros_like(logits)
    labels[0, 0, 2, 2] = 1.0
    logits[0, 0, 2, 2] = 100.0
    logits[0, 0, 6, 6] = 90.0
    assert torch.sigmoid(logits[0, 0, 2, 2]) == torch.sigmoid(logits[0, 0, 6, 6])

    metric = ComponentFROC(
        thresholds=(80.0, 95.0, 105.0), threshold_space="logit"
    )
    metric.update(logits, labels)
    curve = metric.get_curve()
    assert curve.threshold_space == "logit"
    assert curve.detection_probability.tolist() == [1.0, 1.0, 0.0]
    assert curve.false_positive_components_per_image.tolist() == [1.0, 0.0, 0.0]
