from __future__ import annotations

import numpy as np

from tools.bootstrap_sdrr_checkpoint_pair import paired_bootstrap


def _stats(intersection: list[int], union: list[int]) -> dict[str, np.ndarray]:
    count = len(intersection)
    return {
        "intersection": np.asarray(intersection, dtype=np.float64),
        "union": np.asarray(union, dtype=np.float64),
        "matches": np.ones(count, dtype=np.float64),
        "targets": np.ones(count, dtype=np.float64),
        "false_alarm_area": np.zeros(count, dtype=np.float64),
        "image_area": np.full(count, 100.0),
    }


def test_paired_bootstrap_preserves_uniform_positive_iou_delta() -> None:
    baseline = _stats([1, 1, 1], [2, 2, 2])
    candidate = _stats([2, 2, 2], [2, 2, 2])

    result = paired_bootstrap(baseline, candidate, samples=200, seed=7)

    assert result["delta"]["IoU"] == 0.5
    assert result["bootstrap"]["IoU"]["percentile_95_ci"] == [0.5, 0.5]
    assert result["bootstrap"]["IoU"]["probability_delta_gt_zero"] == 1.0
