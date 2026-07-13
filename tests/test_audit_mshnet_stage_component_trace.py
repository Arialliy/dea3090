from __future__ import annotations

import pytest

from tools.audit_mshnet_stage_component_trace import distribution, probability_auc


def test_probability_auc_has_pairwise_probability_semantics() -> None:
    assert probability_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert probability_auc([0.0], [1.0]) == 0.0
    assert probability_auc([1.0], [1.0]) == 0.5
    assert probability_auc([], [1.0]) is None


def test_distribution_is_fail_closed_for_empty_values() -> None:
    assert distribution([])["count"] == 0
    summary = distribution([1.0, 3.0])
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["median"] == pytest.approx(2.0)
