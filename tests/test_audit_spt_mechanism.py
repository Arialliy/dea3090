from tools.audit_spt_mechanism import distribution, probability_auc


def test_probability_auc_and_empty_distribution_are_deterministic() -> None:
    assert probability_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert probability_auc([], [1.0]) is None
    assert distribution([])["count"] == 0
