from __future__ import annotations

import pytest
import torch

from tools.audit_sdrr_deletion_stability import (
    event_set_counts,
    responsibility_mask,
)


def test_responsibility_margin_excludes_boundary_events() -> None:
    z_full = torch.tensor([[[[0.5, 0.00001, 1.0]]]])
    deleted = torch.tensor(
        [[
            [[-0.5, -0.00001, 0.1]],
            [[0.2, 0.2, 0.2]],
            [[0.2, 0.2, 0.2]],
            [[0.2, 0.2, 0.2]],
        ]]
    )
    safe = torch.ones_like(z_full)

    unguarded = responsibility_mask(z_full, deleted, safe, delta=0.0)
    guarded = responsibility_mask(z_full, deleted, safe, delta=1e-4)

    assert int(unguarded.sum()) == 2
    assert int(guarded.sum()) == 1


def test_event_set_counts_reports_threshold_instability() -> None:
    first = torch.tensor([True, True, False, False])
    second = torch.tensor([True, False, True, False])

    assert event_set_counts(first, second) == {
        "first": 2,
        "second": 2,
        "intersection": 1,
        "union": 3,
        "mismatch": 2,
    }


def test_responsibility_rejects_negative_delta() -> None:
    z_full = torch.zeros(1, 1, 1, 1)
    deleted = torch.zeros(1, 4, 1, 1)
    with pytest.raises(ValueError, match="non-negative"):
        responsibility_mask(z_full, deleted, z_full, delta=-1e-3)
