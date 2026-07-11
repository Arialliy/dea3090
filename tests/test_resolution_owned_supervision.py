from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.resolution_owned_supervision import (  # noqa: E402
    OwnedSideSupervisionBuilder,
    ResolutionDecidableSupervisionGraph,
)


def test_rods_hard_assigns_one_owner_per_component() -> None:
    instance_map = torch.zeros((1, 16, 16), dtype=torch.long)
    instance_map[0, 4:6, 4:6] = 1
    instance_map[0, 10:14, 10:14] = 2

    graph = ResolutionDecidableSupervisionGraph(mode="hard")
    assignment = graph(instance_map)

    responsibilities = assignment.responsibilities[0]
    assert responsibilities.shape == (2, 4)
    assert torch.allclose(
        responsibilities.sum(dim=1),
        torch.ones(2),
    )
    assert torch.equal(
        assignment.primary_owner[0],
        responsibilities.argmax(dim=1),
    )


def test_unowned_components_are_ignored_not_background() -> None:
    instance_map = torch.zeros((1, 8, 8), dtype=torch.long)
    instance_map[0, 2:4, 2:4] = 1

    graph = ResolutionDecidableSupervisionGraph(mode="hard")
    assignment = graph(instance_map)
    owner = int(assignment.primary_owner[0][0])
    other_side = 1 if owner != 1 else 2

    builder = OwnedSideSupervisionBuilder(ignore_dilation=1)
    target, valid, _weight, active = builder(instance_map, assignment, other_side)

    assert not bool(active[0])
    assert target.sum() == 0
    assert valid.min() == 0


def test_interval_mode_can_assign_multiple_decidable_sides() -> None:
    instance_map = torch.zeros((1, 32, 32), dtype=torch.long)
    instance_map[0, 8:20, 8:20] = 1

    graph = ResolutionDecidableSupervisionGraph(
        mode="interval",
        interval_ratio=0.0,
        min_decidability=0.0,
    )
    assignment = graph(instance_map)

    assert int((assignment.responsibilities[0][0] > 0).sum()) > 1
