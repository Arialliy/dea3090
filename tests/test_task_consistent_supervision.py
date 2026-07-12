from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.task_consistent_supervision import (  # noqa: E402
    REASON_DISAPPEARED,
    REASON_LOW_IOU,
    REASON_MERGED,
    TaskConsistentAssignment,
    TaskConsistentPartialTargetBuilder,
    TaskConsistentProjectionGraph,
)


def test_stride_one_is_identity_feasible() -> None:
    instance_map = torch.zeros((1, 8, 8), dtype=torch.long)
    instance_map[0, 2:4, 3:5] = 1
    graph = TaskConsistentProjectionGraph(strides=(1,))

    assignment = graph(instance_map)

    assert bool(assignment.feasible[0][0, 0])
    assert assignment.recovery_iou[0][0, 0] == 1
    assert assignment.centroid_distance[0][0, 0] == 0


def test_tiny_target_is_rejected_by_task_recovery_iou() -> None:
    instance_map = torch.zeros((1, 8, 8), dtype=torch.long)
    instance_map[0, 2, 2] = 1
    graph = TaskConsistentProjectionGraph(strides=(2,), min_iou=0.5)

    assignment = graph(instance_map)

    assert not bool(assignment.feasible[0][0, 0])
    assert assignment.recovery_iou[0][0, 0] == 0.25
    assert int(assignment.reason_code[0][0, 0]) == REASON_LOW_IOU


def test_scene_level_merge_rejects_every_participant() -> None:
    instance_map = torch.zeros((1, 8, 8), dtype=torch.long)
    instance_map[0, 0:2, 0] = 1
    instance_map[0, 0:2, 3] = 2
    graph = TaskConsistentProjectionGraph(strides=(2,), min_iou=0.0)

    assignment = graph(instance_map)

    assert not assignment.feasible[0].any()
    assert torch.equal(
        assignment.reason_code[0][:, 0],
        torch.tensor([REASON_MERGED, REASON_MERGED], dtype=torch.int8),
    )


def test_unknown_takes_precedence_over_positive_in_same_cell() -> None:
    instance_map = torch.zeros((1, 4, 4), dtype=torch.long)
    instance_map[0, 0, 0] = 1
    instance_map[0, 1, 1] = 2
    assignment = TaskConsistentAssignment(
        feasible=[torch.tensor([[True], [False]])],
        recovery_iou=[torch.ones((2, 1))],
        centroid_distance=[torch.zeros((2, 1))],
        reason_code=[torch.zeros((2, 1), dtype=torch.int8)],
        component_ids=[torch.tensor([1, 2])],
        strides=(2,),
        min_iou=0.5,
        max_centroid_distance=3.0,
    )
    builder = TaskConsistentPartialTargetBuilder(strides=(2,))

    target, valid, active = builder(instance_map, assignment, side_index=0)

    assert target[0, 0, 0, 0] == 0
    assert valid[0, 0, 0, 0] == 0
    assert not bool(active[0])


def test_component_id_permutation_does_not_change_feasibility_geometry() -> None:
    first = torch.zeros((1, 8, 8), dtype=torch.long)
    first[0, 1:3, 1:3] = 1
    first[0, 5:7, 5:7] = 2
    second = first.clone()
    second[first == 1] = 2
    second[first == 2] = 1
    graph = TaskConsistentProjectionGraph(strides=(1, 2))

    assignment_a = graph(first)
    assignment_b = graph(second)

    assert torch.equal(
        assignment_a.feasible[0].sort(dim=0).values,
        assignment_b.feasible[0].sort(dim=0).values,
    )


def test_direct_stride_projection_matches_repeated_pooling_on_valid_shape() -> None:
    torch.manual_seed(19)
    mask = torch.rand((32, 32)) > 0.9

    direct = TaskConsistentProjectionGraph._project_and_lift(mask, stride=8)
    pooled = mask.float().view(1, 1, 32, 32)
    for _ in range(3):
        pooled = F.max_pool2d(pooled, kernel_size=2, stride=2)
    repeated = F.interpolate(pooled, size=mask.shape, mode="nearest")[0, 0] > 0.5

    assert torch.equal(direct, repeated)


def test_border_target_is_not_dropped_on_supported_divisible_shape() -> None:
    instance_map = torch.zeros((1, 32, 32), dtype=torch.long)
    instance_map[0, 30:32, 30:32] = 1
    graph = TaskConsistentProjectionGraph(strides=(1, 2, 4, 8), min_iou=0.0)

    assignment = graph(instance_map)

    assert torch.isfinite(assignment.centroid_distance[0][0]).all()
    assert not bool(
        (assignment.reason_code[0][0] == REASON_DISAPPEARED).any()
    ), "supported border target must not disappear through pooling floor"
