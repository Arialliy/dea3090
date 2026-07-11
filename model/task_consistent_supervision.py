from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure


REASON_FEASIBLE = 0
REASON_DISAPPEARED = 1
REASON_SPLIT = 2
REASON_MERGED = 3
REASON_LOW_IOU = 4
REASON_LOCALIZATION = 5


@dataclass(frozen=True)
class TaskConsistentAssignment:
    """Binary instance-to-head graph induced by scene-level label projection.

    This is deliberately a *projection-consistency* object.  It evaluates the
    label produced by the configured projector; it does not claim to solve the
    best representation in the complete auxiliary output space.
    """

    feasible: list[torch.Tensor]
    recovery_iou: list[torch.Tensor]
    centroid_distance: list[torch.Tensor]
    reason_code: list[torch.Tensor]
    component_ids: list[torch.Tensor]
    strides: tuple[int, ...]
    min_iou: float
    max_centroid_distance: float


class TaskConsistentProjectionGraph:
    """Build a binary graph from max-pool projection and task-space recovery.

    A projected component is usable only when its overlap relation with the
    native instance map is one-to-one.  Consequently, every participant in a
    merge or split is rejected; feasibility cannot depend on component order.
    """

    def __init__(
        self,
        strides: tuple[int, ...] = (1, 2, 4, 8),
        min_iou: float = 0.5,
        max_centroid_distance: float = 3.0,
        connectivity: int = 2,
    ) -> None:
        if not strides or min(strides) < 1 or len(set(strides)) != len(strides):
            raise ValueError("strides must be non-empty, positive, and unique")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be in [0, 1]")
        if max_centroid_distance <= 0:
            raise ValueError("max_centroid_distance must be positive")
        if connectivity not in (1, 2):
            raise ValueError("connectivity must be 1 or 2")
        self.strides = tuple(int(value) for value in strides)
        self.min_iou = float(min_iou)
        self.max_centroid_distance = float(max_centroid_distance)
        self.connectivity = int(connectivity)

    @torch.no_grad()
    def __call__(self, instance_map: torch.Tensor) -> TaskConsistentAssignment:
        if instance_map.ndim == 4 and instance_map.shape[1] == 1:
            instance_map = instance_map[:, 0]
        if instance_map.ndim != 3:
            raise ValueError("instance_map must have shape [B,H,W] or [B,1,H,W]")

        feasible_rows: list[torch.Tensor] = []
        iou_rows: list[torch.Tensor] = []
        distance_rows: list[torch.Tensor] = []
        reason_rows: list[torch.Tensor] = []
        component_ids: list[torch.Tensor] = []

        for single_map in instance_map.long():
            ids = torch.unique(single_map)
            ids = ids[ids > 0]
            component_ids.append(ids)
            shape = (int(ids.numel()), len(self.strides))
            feasible = torch.zeros(shape, device=instance_map.device, dtype=torch.bool)
            ious = torch.zeros(shape, device=instance_map.device, dtype=torch.float32)
            distances = torch.full(
                shape,
                float("inf"),
                device=instance_map.device,
                dtype=torch.float32,
            )
            reasons = torch.full(
                shape,
                REASON_DISAPPEARED,
                device=instance_map.device,
                dtype=torch.int8,
            )

            native_np = single_map.detach().cpu().numpy().astype(np.int64, copy=False)
            for side_index, stride in enumerate(self.strides):
                projected = self._project_and_lift(single_map > 0, stride)
                projected_np = measure.label(
                    projected.detach().cpu().numpy().astype(bool),
                    connectivity=self.connectivity,
                    background=0,
                ).astype(np.int64, copy=False)
                self._score_scale(
                    native_np,
                    projected_np,
                    ids,
                    side_index,
                    feasible,
                    ious,
                    distances,
                    reasons,
                )

            feasible_rows.append(feasible)
            iou_rows.append(ious)
            distance_rows.append(distances)
            reason_rows.append(reasons)

        return TaskConsistentAssignment(
            feasible=feasible_rows,
            recovery_iou=iou_rows,
            centroid_distance=distance_rows,
            reason_code=reason_rows,
            component_ids=component_ids,
            strides=self.strides,
            min_iou=self.min_iou,
            max_centroid_distance=self.max_centroid_distance,
        )

    @staticmethod
    def _project_and_lift(mask: torch.Tensor, stride: int) -> torch.Tensor:
        if stride == 1:
            return mask.bool()
        pooled = F.max_pool2d(
            mask.float().view(1, 1, *mask.shape),
            kernel_size=stride,
            stride=stride,
        )
        lifted = F.interpolate(pooled, size=mask.shape, mode="nearest")
        return lifted[0, 0] > 0.5

    def _score_scale(
        self,
        native: np.ndarray,
        projected: np.ndarray,
        ids: torch.Tensor,
        side_index: int,
        feasible: torch.Tensor,
        ious: torch.Tensor,
        distances: torch.Tensor,
        reasons: torch.Tensor,
    ) -> None:
        for row, component_id_tensor in enumerate(ids):
            component_id = int(component_id_tensor.detach().cpu())
            native_component = native == component_id
            overlapping_projected = np.unique(projected[native_component])
            overlapping_projected = overlapping_projected[overlapping_projected > 0]
            if overlapping_projected.size == 0:
                reasons[row, side_index] = REASON_DISAPPEARED
                continue
            if overlapping_projected.size > 1:
                reasons[row, side_index] = REASON_SPLIT
                continue

            projected_id = int(overlapping_projected[0])
            projected_component = projected == projected_id
            overlapping_native = np.unique(native[projected_component])
            overlapping_native = overlapping_native[overlapping_native > 0]
            if overlapping_native.size != 1 or int(overlapping_native[0]) != component_id:
                reasons[row, side_index] = REASON_MERGED
                continue

            intersection = int(np.logical_and(native_component, projected_component).sum())
            union = int(np.logical_or(native_component, projected_component).sum())
            iou = float(intersection) / float(max(1, union))
            native_centroid = np.argwhere(native_component).mean(axis=0)
            projected_centroid = np.argwhere(projected_component).mean(axis=0)
            distance = float(np.linalg.norm(native_centroid - projected_centroid))
            ious[row, side_index] = iou
            distances[row, side_index] = distance

            if iou < self.min_iou:
                reasons[row, side_index] = REASON_LOW_IOU
            elif distance >= self.max_centroid_distance:
                reasons[row, side_index] = REASON_LOCALIZATION
            else:
                feasible[row, side_index] = True
                reasons[row, side_index] = REASON_FEASIBLE


class TaskConsistentPartialTargetBuilder:
    """Construct side targets with unknown taking precedence over positive."""

    def __init__(self, strides: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        self.strides = tuple(int(value) for value in strides)

    @torch.no_grad()
    def __call__(
        self,
        instance_map: torch.Tensor,
        assignment: TaskConsistentAssignment,
        side_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if instance_map.ndim == 4 and instance_map.shape[1] == 1:
            instance_map = instance_map[:, 0]
        if instance_map.ndim != 3:
            raise ValueError("instance_map must have shape [B,H,W] or [B,1,H,W]")
        if side_index < 0 or side_index >= len(self.strides):
            raise ValueError("side_index out of range")

        positive = torch.zeros_like(instance_map, dtype=torch.bool)
        unknown = torch.zeros_like(instance_map, dtype=torch.bool)
        for batch_index in range(instance_map.shape[0]):
            ids = assignment.component_ids[batch_index]
            graph = assignment.feasible[batch_index]
            for row, component_id in enumerate(ids):
                component = instance_map[batch_index] == component_id
                if bool(graph[row, side_index]):
                    positive[batch_index] |= component
                else:
                    unknown[batch_index] |= component

        target = positive.float().unsqueeze(1)
        unknown_map = unknown.float().unsqueeze(1)
        stride = self.strides[side_index]
        if stride > 1:
            target = F.max_pool2d(target, kernel_size=stride, stride=stride)
            unknown_map = F.max_pool2d(
                unknown_map,
                kernel_size=stride,
                stride=stride,
            )

        target = target > 0.5
        valid = unknown_map <= 0.5
        # A cell that contains both a feasible and an infeasible identity is
        # task-ambiguous.  Unknown therefore wins and the positive is removed.
        target = target & valid
        active = target.flatten(1).any(dim=1)
        return target.float(), valid.float(), active


__all__ = [
    "REASON_DISAPPEARED",
    "REASON_FEASIBLE",
    "REASON_LOCALIZATION",
    "REASON_LOW_IOU",
    "REASON_MERGED",
    "REASON_SPLIT",
    "TaskConsistentAssignment",
    "TaskConsistentPartialTargetBuilder",
    "TaskConsistentProjectionGraph",
]
