from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ResolutionOwnedAssignment:
    responsibilities: list[torch.Tensor]
    primary_owner: list[torch.Tensor]
    decidability: list[torch.Tensor]
    utility: list[torch.Tensor]
    component_ids: list[torch.Tensor]
    strides: tuple[int, ...]
    min_decidability: float


class ResolutionDecidableSupervisionGraph:
    """Build an instance-to-side-head supervision graph from GT geometry."""

    def __init__(
        self,
        strides: tuple[int, ...] = (1, 2, 4, 8),
        preferred_diameter_cells: float = 3.0,
        sigma: float = 0.75,
        min_decidability: float = 0.25,
        interval_ratio: float = 0.5,
        mode: str = "interval",
        fallback: str = "side0",
        area_min_cells: float = 2.0,
        quantization_tau: float = 1.0,
    ) -> None:
        if mode not in {"interval", "hard", "random", "area_only"}:
            raise ValueError("mode must be interval, hard, random, or area_only")
        if fallback not in {"side0", "final_only"}:
            raise ValueError("fallback must be side0 or final_only")
        if min(strides) < 1 or len(set(strides)) != len(strides):
            raise ValueError("strides must be positive and unique")
        self.strides = tuple(int(value) for value in strides)
        self.preferred_diameter_cells = float(preferred_diameter_cells)
        self.sigma = float(sigma)
        self.min_decidability = float(min_decidability)
        self.interval_ratio = float(interval_ratio)
        self.mode = mode
        self.fallback = fallback
        self.area_min_cells = float(area_min_cells)
        self.quantization_tau = float(quantization_tau)

    @torch.no_grad()
    def __call__(self, instance_map: torch.Tensor) -> ResolutionOwnedAssignment:
        if instance_map.ndim == 4 and instance_map.shape[1] == 1:
            instance_map = instance_map[:, 0]
        if instance_map.ndim != 3:
            raise ValueError("instance_map must have shape [B,H,W] or [B,1,H,W]")

        responsibilities: list[torch.Tensor] = []
        primary_owner: list[torch.Tensor] = []
        decidability: list[torch.Tensor] = []
        utility: list[torch.Tensor] = []
        component_ids: list[torch.Tensor] = []
        for single_map in instance_map.long():
            ids = torch.unique(single_map)
            ids = ids[ids > 0]
            component_ids.append(ids)
            if ids.numel() == 0:
                empty_2d = torch.zeros(
                    (0, len(self.strides)),
                    device=instance_map.device,
                    dtype=torch.float32,
                )
                responsibilities.append(empty_2d)
                decidability.append(empty_2d.clone())
                utility.append(empty_2d.clone())
                primary_owner.append(
                    torch.zeros((0,), device=instance_map.device, dtype=torch.long)
                )
                continue

            q_rows = []
            u_rows = []
            for component_id in ids:
                component = single_map == component_id
                q, u = self._component_scores(component, single_map)
                q_rows.append(q)
                u_rows.append(u)
            q_matrix = torch.stack(q_rows, dim=0)
            u_matrix = torch.stack(u_rows, dim=0)
            r_matrix, owner = self._assign(q_matrix, u_matrix)

            responsibilities.append(r_matrix)
            primary_owner.append(owner)
            decidability.append(q_matrix)
            utility.append(u_matrix)

        return ResolutionOwnedAssignment(
            responsibilities=responsibilities,
            primary_owner=primary_owner,
            decidability=decidability,
            utility=utility,
            component_ids=component_ids,
            strides=self.strides,
            min_decidability=self.min_decidability,
        )

    def _component_scores(
        self,
        component: torch.Tensor,
        instance_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = component.device
        dtype = torch.float32
        area = component.sum().to(dtype=dtype).clamp_min(1.0)
        diameter = 2.0 * torch.sqrt(area / torch.tensor(torch.pi, device=device))
        centroid_yx = self._centroid(component)

        q_values = []
        u_values = []
        for stride in self.strides:
            pooled = self._pool_bool(component, stride)
            pooled_area = pooled.sum().to(dtype=dtype)
            q_area = torch.minimum(
                torch.ones((), device=device),
                pooled_area / max(self.area_min_cells, 1e-6),
            )
            quant_error = self._quantization_error(centroid_yx, stride)
            q_quant = torch.exp(-quant_error / max(self.quantization_tau, 1e-6))
            collision = self._has_collision(component, instance_map, stride)
            q_merge = torch.zeros((), device=device) if collision else torch.ones((), device=device)

            if self.mode == "area_only":
                q = torch.ones((), device=device) if pooled_area > 0 else torch.zeros((), device=device)
            else:
                q = q_area * q_quant * q_merge

            rho = diameter / float(stride)
            log_delta = torch.log2(rho + 1e-6) - torch.log2(
                torch.tensor(self.preferred_diameter_cells, device=device)
            )
            preference = torch.exp(-(log_delta.square()) / (2.0 * self.sigma * self.sigma))
            utility = q * preference
            q_values.append(q)
            u_values.append(utility)
        return torch.stack(q_values), torch.stack(u_values)

    @staticmethod
    def _centroid(component: torch.Tensor) -> torch.Tensor:
        ys, xs = torch.where(component)
        if ys.numel() == 0:
            return torch.zeros((2,), device=component.device, dtype=torch.float32)
        return torch.stack([ys.float().mean(), xs.float().mean()])

    @staticmethod
    def _pool_bool(mask: torch.Tensor, stride: int) -> torch.Tensor:
        if stride == 1:
            return mask
        pooled = F.max_pool2d(
            mask.float().view(1, 1, *mask.shape),
            kernel_size=stride,
            stride=stride,
        )
        return pooled[0, 0] > 0.5

    @staticmethod
    def _quantization_error(centroid_yx: torch.Tensor, stride: int) -> torch.Tensor:
        if stride == 1:
            return torch.zeros((), device=centroid_yx.device)
        cell = torch.floor(centroid_yx / float(stride))
        center = cell * float(stride) + (float(stride) - 1.0) / 2.0
        return torch.linalg.vector_norm(center - centroid_yx)

    @staticmethod
    def _has_collision(
        component: torch.Tensor,
        instance_map: torch.Tensor,
        stride: int,
    ) -> bool:
        if stride == 1:
            return False
        pooled_component = ResolutionDecidableSupervisionGraph._pool_bool(
            component, stride
        )
        occupied = F.max_pool2d(
            (instance_map > 0).float().view(1, 1, *instance_map.shape),
            kernel_size=stride,
            stride=stride,
        )[0, 0] > 0.5
        if not bool((pooled_component & occupied).any()):
            return False

        cell_indices = torch.where(pooled_component)
        for y, x in zip(cell_indices[0], cell_indices[1]):
            patch = instance_map[
                int(y) * stride : (int(y) + 1) * stride,
                int(x) * stride : (int(x) + 1) * stride,
            ]
            ids = torch.unique(patch)
            ids = ids[ids > 0]
            if ids.numel() > 1:
                return True
        return False

    def _assign(
        self,
        decidability: torch.Tensor,
        utility: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        active = (decidability >= self.min_decidability) & (utility > 0)
        if self.mode in {"interval", "area_only"}:
            max_utility = utility.max(dim=1, keepdim=True).values
            active = active & (utility >= self.interval_ratio * max_utility.clamp_min(1e-12))
        elif self.mode in {"hard", "random"}:
            hard_active = torch.zeros_like(active)
            for row in range(active.shape[0]):
                valid = torch.where(active[row])[0]
                if valid.numel() > 0:
                    if self.mode == "random":
                        chosen = valid[torch.randint(valid.numel(), (1,), device=valid.device)[0]]
                    else:
                        chosen = valid[torch.argmax(utility[row, valid])]
                    hard_active[row, chosen] = True
            active = hard_active

        if self.mode == "random":
            # The branch above chooses one valid scale.  If no scale is valid,
            # fallback below mirrors the deterministic modes.
            pass

        owner = torch.full(
            (active.shape[0],),
            -1,
            device=utility.device,
            dtype=torch.long,
        )
        for row in range(active.shape[0]):
            valid = torch.where(active[row])[0]
            if valid.numel() == 0:
                if self.fallback == "side0":
                    active[row, 0] = True
                    owner[row] = 0
                continue
            owner[row] = valid[torch.argmax(utility[row, valid])]

        responsibilities = active.to(dtype=torch.float32)
        row_sum = responsibilities.sum(dim=1, keepdim=True).clamp_min(1.0)
        responsibilities = responsibilities / row_sum
        return responsibilities, owner


class OwnedSideSupervisionBuilder:
    def __init__(
        self,
        strides: tuple[int, ...] = (1, 2, 4, 8),
        ignore_dilation: int = 3,
    ) -> None:
        if ignore_dilation < 1 or ignore_dilation % 2 == 0:
            raise ValueError("ignore_dilation must be a positive odd integer")
        self.strides = tuple(int(value) for value in strides)
        self.ignore_dilation = int(ignore_dilation)

    @torch.no_grad()
    def __call__(
        self,
        instance_map: torch.Tensor,
        assignment: ResolutionOwnedAssignment,
        side_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if instance_map.ndim == 4 and instance_map.shape[1] == 1:
            instance_map = instance_map[:, 0]
        if instance_map.ndim != 3:
            raise ValueError("instance_map must have shape [B,H,W] or [B,1,H,W]")
        if side_index < 0 or side_index >= len(self.strides):
            raise ValueError("side_index out of range")

        assigned = torch.zeros_like(instance_map, dtype=torch.bool)
        unassigned = torch.zeros_like(instance_map, dtype=torch.bool)
        active = torch.zeros(
            (instance_map.shape[0],),
            device=instance_map.device,
            dtype=torch.bool,
        )
        positive_weights = torch.zeros_like(instance_map, dtype=torch.float32)

        for batch_index in range(instance_map.shape[0]):
            ids = assignment.component_ids[batch_index]
            resp = assignment.responsibilities[batch_index]
            for row, component_id in enumerate(ids):
                component = instance_map[batch_index] == component_id
                if bool(resp[row, side_index] > 0):
                    assigned[batch_index] |= component
                    positive_weights[batch_index][component] = resp[
                        row,
                        side_index,
                    ].float()
                    active[batch_index] = True
                else:
                    unassigned[batch_index] |= component

        target = assigned.float().unsqueeze(1)
        ignore = unassigned.float().unsqueeze(1)
        positive_weight = positive_weights.unsqueeze(1)
        stride = self.strides[side_index]
        if stride > 1:
            target = F.max_pool2d(target, stride, stride)
            ignore = F.max_pool2d(ignore, stride, stride)
            positive_weight = F.max_pool2d(positive_weight, stride, stride)

        if self.ignore_dilation > 1:
            padding = self.ignore_dilation // 2
            ignore = F.max_pool2d(
                ignore,
                kernel_size=self.ignore_dilation,
                stride=1,
                padding=padding,
            )

        target = (target > 0.5).float()
        valid = (ignore < 0.5).float()
        valid = torch.maximum(valid, target)
        weight = torch.where(
            target > 0,
            positive_weight.clamp_min(1e-6),
            torch.ones_like(target),
        )
        return target, valid, weight, active


__all__ = [
    "OwnedSideSupervisionBuilder",
    "ResolutionDecidableSupervisionGraph",
    "ResolutionOwnedAssignment",
]
