"""Component-level FROC metrics for infrared small-target detection.

The existing repository ``FA`` metric measures the area of unmatched
prediction components per image area.  That does not directly answer how many
spurious target-like objects remain.  This module instead reports false
positive components per image (FPPI) against target-instance detection
probability over a fixed probability-threshold grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from utils.metric import match_connected_components


DEFAULT_COMPONENT_BUDGETS: tuple[float, ...] = (
    0.01,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
)


@dataclass(frozen=True)
class ComponentFROCCurve:
    thresholds: np.ndarray
    threshold_space: str
    detection_probability: np.ndarray
    false_positive_components_per_image: np.ndarray
    num_images: int
    num_targets: int


@dataclass(frozen=True)
class ComponentBudgetPoint:
    budget: float
    detection_probability: float
    threshold: float
    achieved_fppi: float


class ComponentFROC:
    """Accumulate a fixed-grid component FROC without storing predictions."""

    def __init__(
        self,
        *,
        thresholds: Sequence[float] | None = None,
        num_thresholds: int = 51,
        max_centroid_distance: float = 3.0,
        threshold_space: str = "probability",
    ) -> None:
        if threshold_space not in ("probability", "logit"):
            raise ValueError("threshold_space must be probability or logit")
        if thresholds is None:
            if num_thresholds < 2:
                raise ValueError("num_thresholds must be at least 2")
            bounds = (-20.0, 160.0) if threshold_space == "logit" else (0.0, 1.0)
            thresholds = np.linspace(*bounds, num_thresholds).tolist()
        values = np.asarray(tuple(float(x) for x in thresholds), dtype=np.float64)
        if values.ndim != 1 or values.size < 2:
            raise ValueError("thresholds must be a one-dimensional sequence")
        if threshold_space == "probability" and (
            np.any(values < 0.0) or np.any(values > 1.0)
        ):
            raise ValueError("probability thresholds must lie in [0, 1]")
        if threshold_space == "logit" and not np.all(np.isfinite(values)):
            raise ValueError("logit thresholds must be finite")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("thresholds must be strictly increasing")
        if max_centroid_distance <= 0.0:
            raise ValueError("max_centroid_distance must be positive")
        self.thresholds = values
        self.threshold_space = threshold_space
        if threshold_space == "logit":
            self.decision_thresholds = values.copy()
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                self.decision_thresholds = np.log(values) - np.log1p(-values)
        self.max_centroid_distance = float(max_centroid_distance)
        self.matched_targets = np.zeros(values.size, dtype=np.int64)
        self.false_positive_components = np.zeros(values.size, dtype=np.int64)
        self.num_images = 0
        self.num_targets = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logit_array = logits.detach().cpu().numpy()
        label_array = labels.detach().cpu().numpy()
        if logit_array.ndim == 3:
            logit_array = logit_array[:, None, :, :]
        if label_array.ndim == 3:
            label_array = label_array[:, None, :, :]
        if logit_array.ndim != 4 or label_array.ndim != 4:
            raise ValueError("logits and labels must be BCHW or BHW tensors")
        if logit_array.shape != label_array.shape:
            raise ValueError("logits and labels must have equal shape")

        for batch_index in range(logit_array.shape[0]):
            image_logits = logit_array[batch_index, 0]
            target = (label_array[batch_index, 0] > 0.5).astype(np.int64)
            # Target count is threshold-independent.  Reuse the result from
            # the first threshold to keep the definition exactly aligned with
            # the repository's established component matching rule.
            target_count: int | None = None
            for index, threshold in enumerate(self.decision_thresholds):
                # Compare in raw-logit space.  Computing float32 sigmoid first
                # collapses all sufficiently confident scores to exactly one
                # and destroys the ranking needed at very low FPPI budgets.
                prediction = (image_logits > threshold).astype(np.int64)
                match = match_connected_components(
                    prediction,
                    target,
                    max_centroid_distance=self.max_centroid_distance,
                )
                if target_count is None:
                    target_count = len(match.target_regions)
                self.matched_targets[index] += len(match.matches)
                self.false_positive_components[index] += len(
                    match.unmatched_prediction_indices
                )
            self.num_targets += int(target_count or 0)
            self.num_images += 1

    def get_curve(self) -> ComponentFROCCurve:
        if self.num_images < 1:
            raise ValueError("component FROC has no images")
        detection_probability = np.divide(
            self.matched_targets,
            self.num_targets,
            out=np.zeros_like(self.matched_targets, dtype=np.float64),
            where=self.num_targets != 0,
        )
        fppi = self.false_positive_components.astype(np.float64) / self.num_images
        return ComponentFROCCurve(
            thresholds=self.thresholds.copy(),
            threshold_space=self.threshold_space,
            detection_probability=detection_probability,
            false_positive_components_per_image=fppi,
            num_images=self.num_images,
            num_targets=self.num_targets,
        )

    def at_budgets(
        self,
        budgets: Iterable[float] = DEFAULT_COMPONENT_BUDGETS,
    ) -> tuple[ComponentBudgetPoint, ...]:
        curve = self.get_curve()
        points: list[ComponentBudgetPoint] = []
        for raw_budget in budgets:
            budget = float(raw_budget)
            if budget < 0.0:
                raise ValueError("component budgets must be non-negative")
            feasible = np.flatnonzero(
                curve.false_positive_components_per_image <= budget + 1e-12
            )
            if feasible.size == 0:
                # A threshold of exactly one should normally make this branch
                # unreachable, but fail closed for custom grids.
                points.append(ComponentBudgetPoint(budget, 0.0, 1.0, np.inf))
                continue
            feasible_pd = curve.detection_probability[feasible]
            best_pd = float(feasible_pd.max())
            best = feasible[np.flatnonzero(feasible_pd == best_pd)]
            # When PD ties, use the point with fewer false components; when
            # that also ties, use the lower threshold for determinism.
            best_fppi = curve.false_positive_components_per_image[best]
            best = best[best_fppi == best_fppi.min()]
            selected = int(best[0])
            points.append(
                ComponentBudgetPoint(
                    budget=budget,
                    detection_probability=best_pd,
                    threshold=float(curve.thresholds[selected]),
                    achieved_fppi=float(
                        curve.false_positive_components_per_image[selected]
                    ),
                )
            )
        return tuple(points)

    def mean_low_budget_detection(
        self,
        budgets: Iterable[float] = DEFAULT_COMPONENT_BUDGETS,
    ) -> float:
        points = self.at_budgets(budgets)
        if not points:
            raise ValueError("at least one component budget is required")
        return float(np.mean([point.detection_probability for point in points]))


__all__ = [
    "ComponentBudgetPoint",
    "ComponentFROC",
    "ComponentFROCCurve",
    "DEFAULT_COMPONENT_BUDGETS",
]
