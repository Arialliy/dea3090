"""Measure-conditioned SLS for targets with and without foreground mass."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.loss import SLSIoULoss
from model.partial_sls_loss import PartialSLSIoULoss


class MeasureConditionedSLSIoULoss(nn.Module):
    """Use canonical SLS on non-null masks and proper null risk otherwise.

    A location functional is undefined for a zero-mass target.  Canonical SLS
    nevertheless evaluates its location term there, producing a spatially
    signed gradient unrelated to background suppression.  This loss preserves
    canonical SLS exactly when every sample is non-null and replaces only null
    samples by a threshold-consistent false-alarm barrier.  The barrier is
    active only where a null image predicts foreground (logit > 0), matching
    the evaluation decision boundary without a dense constant background
    force that can overwhelm sparse positive targets.
    """

    def __init__(self, null_mode: str = "false_alarm_barrier") -> None:
        super().__init__()
        if null_mode not in ("false_alarm_barrier", "abstain"):
            raise ValueError("null_mode must be false_alarm_barrier or abstain")
        self.null_mode = null_mode
        self.canonical = SLSIoULoss()
        self.per_sample = PartialSLSIoULoss()

    def forward(
        self,
        pred_log: torch.Tensor,
        target: torch.Tensor,
        warm_epoch: int,
        epoch: int,
    ) -> torch.Tensor:
        if pred_log.shape != target.shape:
            raise ValueError("pred_log and target shapes must match")
        non_null = target.flatten(1).sum(dim=1) > 0
        if bool(non_null.all().detach().cpu()):
            return self.canonical(pred_log, target, warm_epoch, epoch)

        if self.null_mode == "abstain":
            null_risk = pred_log.flatten(1).sum(dim=1) * 0.0
        else:
            null_risk = F.relu(pred_log).square().flatten(1).mean(dim=1)
        if not bool(non_null.any().detach().cpu()):
            return null_risk.mean()

        valid = torch.ones_like(target)
        canonical_terms = self.per_sample(
            pred_log,
            target,
            valid,
            warm_epoch,
            epoch,
            reduction="none",
        )
        return torch.where(non_null, canonical_terms, null_risk).mean()


__all__ = ["MeasureConditionedSLSIoULoss"]
