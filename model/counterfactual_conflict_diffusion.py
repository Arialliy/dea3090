"""Counterfactual Conflict-Field Diffusion (CCFD) for MSHNet.

CCFD retains the native affine prediction and treats its disagreement with the
smooth worst one-scale-deletion coalition as a spatial conflict field.  One
shared zero-DC stencil transports that conflict locally before it is returned
to the affine logit.  The eight stencil coefficients are the only new model
parameters and are initialized to zero, embedding the complete baseline
exactly while exposing a non-zero first-order learning direction.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from model.dea_shared_discrepancy_stencil import SharedDiscrepancyStencil
from model.deletion_stable_fusion import DSFMSHNet


def counterfactual_conflict_diffusion(
    affine_pred: Tensor,
    robust_pred: Tensor,
    stencil: SharedDiscrepancyStencil,
) -> dict[str, Tensor]:
    """Apply a mean-conserving local update to deletion conflict."""

    if affine_pred.shape != robust_pred.shape:
        raise ValueError("affine_pred and robust_pred must have equal shape")
    conflict = robust_pred - affine_pred
    correction_raw = stencil(conflict)
    correction = correction_raw - correction_raw.mean(
        dim=(-2, -1), keepdim=True
    )
    return {
        "pred": affine_pred + correction,
        "conflict_field": conflict,
        "conflict_correction": correction,
        "conflict_correction_raw": correction_raw,
    }


class CounterfactualConflictDiffusionMSHNet(DSFMSHNet):
    """MSHNet with a single shared counterfactual-conflict diffusion step."""

    def __init__(self, input_channels: int) -> None:
        super().__init__(input_channels)
        self.conflict_stencil = SharedDiscrepancyStencil()

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_conflict_state: bool = False,
    ) -> tuple[list[Tensor], Tensor] | dict[str, Any]:
        if not warm_flag:
            if return_conflict_state:
                raise ValueError("conflict state requires the four-scale path")
            return super().forward(x, False)
        state = super().forward(x, True, return_deletion_state=True)
        robust_pred = state["pred"]
        diffusion = counterfactual_conflict_diffusion(
            state["affine_pred"], robust_pred, self.conflict_stencil
        )
        state["robust_pred"] = robust_pred
        state.update(diffusion)
        if return_conflict_state:
            return state
        return list(state["side_logits"]), state["pred"]


CCFDMSHNet = CounterfactualConflictDiffusionMSHNet


__all__ = [
    "CCFDMSHNet",
    "CounterfactualConflictDiffusionMSHNet",
    "counterfactual_conflict_diffusion",
]
