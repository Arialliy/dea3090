"""Decision-Conditional Deletion Fusion (DCDF) for MSHNet.

Uniform worst-coalition fusion is unnecessarily conservative.  DCDF applies
deletion stability only in the decision regime that motivated the method: the
full affine fusion tends positive while the smooth weakest deletion coalition
tends negative.  Stable positive decisions and negative decisions retain the
native affine prediction up to a vanishing smooth gate.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from model.deletion_stable_fusion import DSFMSHNet


def decision_conditional_deletion_fusion(
    affine_pred: Tensor,
    robust_pred: Tensor,
) -> dict[str, Tensor]:
    """Smoothly correct only positive-to-negative deletion fragility."""

    if affine_pred.shape != robust_pred.shape:
        raise ValueError("affine_pred and robust_pred must have equal shape")
    fragility_gap = F.relu(affine_pred - robust_pred)
    decision_gate = torch.sigmoid(affine_pred) * torch.sigmoid(-robust_pred)
    correction = decision_gate * fragility_gap
    return {
        "pred": affine_pred - correction,
        "fragility_gap": fragility_gap,
        "decision_gate": decision_gate,
        "deletion_correction": correction,
    }


class DecisionConditionalDeletionFusionMSHNet(DSFMSHNet):
    """MSHNet with a zero-parameter, decision-conditional robust fusion."""

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_deletion_state: bool = False,
    ) -> tuple[list[Tensor], Tensor] | dict[str, Any]:
        if not warm_flag:
            return super().forward(
                x,
                warm_flag,
                return_deletion_state=return_deletion_state,
            )
        state = super().forward(x, True, return_deletion_state=True)
        robust_pred = state["pred"]
        conditional = decision_conditional_deletion_fusion(
            state["affine_pred"], robust_pred
        )
        state["robust_pred"] = robust_pred
        state.update(conditional)
        if return_deletion_state:
            return state
        return list(state["side_logits"]), state["pred"]


DCDFMSHNet = DecisionConditionalDeletionFusionMSHNet


__all__ = [
    "DCDFMSHNet",
    "DecisionConditionalDeletionFusionMSHNet",
    "decision_conditional_deletion_fusion",
]
