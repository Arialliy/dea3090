"""Complete SDR-MSHNet with an unchanged canonical deployment graph.

The encoder, decoder, side heads, and final affine fusion are exactly the
deterministic canonical MSHNet.  During training, this class can additionally
expose the native fusion contributions and deletion logits required by the
single Scale-Deletion Responsibility Refinement (SDRR) objective.  No module,
parameter, or inference operation is added.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.counterfactual_responsibility import (
    counterfactual_responsibility_suppression,
    responsibility_conserving_gradient_routing,
    responsibility_density_risk,
)
from model.scale_coalition_supervision import leave_one_scale_out_coalitions


class SDRMSHNet(DeterministicMSHNet):
    """Parameter-identical MSHNet with a training-only responsibility view."""

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_responsibility_state: bool = False,
    ) -> tuple[list[Tensor], Tensor] | dict[str, Any]:
        side_logits, pred = super().forward(x, warm_flag)
        if not return_responsibility_state:
            return side_logits, pred
        if not warm_flag:
            raise ValueError(
                "responsibility state requires the four-scale warm path"
            )

        coalition = leave_one_scale_out_coalitions(
            side_logits, pred, self.final
        )
        return {
            "side_logits": tuple(side_logits),
            "pred": pred,
            "scale_logits": coalition["scale_logits"],
            "contributions": coalition["contributions"],
            "deletion_logits": coalition["coalition_logits"],
            "reconstructed": coalition["reconstructed"],
        }

    def responsibility_objective(
        self,
        state: dict[str, Any],
        target: Tensor,
        *,
        safe_kernel: int = 15,
        normalization: str = "event",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the sole SDRR auxiliary objective from a forward state."""

        pred = state.get("pred")
        contributions = state.get("contributions")
        if not isinstance(pred, Tensor) or not isinstance(contributions, Tensor):
            raise ValueError(
                "state must be produced by return_responsibility_state=True"
            )
        return counterfactual_responsibility_suppression(
            pred,
            contributions,
            target,
            safe_kernel=safe_kernel,
            normalization=normalization,
        )

    def responsibility_routing_objective(
        self,
        state: dict[str, Any],
        target: Tensor,
        *,
        safe_kernel: int = 15,
        normalization: str = "unique_pixel",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute responsibility-conserving decision-gradient routing."""

        pred = state.get("pred")
        contributions = state.get("contributions")
        if not isinstance(pred, Tensor) or not isinstance(contributions, Tensor):
            raise ValueError(
                "state must be produced by return_responsibility_state=True"
            )
        return responsibility_conserving_gradient_routing(
            pred,
            contributions,
            target,
            safe_kernel=safe_kernel,
            normalization=normalization,
        )

    def responsibility_density_objective(
        self,
        state: dict[str, Any],
        target: Tensor,
        *,
        safe_kernel: int = 15,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute the final Responsibility Density Risk objective."""

        pred = state.get("pred")
        contributions = state.get("contributions")
        if not isinstance(pred, Tensor) or not isinstance(contributions, Tensor):
            raise ValueError(
                "state must be produced by return_responsibility_state=True"
            )
        return responsibility_density_risk(
            pred,
            contributions,
            target,
            safe_kernel=safe_kernel,
        )


__all__ = ["SDRMSHNet"]
