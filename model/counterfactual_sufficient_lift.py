"""Counterfactual Sufficient Lift (CSL) for MSHNet downsampling.

Max pooling keeps the factual maximum ``m1`` but hides whether that value
survives deletion of its strongest spatial site.  In a 2x2 cell the deleted
maximum is the second order statistic ``m2`` and the exact exclusive evidence
is ``r = m1 - m2``.  CSL transports ``[m1, r]`` instead of suppressing ``m1``.

The following native encoder block is widened only at its input domain.  Its
new residual kernels are initialised to zero, so ``[W, 0] [m1, r] = W m1``:
the complete model initially embeds canonical max-pool MSHNet exactly.  CSL is
one sufficient-statistic replacement of a downsampling boundary, not an
attention branch, gate, auxiliary head, or extra loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.baselines.mshnet_deterministic import MSHNet, ResNet
from model.baseline_embedded_resnet import widen_resnet_input_with_zeros


class CounterfactualSufficientPool2d(nn.Module):
    """Lift each 2x2 cell to its factual maximum and deletion residual."""

    def forward(
        self,
        x: Tensor,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4:
            raise ValueError("x must be a BCHW tensor")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("spatial dimensions must be divisible by two")

        batch, channels, height, width = x.shape
        cells = F.unfold(x, kernel_size=2, stride=2)
        cells = cells.view(batch, channels, 4, height // 2, width // 2)
        largest = torch.topk(cells, k=2, dim=2, sorted=True).values
        factual_maximum = largest[:, :, 0]
        deleted_maximum = largest[:, :, 1]
        exclusive_residual = factual_maximum - deleted_maximum
        lifted = torch.cat([factual_maximum, exclusive_residual], dim=1)
        if not return_state:
            return lifted
        return lifted, {
            "factual_maximum": factual_maximum,
            "deleted_maximum": deleted_maximum,
            "exclusive_residual": exclusive_residual,
        }


def _copy_with_zero_residual_domain(source: ResNet) -> ResNet:
    """Widen one native ResNet input while exactly embedding its function."""
    return widen_resnet_input_with_zeros(
        source, 2 * source.conv1.in_channels
    )


class CounterfactualSufficientLiftMSHNet(MSHNet):
    """MSHNet with CSL at explicitly selected encoder boundaries."""

    _DOWNSTREAM_LAYERS = (
        "encoder_1",
        "encoder_2",
        "encoder_3",
        "middle_layer",
    )

    def __init__(
        self,
        input_channels: int,
        *,
        active_stages: tuple[int, ...] = (0,),
    ) -> None:
        super().__init__(input_channels)
        active_stages = tuple(int(stage) for stage in active_stages)
        if not active_stages or len(set(active_stages)) != len(active_stages):
            raise ValueError("active_stages must be non-empty and unique")
        if any(stage not in range(4) for stage in active_stages):
            raise ValueError("active_stages must be drawn from 0, 1, 2, 3")
        self.active_stages = active_stages
        self.counterfactual_lift = CounterfactualSufficientPool2d()

        for stage in active_stages:
            layer = getattr(self, self._DOWNSTREAM_LAYERS[stage])
            layer[0] = _copy_with_zero_residual_domain(layer[0])

    def load_canonical_state_dict(
        self,
        state_dict: dict[str, Tensor],
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load canonical MSHNet weights through the exact CSL embedding."""

        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[len("module.") :]: value for key, value in state_dict.items()
            }
        target_state = self.state_dict()
        widened = {
            f"{self._DOWNSTREAM_LAYERS[stage]}.0.{suffix}"
            for stage in self.active_stages
            for suffix in ("conv1.weight", "shortcut.0.weight")
        }
        embedded: dict[str, Tensor] = {}
        for key, target in target_state.items():
            if key not in state_dict:
                raise ValueError(f"canonical state is missing {key}")
            source = state_dict[key]
            if key in widened:
                if target.shape[1] != 2 * source.shape[1]:
                    raise ValueError(f"unexpected widened checkpoint shape for {key}")
                value = torch.zeros_like(target)
                value[:, : source.shape[1]].copy_(source)
                embedded[key] = value
            else:
                if target.shape != source.shape:
                    raise ValueError(f"canonical checkpoint shape differs at {key}")
                embedded[key] = source
        unexpected = sorted(set(state_dict).difference(target_state))
        if unexpected:
            raise ValueError(f"canonical state has unexpected keys: {unexpected[:5]}")
        return self.load_state_dict(embedded, strict=True)

    def _transport(self, x: Tensor, stage: int) -> Tensor:
        if stage in self.active_stages:
            return self.counterfactual_lift(x)
        return self.pool(x)

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_lift_state: bool = False,
    ):
        e0 = self.encoder_0(self.conv_init(x))
        if return_lift_state and 0 in self.active_stages:
            transport0, lift_state = self.counterfactual_lift(
                e0, return_state=True
            )
        else:
            transport0 = self._transport(e0, 0)
            lift_state = None
        e1 = self.encoder_1(transport0)
        e2 = self.encoder_2(self._transport(e1, 1))
        e3 = self.encoder_3(self._transport(e2, 2))
        middle = self.middle_layer(self._transport(e3, 3))

        d3 = self.decoder_3(torch.cat([e3, self.up(middle)], dim=1))
        d2 = self.decoder_2(torch.cat([e2, self.up(d3)], dim=1))
        d1 = self.decoder_1(torch.cat([e1, self.up(d2)], dim=1))
        d0 = self.decoder_0(torch.cat([e0, self.up(d1)], dim=1))

        if not warm_flag:
            masks: list[Tensor] = []
            prediction = self.output_0(d0)
        else:
            masks = [
                self.output_0(d0),
                self.output_1(d1),
                self.output_2(d2),
                self.output_3(d3),
            ]
            prediction = self.final(
                torch.cat(
                    [
                        masks[0],
                        self.up(masks[1]),
                        self.up_4(masks[2]),
                        self.up_8(masks[3]),
                    ],
                    dim=1,
                )
            )
        if not return_lift_state:
            return masks, prediction
        return {
            "masks": masks,
            "pred": prediction,
            "stage0_lift": lift_state,
        }


CSLMSHNet = CounterfactualSufficientLiftMSHNet


__all__ = [
    "CSLMSHNet",
    "CounterfactualSufficientLiftMSHNet",
    "CounterfactualSufficientPool2d",
]
