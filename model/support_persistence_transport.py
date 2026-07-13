"""Support-persistence transport for MSHNet's native downsampling boundary.

Max pooling transmits a cell's strongest activation even when that activation
is owned by only one spatial site.  This file replaces that magnitude-only
rule with one counterfactual equation: compare the factual maximum with the
maximum after deleting its strongest site, then restore the deleted evidence
only when feature channels agree on where the spatial support lies.

The first design stage activates the same operator only at boundary 0.  It is
not an auxiliary branch and adds no convolutional block or prediction head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.baselines.mshnet_deterministic import MSHNet


class SupportPersistencePool2d(nn.Module):
    """Parameter-free channel-consensus survival law for 2x2 cells."""

    def forward(
        self, x: Tensor, *, return_state: bool = False
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4:
            raise ValueError("x must be a BCHW tensor")
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("spatial dimensions must be divisible by two")

        batch, channels, height, width = x.shape
        cells = F.unfold(x, kernel_size=2, stride=2)
        cells = cells.view(batch, channels, 4, height // 2, width // 2)
        mean = cells.mean(dim=2, keepdim=True)
        std = cells.std(dim=2, keepdim=True, unbiased=False)
        ownership = torch.softmax((cells - mean) / (std + 1e-6), dim=2)
        population = ownership.mean(dim=1, keepdim=True)
        agreement = (ownership * population).sum(dim=2)
        persistence = ((agreement - 0.25) / 0.75).clamp(0.0, 1.0)

        largest = torch.topk(cells, k=2, dim=2, sorted=True).values
        maximum = largest[:, :, 0]
        deleted_maximum = largest[:, :, 1]
        # Equal prior weight is assigned to factual survival and to measured
        # cross-channel persistence.  This prevents both complete deletion and
        # a learnable identity escape.
        gate = 0.5 * (1.0 + persistence)
        output = deleted_maximum + gate * (maximum - deleted_maximum)
        if not return_state:
            return output
        return output, {
            "maximum": maximum,
            "deleted_maximum": deleted_maximum,
            "single_site_ownership": maximum - deleted_maximum,
            "channel_persistence": persistence,
            "survival_gate": gate,
        }


class SupportPersistenceMSHNet(MSHNet):
    """Canonical MSHNet with one shared persistence law at selected boundaries."""

    def __init__(self, input_channels: int, *, active_stages: tuple[int, ...] = (0,)):
        super().__init__(input_channels)
        active_stages = tuple(int(stage) for stage in active_stages)
        if not active_stages or len(set(active_stages)) != len(active_stages):
            raise ValueError("active_stages must be non-empty and unique")
        if any(stage not in range(4) for stage in active_stages):
            raise ValueError("active_stages must be drawn from 0, 1, 2, 3")
        self.active_stages = active_stages
        self.support_persistence = SupportPersistencePool2d()

    def _transport(self, x: Tensor, stage: int) -> Tensor:
        if stage in self.active_stages:
            return self.support_persistence(x)
        return self.pool(x)

    def forward(self, x: Tensor, warm_flag: bool):
        e0 = self.encoder_0(self.conv_init(x))
        e1 = self.encoder_1(self._transport(e0, 0))
        e2 = self.encoder_2(self._transport(e1, 1))
        e3 = self.encoder_3(self._transport(e2, 2))
        middle = self.middle_layer(self._transport(e3, 3))

        d3 = self.decoder_3(torch.cat([e3, self.up(middle)], dim=1))
        d2 = self.decoder_2(torch.cat([e2, self.up(d3)], dim=1))
        d1 = self.decoder_1(torch.cat([e1, self.up(d2)], dim=1))
        d0 = self.decoder_0(torch.cat([e0, self.up(d1)], dim=1))

        if not warm_flag:
            return [], self.output_0(d0)
        masks = [
            self.output_0(d0),
            self.output_1(d1),
            self.output_2(d2),
            self.output_3(d3),
        ]
        output = self.final(
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
        return masks, output


__all__ = ["SupportPersistenceMSHNet", "SupportPersistencePool2d"]
