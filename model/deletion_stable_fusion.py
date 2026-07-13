"""Deletion-Stable Fusion (DSF) for canonical MSHNet.

The original affine fusion accepts a positive decision even when it collapses
after deleting one native scale.  DSF evaluates the four exact leave-one-scale
out logits and replaces the unrestricted sum by their normalized soft minimum:

    z_dsf = -log(mean_i exp(-(z - c_i))).

This is a single robust fusion operator.  It preserves every native feature,
head, kernel, and checkpoint parameter, but makes the deployed decision train
against its most fragile one-scale deletion.  The derivative with respect to
the deletion logits is a simplex responsibility distribution concentrated on
the least stable coalition.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.orthogonal_scale_ownership import native_fusion_contributions


def normalized_soft_min(value: Tensor, dim: int) -> Tensor:
    """Smooth minimum that is exact when all entries are equal."""

    count = value.shape[dim]
    if count < 1:
        raise ValueError("soft minimum requires a non-empty dimension")
    return -torch.logsumexp(-value, dim=dim, keepdim=True) + math.log(count)


def deletion_stable_fusion(
    scale_logits: Tensor,
    fusion,
) -> dict[str, Tensor]:
    """Return the affine and deletion-stable fusion states."""

    contributions = native_fusion_contributions(scale_logits, fusion)
    affine = contributions.sum(dim=1, keepdim=True)
    if fusion.bias is not None:
        affine = affine + fusion.bias.view(1, 1, 1, 1)
    deletion_logits = affine - contributions
    pred = normalized_soft_min(deletion_logits, dim=1)
    responsibility = torch.softmax(-deletion_logits, dim=1)
    return {
        "contributions": contributions,
        "affine_pred": affine,
        "deletion_logits": deletion_logits,
        "deletion_responsibility": responsibility,
        "pred": pred,
    }


class DeletionStableFusionMSHNet(DeterministicMSHNet):
    """MSHNet whose final decision must survive every single-scale deletion."""

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_deletion_state: bool = False,
    ) -> tuple[list[Tensor], Tensor] | dict[str, Any]:
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], dim=1))

        if not warm_flag:
            pred = self.output_0(x_d0)
            if return_deletion_state:
                raise ValueError("deletion state requires the four-scale path")
            return [], pred

        side_logits = [
            self.output_0(x_d0),
            self.output_1(x_d1),
            self.output_2(x_d2),
            self.output_3(x_d3),
        ]
        scale_logits = torch.cat(
            [
                side_logits[0],
                self.up(side_logits[1]),
                self.up_4(side_logits[2]),
                self.up_8(side_logits[3]),
            ],
            dim=1,
        )
        state = deletion_stable_fusion(scale_logits, self.final)
        if return_deletion_state:
            return {
                "side_logits": tuple(side_logits),
                "scale_logits": scale_logits,
                **state,
            }
        return side_logits, state["pred"]


DSFMSHNet = DeletionStableFusionMSHNet


__all__ = [
    "DSFMSHNet",
    "DeletionStableFusionMSHNet",
    "deletion_stable_fusion",
    "normalized_soft_min",
]
