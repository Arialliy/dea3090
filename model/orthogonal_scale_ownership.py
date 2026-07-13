"""Orthogonal Scale Ownership (OSO) fusion for canonical MSHNet.

MSHNet asks four native scale heads to predict the same mask and then mixes
their logits with an unconstrained convolution.  OSO keeps the backbone,
decoder, side heads, and fusion parameters, but changes the *fusion operator*:
each native scale may contribute only to one member of an exact orthogonal
multiresolution decomposition.

For nested block-average projections P1, P2, P3, the four ownership bands are

    Q0 = I - P1,  Q1 = P1 - P2,  Q2 = P2 - P3,  Q3 = P3.

They satisfy sum_i Qi = I and Qi Qj = 0 for i != j.  Hence evidence from two
scales cannot redundantly occupy the same spatial subspace.  This is a single
structural replacement of late logit mixing, not an attention/gating stack.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet


_NUM_SCALES = 4


def block_average_projection(value: Tensor, block_size: int) -> Tensor:
    """Orthogonally project BCHW tensors onto aligned block constants."""

    if value.ndim != 4:
        raise ValueError("projection input must be BCHW")
    block_size = int(block_size)
    if block_size < 1 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    height, width = value.shape[-2:]
    if height % block_size or width % block_size:
        raise ValueError(
            "OSO requires height and width divisible by the coarsest block; "
            f"got {(height, width)} for block_size={block_size}"
        )
    if block_size == 1:
        return value
    pooled = F.avg_pool2d(value, kernel_size=block_size, stride=block_size)
    return pooled.repeat_interleave(block_size, dim=-2).repeat_interleave(
        block_size, dim=-1
    )


def orthogonal_ownership_bands(contributions: Tensor) -> Tensor:
    """Assign four full-resolution source contributions to disjoint bands.

    Args:
        contributions: ``[B, 4, H, W]`` outputs of the four native fusion
            kernels before adding the shared bias.

    Returns:
        Tensor with the same shape. Channel ``i`` belongs exactly to the
        ownership subspace ``Qi`` described in the module docstring.
    """

    if contributions.ndim != 4 or contributions.shape[1] != _NUM_SCALES:
        raise ValueError("OSO requires contributions shaped [B, 4, H, W]")
    if contributions.shape[-2] % 8 or contributions.shape[-1] % 8:
        raise ValueError("OSO requires spatial dimensions divisible by 8")

    source0, source1, source2, source3 = contributions.unbind(dim=1)
    source0 = source0.unsqueeze(1)
    source1 = source1.unsqueeze(1)
    source2 = source2.unsqueeze(1)
    source3 = source3.unsqueeze(1)

    p1_source0 = block_average_projection(source0, 2)
    p1_source1 = block_average_projection(source1, 2)
    p2_source1 = block_average_projection(source1, 4)
    p2_source2 = block_average_projection(source2, 4)
    p3_source2 = block_average_projection(source2, 8)
    p3_source3 = block_average_projection(source3, 8)

    return torch.cat(
        [
            source0 - p1_source0,
            p1_source1 - p2_source1,
            p2_source2 - p3_source2,
            p3_source3,
        ],
        dim=1,
    )


def native_fusion_contributions(
    scale_logits: Tensor,
    fusion: nn.Conv2d,
) -> Tensor:
    """Apply each input-channel kernel of MSHNet's native final convolution."""

    if scale_logits.ndim != 4 or scale_logits.shape[1] != _NUM_SCALES:
        raise ValueError("scale_logits must be shaped [B, 4, H, W]")
    if fusion.in_channels != _NUM_SCALES or fusion.out_channels != 1:
        raise ValueError("fusion must be MSHNet's Conv2d(4, 1, ...)")
    weight = fusion.weight.permute(1, 0, 2, 3).contiguous()
    return F.conv2d(
        scale_logits,
        weight,
        bias=None,
        stride=fusion.stride,
        padding=fusion.padding,
        dilation=fusion.dilation,
        groups=_NUM_SCALES,
    )


def orthogonal_scale_ownership_fusion(
    scale_logits: Tensor,
    fusion: nn.Conv2d,
) -> dict[str, Tensor]:
    """Fuse native logits after exact complementary-subspace assignment."""

    raw = native_fusion_contributions(scale_logits, fusion)
    owned = orthogonal_ownership_bands(raw)
    pred = owned.sum(dim=1, keepdim=True)
    if fusion.bias is not None:
        pred = pred + fusion.bias.view(1, 1, 1, 1)
    return {
        "raw_contributions": raw,
        "ownership_contributions": owned,
        "pred": pred,
        "reconstructed": owned.sum(dim=1, keepdim=True)
        + (
            fusion.bias.view(1, 1, 1, 1)
            if fusion.bias is not None
            else pred.new_zeros(())
        ),
    }


class OrthogonalScaleOwnershipMSHNet(DeterministicMSHNet):
    """Canonical MSHNet with one structural replacement at scale fusion.

    The module/parameter names are identical to deterministic MSHNet.  A
    canonical checkpoint and its Adagrad state therefore remain strictly
    loadable; only the semantics of ``final`` are changed from unconstrained
    overlapping mixing to complementary-subspace synthesis.
    """

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_ownership_state: bool = False,
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
            if return_ownership_state:
                raise ValueError("ownership state requires the four-scale path")
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
        fusion_state = orthogonal_scale_ownership_fusion(
            scale_logits, self.final
        )
        if return_ownership_state:
            return {
                "side_logits": tuple(side_logits),
                "scale_logits": scale_logits,
                **fusion_state,
            }
        return side_logits, fusion_state["pred"]


OSOMSHNet = OrthogonalScaleOwnershipMSHNet


__all__ = [
    "OSOMSHNet",
    "OrthogonalScaleOwnershipMSHNet",
    "block_average_projection",
    "native_fusion_contributions",
    "orthogonal_ownership_bands",
    "orthogonal_scale_ownership_fusion",
]
