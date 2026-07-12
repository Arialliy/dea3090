"""Counterfactual scale-coalition views for native MSHNet fusion.

MSHNet fuses four full-resolution side logits with one linear convolution.
Because the fusion is linear in its four input channels, deleting one scale
has an exact counterfactual logit: ``z_full - contribution_i``.  The helper
below exposes the four leave-one-scale-out coalitions without adding learned
parameters or changing the inference graph.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def assemble_mshnet_scale_logits(masks: Sequence[Tensor]) -> Tensor:
    """Lift the four native MSHNet side logits to the finest resolution."""

    if len(masks) != 4:
        raise ValueError("MSHNet scale coalitions require exactly four masks")
    if any(mask.ndim != 4 or mask.shape[1] != 1 for mask in masks):
        raise ValueError("every side logit must have shape [B,1,H,W]")
    batch_size = masks[0].shape[0]
    if any(mask.shape[0] != batch_size for mask in masks):
        raise ValueError("side-logit batch sizes must match")

    full_size = masks[0].shape[-2:]
    lifted = [masks[0]]
    for mask in masks[1:]:
        lifted.append(
            F.interpolate(
                mask,
                size=full_size,
                mode="bilinear",
                align_corners=True,
            )
        )
    return torch.cat(lifted, dim=1)


def leave_one_scale_out_coalitions(
    masks: Sequence[Tensor],
    z_full: Tensor,
    fusion: nn.Conv2d,
) -> dict[str, Tensor]:
    """Return exact per-scale contributions and four deletion coalitions.

    The output ``coalition_logits[:, i]`` is the native fused prediction when
    scale ``i`` contributes zero while the fusion bias and all other scales
    remain unchanged.  Gradients therefore reach three scales per coalition
    and never reach the deleted scale through that coalition.
    """

    if not isinstance(fusion, nn.Conv2d):
        raise TypeError("fusion must be torch.nn.Conv2d")
    scale_logits = assemble_mshnet_scale_logits(masks)
    if fusion.in_channels != scale_logits.shape[1] or fusion.out_channels != 1:
        raise ValueError("fusion must map the four scale channels to one logit")
    if z_full.ndim != 4 or z_full.shape[1] != 1:
        raise ValueError("z_full must have shape [B,1,H,W]")

    grouped_weight = fusion.weight[0].unsqueeze(1)
    contributions = F.conv2d(
        scale_logits,
        grouped_weight,
        bias=None,
        stride=fusion.stride,
        padding=fusion.padding,
        dilation=fusion.dilation,
        groups=scale_logits.shape[1],
    )
    if contributions.shape[-2:] != z_full.shape[-2:]:
        raise ValueError("fusion contribution and final-logit shapes differ")

    coalition_logits = z_full - contributions
    reconstructed = contributions.sum(dim=1, keepdim=True)
    if fusion.bias is not None:
        reconstructed = reconstructed + fusion.bias.view(1, 1, 1, 1)
    return {
        "scale_logits": scale_logits,
        "contributions": contributions,
        "coalition_logits": coalition_logits,
        "reconstructed": reconstructed,
    }


def direct_zero_channel_coalitions(
    scale_logits: Tensor,
    fusion: nn.Conv2d,
) -> Tensor:
    """Audit-only deletion by actually zeroing each fusion input channel.

    Unlike the algebraic ``z_full - contribution_i`` path used for efficient
    training, this function invokes the native final convolution four times.
    It is intentionally kept out of the training objective and exists to
    quantify floating-point event-mask sensitivity at the decision boundary.
    """

    if scale_logits.ndim != 4 or scale_logits.shape[1] != 4:
        raise ValueError("scale_logits must have shape [B,4,H,W]")
    if not isinstance(fusion, nn.Conv2d):
        raise TypeError("fusion must be torch.nn.Conv2d")
    if fusion.in_channels != 4 or fusion.out_channels != 1:
        raise ValueError("fusion must map four scale channels to one logit")
    outputs = []
    for scale in range(4):
        retained = scale_logits.clone()
        retained[:, scale] = 0.0
        outputs.append(fusion(retained))
    return torch.cat(outputs, dim=1)


def nested_scale_filtration(
    masks: Sequence[Tensor],
    z_full: Tensor,
    fusion: nn.Conv2d,
) -> dict[str, Tensor]:
    """Build the native fine-to-coarse filtration of fusion contributions.

    Filtration state ``k`` contains scales ``0..k``.  The last state uses the
    direct native prediction, so the deployed MSHNet path remains the exact
    terminal training objective rather than a numerically reconstructed copy.
    """

    coalition = leave_one_scale_out_coalitions(masks, z_full, fusion)
    contributions = coalition["contributions"]
    if fusion.bias is None:
        bias = z_full.new_zeros((1, 1, 1, 1))
    else:
        bias = fusion.bias.view(1, 1, 1, 1)
    prefix = bias + contributions.cumsum(dim=1)
    filtration_logits = torch.cat(
        [prefix[:, :3], z_full],
        dim=1,
    )
    return {
        **coalition,
        "filtration_logits": filtration_logits,
    }


__all__ = [
    "assemble_mshnet_scale_logits",
    "direct_zero_channel_coalitions",
    "leave_one_scale_out_coalitions",
    "nested_scale_filtration",
]
