"""Birth-Constrained Scale Filtration (BCSF) for MSHNet.

MSHNet's affine fusion lets every coarse scale both refine an existing fine
hypothesis and create a new regional maximum.  The latter operation directly
creates a new branch in the superlevel-set component filtration and can become
a low-FPPI false alarm.

BCSF retains the exact native per-scale fusion contributions, but changes their
composition law.  Positive coarse evidence may raise only strict maxima already
born in the finer prefix and is propagated by grayscale geodesic dilation over
that scale's native footprint.  Negative evidence remains free to suppress a
fine hypothesis.  This is one parameter-free fusion operator, not an extra
prediction branch, gate, attention block, or loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from model.baselines.mshnet_deterministic import MSHNet
from model.orthogonal_scale_ownership import native_fusion_contributions


def strict_local_maxima(value: Tensor) -> Tensor:
    """Detached mask of strict 8-neighbour maxima in a BCHW logit map."""

    if value.ndim != 4 or value.shape[1] != 1:
        raise ValueError("value must have shape [B,1,H,W]")
    # Pixels outside the image are not neighbours.  Padding with zero would
    # incorrectly discard a negative-valued maximum on the image boundary.
    padded = F.pad(value, (1, 1, 1, 1), mode="constant", value=float("-inf"))
    patches = F.unfold(padded, kernel_size=3)
    patches = patches.view(value.shape[0], 1, 9, value.shape[2], value.shape[3])
    neighbour_indices = (0, 1, 2, 3, 5, 6, 7, 8)
    neighbour_max = patches[:, :, neighbour_indices].amax(dim=2)
    return (value > neighbour_max).to(value.dtype).detach()


def geodesic_dilation(
    marker: Tensor,
    mask: Tensor,
    *,
    iterations: int,
) -> Tensor:
    """Finite-radius grayscale reconstruction by dilation under ``mask``."""

    if marker.shape != mask.shape:
        raise ValueError("marker and mask must have equal shapes")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if bool((marker > mask + 1e-6).any().detach().cpu()):
        raise ValueError("grayscale reconstruction requires marker <= mask")
    reconstruction = marker
    for _ in range(iterations):
        reconstruction = torch.minimum(
            F.max_pool2d(reconstruction, kernel_size=3, stride=1, padding=1),
            mask,
        )
    return reconstruction


def birth_constrained_scale_filtration(
    contributions: Tensor,
    fusion_bias: Tensor | None,
    *,
    return_state: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Any]]:
    """Compose four exact fine-to-coarse contributions without new births.

    Scale ``i`` has native stride ``2**i`` relative to scale zero.  Its
    reconstruction radius is therefore fixed by the architecture rather than
    tuned on a dataset.
    """

    if contributions.ndim != 4 or contributions.shape[1] != 4:
        raise ValueError("contributions must have shape [B,4,H,W]")
    if fusion_bias is None:
        bias = contributions.new_zeros((1, 1, 1, 1))
    else:
        bias = fusion_bias.reshape(1, 1, 1, 1).to(
            device=contributions.device,
            dtype=contributions.dtype,
        )

    prefix = bias + contributions[:, 0:1]
    prefixes = [prefix]
    birth_masks = []
    for scale in (1, 2, 3):
        increment = contributions[:, scale : scale + 1]
        positive = increment.clamp_min(0.0)
        negative = increment.clamp_max(0.0)
        births = strict_local_maxima(prefix)
        upper_mask = prefix + positive
        marker = prefix + births * positive
        grown = geodesic_dilation(
            marker,
            upper_mask,
            iterations=2**scale,
        )
        prefix = grown + negative
        prefixes.append(prefix)
        birth_masks.append(births)

    if not return_state:
        return prefix
    return prefix, {
        "prefixes": tuple(prefixes),
        "birth_masks": tuple(birth_masks),
        "contributions": contributions,
    }


class BirthConstrainedScaleFiltrationMSHNet(MSHNet):
    """Canonical MSHNet with BCSF replacing only the final composition law."""

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_filtration_state: bool = False,
    ):
        e0 = self.encoder_0(self.conv_init(x))
        e1 = self.encoder_1(self.pool(e0))
        e2 = self.encoder_2(self.pool(e1))
        e3 = self.encoder_3(self.pool(e2))
        middle = self.middle_layer(self.pool(e3))

        d3 = self.decoder_3(torch.cat([e3, self.up(middle)], dim=1))
        d2 = self.decoder_2(torch.cat([e2, self.up(d3)], dim=1))
        d1 = self.decoder_1(torch.cat([e1, self.up(d2)], dim=1))
        d0 = self.decoder_0(torch.cat([e0, self.up(d1)], dim=1))

        if not warm_flag:
            if return_filtration_state:
                raise ValueError("filtration state requires the four-scale path")
            return [], self.output_0(d0)

        masks = [
            self.output_0(d0),
            self.output_1(d1),
            self.output_2(d2),
            self.output_3(d3),
        ]
        scale_logits = torch.cat(
            [
                masks[0],
                self.up(masks[1]),
                self.up_4(masks[2]),
                self.up_8(masks[3]),
            ],
            dim=1,
        )
        contributions = native_fusion_contributions(scale_logits, self.final)
        result = birth_constrained_scale_filtration(
            contributions,
            self.final.bias,
            return_state=return_filtration_state,
        )
        if return_filtration_state:
            pred, state = result
            return {
                "masks": masks,
                "pred": pred,
                "scale_logits": scale_logits,
                **state,
            }
        return masks, result


BCSFMSHNet = BirthConstrainedScaleFiltrationMSHNet


__all__ = [
    "BCSFMSHNet",
    "BirthConstrainedScaleFiltrationMSHNet",
    "birth_constrained_scale_filtration",
    "geodesic_dilation",
    "strict_local_maxima",
]
