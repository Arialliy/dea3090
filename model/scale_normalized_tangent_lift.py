"""Scale-Normalized Level-Set Tangent Lift (SNLT) for MSHNet.

The frozen baseline audit shows that false components and detected targets are
better separated by the relative slope of encoder-0 activation energy than by
peak magnitude or single-site deletion alone.  SNLT therefore transports the
native max-pooled features together with one bounded, scale-normalized tangent
coordinate of their activation-energy level sets.

Only the following native ResNet input domain is widened.  The tangent kernels
start at zero, so the complete network initially equals max-pool MSHNet.  This
is a single geometric downsampling lift, not an attention gate, extra head, or
auxiliary loss.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.baseline_embedded_resnet import widen_resnet_input_with_zeros
from model.baselines.mshnet_deterministic import MSHNet


def _depthwise_sobel(value: Tensor, kernel: Tensor) -> Tensor:
    channels = value.shape[1]
    weight = kernel.to(device=value.device, dtype=value.dtype).view(1, 1, 3, 3)
    weight = weight.repeat(channels, 1, 1, 1)
    return F.conv2d(
        F.pad(value, (1, 1, 1, 1), mode="replicate"),
        weight,
        groups=channels,
    )


class ScaleNormalizedTangentPool2d(nn.Module):
    """Lift max pooling by one bounded relative level-set slope coordinate."""

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
        energy = x.square().mean(dim=1, keepdim=True).sqrt()
        sobel_x = x.new_tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.t().contiguous()
        grad_x = _depthwise_sobel(energy, sobel_x)
        grad_y = _depthwise_sobel(energy, sobel_y)
        slope = torch.linalg.vector_norm(
            torch.stack([grad_x, grad_y], dim=0), dim=0
        )
        denominator = energy + slope
        tiny = torch.finfo(x.dtype).tiny
        relative_slope = torch.where(
            denominator > 0,
            slope / denominator.clamp_min(tiny),
            torch.zeros_like(slope),
        )
        factual = F.max_pool2d(x, kernel_size=2, stride=2)
        # A scalar density is restricted by its cell average; unlike max, this
        # does not manufacture a stronger tangent value during downsampling.
        tangent = F.avg_pool2d(relative_slope, kernel_size=2, stride=2)
        lifted = torch.cat([factual, tangent], dim=1)
        if not return_state:
            return lifted
        return lifted, {
            "factual_maximum": factual,
            "activation_energy": energy,
            "level_set_slope": slope,
            "relative_slope": relative_slope,
            "restricted_tangent": tangent,
        }


class ScaleNormalizedTangentLiftMSHNet(MSHNet):
    """Canonical MSHNet with SNLT at the first downsampling boundary."""

    def __init__(self, input_channels: int) -> None:
        super().__init__(input_channels)
        self.scale_normalized_tangent = ScaleNormalizedTangentPool2d()
        native_channels = self.encoder_1[0].conv1.in_channels
        self.encoder_1[0] = widen_resnet_input_with_zeros(
            self.encoder_1[0], native_channels + 1
        )

    def load_canonical_state_dict(
        self, state_dict: dict[str, Tensor]
    ) -> torch.nn.modules.module._IncompatibleKeys:
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[len("module.") :]: value for key, value in state_dict.items()
            }
        target_state = self.state_dict()
        widened = {
            "encoder_1.0.conv1.weight",
            "encoder_1.0.shortcut.0.weight",
        }
        embedded: dict[str, Tensor] = {}
        for key, target in target_state.items():
            if key not in state_dict:
                raise ValueError(f"canonical state is missing {key}")
            source = state_dict[key]
            if key in widened:
                if target.shape[1] != source.shape[1] + 1:
                    raise ValueError(f"unexpected tangent checkpoint shape at {key}")
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

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_tangent_state: bool = False,
    ):
        e0 = self.encoder_0(self.conv_init(x))
        if return_tangent_state:
            transported, tangent_state = self.scale_normalized_tangent(
                e0, return_state=True
            )
        else:
            transported = self.scale_normalized_tangent(e0)
            tangent_state = None
        e1 = self.encoder_1(transported)
        e2 = self.encoder_2(self.pool(e1))
        e3 = self.encoder_3(self.pool(e2))
        middle = self.middle_layer(self.pool(e3))

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
        if not return_tangent_state:
            return masks, prediction
        return {"masks": masks, "pred": prediction, **tangent_state}


SNLTMSHNet = ScaleNormalizedTangentLiftMSHNet


__all__ = [
    "SNLTMSHNet",
    "ScaleNormalizedTangentLiftMSHNet",
    "ScaleNormalizedTangentPool2d",
]
