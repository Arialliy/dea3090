"""Relative Trace-Free Energy Lift (RTFE) at MSHNet's first transition.

The operator forms one bounded geometric coordinate from the learned encoder-0
energy.  It keeps first-order level-set slope and the trace-free (deviatoric)
part of the Hessian while explicitly excluding the Laplacian trace associated
with isotropic peak enhancement.  A single binomial scale is fixed by the
first 2x resolution transition; no multiscale branch or tuned kernel is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.baseline_embedded_resnet import widen_resnet_input_with_zeros
from model.baselines.mshnet_deterministic import MSHNet


def _filter_scalar(value: Tensor, kernel: Tensor) -> Tensor:
    return F.conv2d(
        F.pad(value, (1, 1, 1, 1), mode="replicate"),
        kernel.to(device=value.device, dtype=value.dtype).view(1, 1, 3, 3),
    )


def relative_trace_free_energy_jet(x: Tensor) -> dict[str, Tensor]:
    """Compute one bounded transition-scale geometric coordinate."""

    if x.ndim != 4:
        raise ValueError("x must be a BCHW tensor")
    energy = torch.linalg.vector_norm(x, dim=1, keepdim=True) / (x.shape[1] ** 0.5)
    binomial = x.new_tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    ) / 16.0
    transition_energy = _filter_scalar(energy, binomial)
    dx_kernel = x.new_tensor(
        [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]
    )
    dy_kernel = dx_kernel.t().contiguous()
    dxx_kernel = x.new_tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
    )
    dyy_kernel = dxx_kernel.t().contiguous()
    dxy_kernel = x.new_tensor(
        [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]
    ) / 4.0
    dx = _filter_scalar(transition_energy, dx_kernel)
    dy = _filter_scalar(transition_energy, dy_kernel)
    dxx = _filter_scalar(transition_energy, dxx_kernel)
    dyy = _filter_scalar(transition_energy, dyy_kernel)
    dxy = _filter_scalar(transition_energy, dxy_kernel)

    # The binomial kernel has spatial variance sigma^2 = 1/2.  These powers
    # are therefore fixed by the transition filter, not selected per dataset.
    sigma_squared = 0.5
    sigma_fourth = sigma_squared**2
    gradient_energy = sigma_squared * (dx.square() + dy.square())
    deviatoric_energy = 0.5 * sigma_fourth * (
        (dxx - dyy).square() + 4.0 * dxy.square()
    )
    response = torch.sqrt((gradient_energy + deviatoric_energy).clamp_min(0.0))
    denominator = torch.sqrt(
        (transition_energy.square() + response.square()).clamp_min(0.0)
    )
    tiny = torch.finfo(x.dtype).tiny
    coordinate = torch.where(
        denominator > 0,
        response / denominator.clamp_min(tiny),
        torch.zeros_like(response),
    )
    return {
        "energy": energy,
        "transition_energy": transition_energy,
        "gradient_energy": gradient_energy,
        "deviatoric_energy": deviatoric_energy,
        "response": response,
        "coordinate": coordinate,
    }


class RelativeTraceFreeEnergyPool2d(nn.Module):
    """Append one area-restricted RTFE coordinate to native max pooling."""

    def forward(
        self,
        x: Tensor,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        geometry = relative_trace_free_energy_jet(x)
        factual = F.max_pool2d(x, kernel_size=2, stride=2)
        restricted = F.avg_pool2d(geometry["coordinate"], kernel_size=2, stride=2)
        lifted = torch.cat([factual, restricted], dim=1)
        if not return_state:
            return lifted
        return lifted, {
            "factual_maximum": factual,
            "restricted_coordinate": restricted,
            **geometry,
        }


class RelativeTraceFreeEnergyLiftMSHNet(MSHNet):
    """Canonical MSHNet with RTFE only at encoder boundary zero."""

    def __init__(self, input_channels: int) -> None:
        super().__init__(input_channels)
        self.relative_trace_free_energy = RelativeTraceFreeEnergyPool2d()
        channels = self.encoder_1[0].conv1.in_channels
        self.encoder_1[0] = widen_resnet_input_with_zeros(
            self.encoder_1[0], channels + 1
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
                    raise ValueError(f"unexpected RTFE checkpoint shape at {key}")
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
        return_energy_state: bool = False,
    ):
        e0 = self.encoder_0(self.conv_init(x))
        if return_energy_state:
            transported, energy_state = self.relative_trace_free_energy(
                e0, return_state=True
            )
        else:
            transported = self.relative_trace_free_energy(e0)
            energy_state = None
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
        if not return_energy_state:
            return masks, prediction
        return {"masks": masks, "pred": prediction, **energy_state}


RTFEMSHNet = RelativeTraceFreeEnergyLiftMSHNet


__all__ = [
    "RTFEMSHNet",
    "RelativeTraceFreeEnergyLiftMSHNet",
    "RelativeTraceFreeEnergyPool2d",
    "relative_trace_free_energy_jet",
]
