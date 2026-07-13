"""Jet-Coherent Sufficient Lift (JCSL) for MSHNet downsampling.

JCSL preserves canonical max pooling and adds one max-owned geometric
coordinate.  A single rotation-invariant second-order jet coherence is formed
from encoder-0 activation energy; each channel reads it at the site that owns
its 2x2 maximum.  The following native ResNet receives ``[maximum, owned_jet]``
and its owned-jet kernels start at zero, exactly embedding baseline MSHNet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model.baseline_embedded_resnet import widen_resnet_input_with_zeros
from model.baselines.mshnet_deterministic import MSHNet


def _filter_scalar(value: Tensor, kernel: Tensor) -> Tensor:
    weight = kernel.to(device=value.device, dtype=value.dtype).view(1, 1, 3, 3)
    return F.conv2d(
        F.pad(value, (1, 1, 1, 1), mode="replicate"), weight
    )


def jet_coherence(x: Tensor) -> dict[str, Tensor]:
    """Return the single scale-normalized, rotation-invariant 2-jet field."""

    if x.ndim != 4:
        raise ValueError("x must be a BCHW tensor")
    energy = torch.linalg.vector_norm(x, dim=1, keepdim=True) / (x.shape[1] ** 0.5)
    sobel_x = x.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    sobel_y = sobel_x.t().contiguous()
    dxx_kernel = x.new_tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
    )
    dyy_kernel = dxx_kernel.t().contiguous()
    dxy_kernel = x.new_tensor(
        [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]
    ) / 4.0
    grad_x = _filter_scalar(energy, sobel_x)
    grad_y = _filter_scalar(energy, sobel_y)
    gradient_norm = torch.linalg.vector_norm(
        torch.stack([grad_x, grad_y], dim=0), dim=0
    )
    dxx = _filter_scalar(energy, dxx_kernel)
    dyy = _filter_scalar(energy, dyy_kernel)
    dxy = _filter_scalar(energy, dxy_kernel)
    eigen_gap = torch.linalg.vector_norm(
        torch.stack([dxx - dyy, 2.0 * dxy], dim=0), dim=0
    )
    trace = dxx + dyy
    lambda_plus = 0.5 * (trace + eigen_gap)
    lambda_minus = 0.5 * (trace - eigen_gap)
    curvature_mass = lambda_plus.abs() + lambda_minus.abs()
    tiny = torch.finfo(x.dtype).tiny
    relative_gradient = torch.where(
        energy > 0,
        gradient_norm / energy.clamp_min(tiny),
        torch.zeros_like(energy),
    )
    anisotropy = torch.where(
        curvature_mass > 0,
        eigen_gap / curvature_mass.clamp_min(tiny),
        torch.zeros_like(curvature_mass),
    )
    coherence = relative_gradient * anisotropy
    bounded_coherence = coherence / torch.sqrt(1.0 + coherence.square())
    return {
        "energy": energy,
        "gradient_norm": gradient_norm,
        "relative_gradient": relative_gradient,
        "eigen_gap": eigen_gap,
        "curvature_mass": curvature_mass,
        "anisotropy": anisotropy,
        "coherence": coherence,
        "bounded_coherence": bounded_coherence,
    }


class JetCoherentSufficientPool2d(nn.Module):
    """Transport each native maximum and its co-located jet coherence."""

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
        cells = F.unfold(x, kernel_size=2, stride=2).view(
            batch, channels, 4, height // 2, width // 2
        )
        maximum, owner = cells.max(dim=2)
        geometry = jet_coherence(x)
        geometry_cells = F.unfold(
            geometry["coherence"], kernel_size=2, stride=2
        ).view(batch, 1, 4, height // 2, width // 2)
        owned_coherence = torch.gather(
            geometry_cells.expand(-1, channels, -1, -1, -1),
            2,
            owner.unsqueeze(2),
        ).squeeze(2)
        owned_jet = maximum * owned_coherence
        lifted = torch.cat([maximum, owned_jet], dim=1)
        if not return_state:
            return lifted
        return lifted, {
            "factual_maximum": maximum,
            "owner": owner,
            "owned_coherence": owned_coherence,
            "owned_jet": owned_jet,
            **geometry,
        }


class JetCoherentSufficientLiftMSHNet(MSHNet):
    """Canonical MSHNet with JCSL only at encoder boundary zero."""

    def __init__(self, input_channels: int) -> None:
        super().__init__(input_channels)
        self.jet_coherent_lift = JetCoherentSufficientPool2d()
        channels = self.encoder_1[0].conv1.in_channels
        self.encoder_1[0] = widen_resnet_input_with_zeros(
            self.encoder_1[0], 2 * channels
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
                if target.shape[1] != 2 * source.shape[1]:
                    raise ValueError(f"unexpected JCSL checkpoint shape at {key}")
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
        return_jet_state: bool = False,
    ):
        e0 = self.encoder_0(self.conv_init(x))
        if return_jet_state:
            transported, jet_state = self.jet_coherent_lift(e0, return_state=True)
        else:
            transported = self.jet_coherent_lift(e0)
            jet_state = None
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
        if not return_jet_state:
            return masks, prediction
        return {"masks": masks, "pred": prediction, **jet_state}


JCSLMSHNet = JetCoherentSufficientLiftMSHNet


__all__ = [
    "JCSLMSHNet",
    "JetCoherentSufficientLiftMSHNet",
    "JetCoherentSufficientPool2d",
    "jet_coherence",
]
