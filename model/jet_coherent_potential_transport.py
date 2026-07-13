"""Jet-Coherent Potential Transport (JCPT) at MSHNet's front boundary.

The component screen supports a full-resolution, bounded trace-free jet field,
but averaging or max-restricting that scalar before encoder-1 destroys most of
its target/false-component separation.  JCPT therefore applies the one field
as a channel potential immediately before the native max-pool transition:

    X' = X + a * q(X),

where ``q`` is rotation/positive-scale invariant and ``a`` is one zero-started
channel vector.  The canonical activation ``X`` is never removed; at ``a=0``
the entire model is exactly baseline MSHNet.  There is no parallel prediction,
attention map, additional loss, or stack of differential feature branches.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model.baselines.mshnet_deterministic import MSHNet
from model.jet_coherent_sufficient_lift import jet_coherence


class JetCoherentPotentialTransportMSHNet(MSHNet):
    """Canonical MSHNet with one 16-parameter front potential transport."""

    def __init__(self, input_channels: int) -> None:
        super().__init__(input_channels)
        channels = self.encoder_0[-1].conv2.out_channels
        self.jet_potential = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def load_canonical_state_dict(
        self, state_dict: dict[str, Tensor]
    ) -> torch.nn.modules.module._IncompatibleKeys:
        if state_dict and all(key.startswith("module.") for key in state_dict):
            state_dict = {
                key[len("module.") :]: value for key, value in state_dict.items()
            }
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing != ["jet_potential"] or unexpected:
            raise ValueError(
                "canonical checkpoint embedding failed: "
                f"missing={missing}, unexpected={unexpected}"
            )
        with torch.no_grad():
            self.jet_potential.zero_()
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def forward(
        self,
        x: Tensor,
        warm_flag: bool,
        *,
        return_transport_state: bool = False,
    ):
        e0_native = self.encoder_0(self.conv_init(x))
        geometry = jet_coherence(e0_native)
        coordinate = geometry["bounded_coherence"]
        update = self.jet_potential * coordinate
        e0 = e0_native + update
        e1 = self.encoder_1(self.pool(e0))
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
        if not return_transport_state:
            return masks, prediction
        return {
            "masks": masks,
            "pred": prediction,
            "e0_native": e0_native,
            "e0_transported": e0,
            "transport_update": update,
            **geometry,
        }


JCPTMSHNet = JetCoherentPotentialTransportMSHNet


__all__ = ["JCPTMSHNet", "JetCoherentPotentialTransportMSHNet"]
