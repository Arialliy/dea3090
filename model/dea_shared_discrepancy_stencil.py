"""Shared discrepancy-stencil decoder prototype.

Every factual MSHNet decoder state is retained.  A persistent conditional
increment produces an alternate native-decoder state ``q``; only the
discrepancy ``r = q - d`` is passed through one eight-parameter, zero-DC
spatial stencil shared by all decoder stages::

    d_tilde = d + sum_delta theta_delta * (shift_delta(r) - r).

The stencil contains no channel projection, attention, router, or A/B maps.
Zero weights embed the complete original MSHNet while leaving a first-order
gradient for the eight stencil coefficients.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from model.MSHNet import MSHNet, ResNet


STENCIL_DELTAS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _mean_anchor(value: Tensor) -> Tensor:
    return value.detach().mean(dim=(-2, -1), keepdim=True).expand_as(value)


class SharedDiscrepancyStencil(nn.Module):
    """Eight-neighbour replicate-boundary zero-DC stencil."""

    def __init__(
        self,
        initial_weights: Sequence[float] | None = None,
        *,
        max_l1: float | None = None,
    ) -> None:
        super().__init__()
        if initial_weights is None:
            initial_weights = (0.0,) * len(STENCIL_DELTAS)
        if len(initial_weights) != len(STENCIL_DELTAS):
            raise ValueError("initial_weights must contain exactly 8 values")
        if max_l1 is not None and max_l1 <= 0.0:
            raise ValueError("max_l1 must be positive or None")
        self.theta = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        self.max_l1 = None if max_l1 is None else float(max_l1)

    def effective_weights(self) -> Tensor:
        if self.max_l1 is None:
            return self.theta
        l1 = self.theta.abs().sum()
        scale = torch.clamp(l1 / self.max_l1, min=1.0)
        return self.theta / scale

    def effective_l1(self) -> Tensor:
        return self.effective_weights().abs().sum()

    @staticmethod
    def shifted(value: Tensor, delta: tuple[int, int]) -> Tensor:
        if value.ndim != 4:
            raise ValueError("stencil input must be BCHW")
        dy, dx = delta
        if (dy, dx) not in STENCIL_DELTAS:
            raise ValueError("delta must be one of the eight stencil neighbours")
        # Slice/concatenate is equivalent to one-pixel replicate padding but
        # keeps the CUDA backward path deterministic for kernel-only runs.
        shifted = value
        if dy == -1:
            shifted = torch.cat(
                [shifted[..., :1, :], shifted[..., :-1, :]], dim=-2
            )
        elif dy == 1:
            shifted = torch.cat(
                [shifted[..., 1:, :], shifted[..., -1:, :]], dim=-2
            )
        if dx == -1:
            shifted = torch.cat(
                [shifted[..., :, :1], shifted[..., :, :-1]], dim=-1
            )
        elif dx == 1:
            shifted = torch.cat(
                [shifted[..., :, 1:], shifted[..., :, -1:]], dim=-1
            )
        return shifted

    def forward(self, discrepancy: Tensor) -> Tensor:
        weights = self.effective_weights().to(
            device=discrepancy.device,
            dtype=discrepancy.dtype,
        )
        correction = torch.zeros_like(discrepancy)
        for weight, delta in zip(weights, STENCIL_DELTAS):
            correction = correction + weight * (
                self.shifted(discrepancy, delta) - discrepancy
            )
        return correction


class SharedDiscrepancyStencilMSHNet(MSHNet):
    """MSHNet with one all-scale shared discrepancy stencil.

    ``max_l1=None`` is the default and applies no hidden stability cap.  The
    optional cap is exposed explicitly for a later ablation; both raw and
    effective weights are available through ``stencil.theta`` and
    ``stencil.effective_weights()``.
    """

    def __init__(
        self,
        input_channels: int,
        *,
        initial_weights: Sequence[float] | None = None,
        max_l1: float | None = None,
        freeze_bn_statistics: bool = True,
        block=ResNet,
    ) -> None:
        super().__init__(input_channels, block=block)
        self.stencil = SharedDiscrepancyStencil(
            initial_weights=initial_weights,
            max_l1=max_l1,
        )
        self.freeze_bn_statistics = bool(freeze_bn_statistics)
        if self.freeze_bn_statistics:
            self._freeze_batch_norm_statistics()

    def _freeze_batch_norm_statistics(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "freeze_bn_statistics", False):
            self._freeze_batch_norm_statistics()
        return self

    def _encode(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))
        return x_e0, x_e1, x_e2, x_e3, x_m

    def _decode(
        self,
        encoder_features: tuple[Tensor, Tensor, Tensor, Tensor],
        middle: Tensor,
        *,
        collect_details: bool,
    ) -> tuple[dict[int, Tensor], dict[int, dict[str, Tensor]]]:
        states: dict[int, Tensor] = {}
        details: dict[int, dict[str, Tensor]] = {}

        inherited = self.up(middle)
        base = _mean_anchor(inherited)
        factual = self.decoder_3(
            torch.cat([encoder_features[3], inherited], dim=1)
        )
        local = self.decoder_3(torch.cat([encoder_features[3], base], dim=1))
        state = factual
        increment = factual - local
        states[3] = state
        if collect_details:
            details[3] = {
                "inherited": inherited,
                "base": base,
                "factual": factual,
                "local": local,
                "state": state,
                "increment": increment,
            }

        for stage in (2, 1, 0):
            inherited = self.up(state)
            aligned_increment = self.up(increment)
            base = _mean_anchor(inherited)
            persistent_input = base + aligned_increment
            decoder = getattr(self, "decoder_%d" % stage)

            # Factual is evaluated first so theta=0 follows the native state
            # trajectory exactly.  Counterfactual BN statistics are frozen.
            factual = decoder(
                torch.cat([encoder_features[stage], inherited], dim=1)
            )
            local = decoder(torch.cat([encoder_features[stage], base], dim=1))
            alternate = decoder(
                torch.cat([encoder_features[stage], persistent_input], dim=1)
            )
            discrepancy = alternate - factual
            correction = self.stencil(discrepancy)
            state = factual + correction
            increment = state - local
            states[stage] = state
            if collect_details:
                details[stage] = {
                    "inherited": inherited,
                    "base": base,
                    "persistent_input": persistent_input,
                    "factual": factual,
                    "local": local,
                    "alternate": alternate,
                    "discrepancy": discrepancy,
                    "correction": correction,
                    "state": state,
                    "increment": increment,
                }

        return states, details

    def _warm_outputs(
        self, states: dict[int, Tensor]
    ) -> tuple[list[Tensor], Tensor, Tensor]:
        masks = [
            self.output_0(states[0]),
            self.output_1(states[1]),
            self.output_2(states[2]),
            self.output_3(states[3]),
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
        pred = self.final(scale_logits)
        return masks, scale_logits, pred

    def forward(
        self,
        x: Tensor,
        warm_flag: bool = True,
        *,
        return_dict: bool = False,
        return_details: bool = False,
    ):
        x_e0, x_e1, x_e2, x_e3, x_m = self._encode(x)
        states, details = self._decode(
            (x_e0, x_e1, x_e2, x_e3),
            x_m,
            collect_details=return_dict or return_details,
        )
        if warm_flag:
            masks, scale_logits, pred = self._warm_outputs(states)
        else:
            masks = []
            scale_logits = None
            pred = self.output_0(states[0])

        if return_dict or return_details:
            return {
                "masks": masks,
                "pred": pred,
                "scale_logits_full": scale_logits,
                "decoder_features": tuple(states[stage] for stage in (0, 1, 2, 3)),
                "sds": {
                    "raw_weights": self.stencil.theta,
                    "effective_weights": self.stencil.effective_weights(),
                    "effective_l1": self.stencil.effective_l1(),
                    "max_l1": self.stencil.max_l1,
                    "stage_terms": details,
                },
            }
        return masks, pred


SDSMSHNet = SharedDiscrepancyStencilMSHNet


__all__ = [
    "STENCIL_DELTAS",
    "SharedDiscrepancyStencil",
    "SharedDiscrepancyStencilMSHNet",
    "SDSMSHNet",
]
