"""Decoder-native persistent conditional-increment prototype.

The model reuses every native MSHNet decoder.  At the coarsest decoder stage
it separates the factual state from an anchored local state and carries their
difference ``xi``.  Each finer stage writes only that persistent conditional
increment on top of a new local anchor before calling the same native decoder.

``alpha`` is an external, fixed homotopy value.  It is intentionally not a
parameter in this isolated mechanics prototype.  ``alpha == 0`` takes a hard
native-MSHNet path; the final candidate ``alpha == 1`` needs exactly two calls
to every decoder stage.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from model.MSHNet import MSHNet, ResNet


_VALID_ANCHORS = ("zero", "mean")


def _conditional_anchor(value: Tensor, mode: str) -> Tensor:
    if mode == "zero":
        return torch.zeros_like(value)
    if mode == "mean":
        return value.detach().mean(dim=(-2, -1), keepdim=True).expand_as(value)
    raise ValueError(
        "anchor_mode must be one of %s, got %r" % (_VALID_ANCHORS, mode)
    )


def _center_increment(increment: Tensor, enabled: bool) -> Tensor:
    if not enabled:
        return increment
    return increment - increment.mean(dim=(-2, -1), keepdim=True)


def seed_conditional_increment(
    decoder: nn.Module,
    current: Tensor,
    inherited: Tensor,
    *,
    anchor_mode: str = "mean",
    center_xi: bool = False,
) -> dict[str, Tensor]:
    """Initialize ``d``, local state ``l``, and persistent increment ``xi``."""

    base = _conditional_anchor(inherited, anchor_mode)
    factual = decoder(torch.cat([current, inherited], dim=1))
    local = decoder(torch.cat([current, base], dim=1))
    increment = _center_increment(factual - local, center_xi)
    return {
        "inherited": inherited,
        "base": base,
        "factual": factual,
        "local": local,
        "state": factual,
        "increment": increment,
    }


def persistent_conditional_increment_step(
    decoder: nn.Module,
    current: Tensor,
    inherited: Tensor,
    aligned_previous_increment: Tensor,
    *,
    alpha: float,
    anchor_mode: str = "mean",
    center_xi: bool = False,
) -> dict[str, Tensor | None]:
    """Run one finer conditional-increment transition.

    ``aligned_previous_increment`` is already resized to ``inherited``.  At
    ``alpha == 1`` the unused factual call is skipped.  For intermediate
    homotopy values the returned state is exactly ``(1-a)d + a*q``.
    """

    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("step alpha must be finite and in (0, 1]")
    if aligned_previous_increment.shape != inherited.shape:
        raise ValueError(
            "aligned_previous_increment must match inherited shape, got %s and %s"
            % (tuple(aligned_previous_increment.shape), tuple(inherited.shape))
        )

    base = _conditional_anchor(inherited, anchor_mode)
    persistent_input = base + aligned_previous_increment

    if alpha == 1.0:
        factual = None
    else:
        factual = decoder(torch.cat([current, inherited], dim=1))
    local = decoder(torch.cat([current, base], dim=1))
    alternate = decoder(torch.cat([current, persistent_input], dim=1))

    if factual is None:
        state = alternate
    else:
        state = (1.0 - alpha) * factual + alpha * alternate
    increment = _center_increment(state - local, center_xi)
    return {
        "inherited": inherited,
        "base": base,
        "persistent_input": persistent_input,
        "factual": factual,
        "local": local,
        "alternate": alternate,
        "state": state,
        "increment": increment,
    }


class PersistentConditionalIncrementMSHNet(MSHNet):
    """MSHNet with one persistent decoder-native conditional increment.

    No parameter or buffer is added.  The default is the final mechanics
    candidate (mean anchor, fixed ``alpha=1``); ``center_xi`` is exposed only
    as an ablation and defaults to disabled.
    """

    def __init__(
        self,
        input_channels: int,
        *,
        alpha: float = 1.0,
        anchor_mode: str = "mean",
        center_xi: bool = False,
        freeze_bn_statistics: bool = True,
        block=ResNet,
    ) -> None:
        super().__init__(input_channels, block=block)
        self.alpha = 1.0
        self.anchor_mode = "mean"
        self.center_xi = bool(center_xi)
        self.freeze_bn_statistics = bool(freeze_bn_statistics)
        self.set_alpha(alpha)
        self.set_anchor_mode(anchor_mode)
        if self.freeze_bn_statistics:
            self._freeze_batch_norm_statistics()

    def set_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        self.alpha = alpha

    def set_anchor_mode(self, anchor_mode: str) -> None:
        if anchor_mode not in _VALID_ANCHORS:
            raise ValueError(
                "anchor_mode must be one of %s, got %r"
                % (_VALID_ANCHORS, anchor_mode)
            )
        self.anchor_mode = anchor_mode

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

    def _native_decode(
        self,
        encoder_features: tuple[Tensor, Tensor, Tensor, Tensor],
        middle: Tensor,
        *,
        collect_details: bool,
    ) -> tuple[dict[int, Tensor], dict[int, dict[str, Any]]]:
        state = middle
        states: dict[int, Tensor] = {}
        details: dict[int, dict[str, Any]] = {}
        for stage in (3, 2, 1, 0):
            inherited = self.up(state)
            current = encoder_features[stage]
            decoder = getattr(self, "decoder_%d" % stage)
            state = decoder(torch.cat([current, inherited], dim=1))
            states[stage] = state
            if collect_details:
                details[stage] = {
                    "inherited": inherited,
                    "factual": state,
                    "state": state,
                }
        return states, details

    def _conditional_decode(
        self,
        encoder_features: tuple[Tensor, Tensor, Tensor, Tensor],
        middle: Tensor,
        *,
        alpha: float,
        collect_details: bool,
    ) -> tuple[dict[int, Tensor], dict[int, dict[str, Any]]]:
        states: dict[int, Tensor] = {}
        details: dict[int, dict[str, Any]] = {}

        inherited = self.up(middle)
        seed = seed_conditional_increment(
            self.decoder_3,
            encoder_features[3],
            inherited,
            anchor_mode=self.anchor_mode,
            center_xi=self.center_xi,
        )
        state = seed["state"]
        increment = seed["increment"]
        states[3] = state
        if collect_details:
            details[3] = seed

        for stage in (2, 1, 0):
            inherited = self.up(state)
            aligned_increment = self.up(increment)
            decoder = getattr(self, "decoder_%d" % stage)
            step = persistent_conditional_increment_step(
                decoder,
                encoder_features[stage],
                inherited,
                aligned_increment,
                alpha=alpha,
                anchor_mode=self.anchor_mode,
                center_xi=self.center_xi,
            )
            state = step["state"]
            increment = step["increment"]
            if state is None or increment is None:
                raise RuntimeError("conditional transition returned an empty state")
            states[stage] = state
            if collect_details:
                details[stage] = step

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
        alpha: float | None = None,
    ):
        strength = self.alpha if alpha is None else float(alpha)
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")

        hard_baseline = strength == 0.0
        if hard_baseline and not (return_dict or return_details):
            return super().forward(x, warm_flag)

        x_e0, x_e1, x_e2, x_e3, x_m = self._encode(x)
        encoder_features = (x_e0, x_e1, x_e2, x_e3)
        collect_details = return_dict or return_details
        if hard_baseline:
            states, stage_details = self._native_decode(
                encoder_features,
                x_m,
                collect_details=collect_details,
            )
        else:
            states, stage_details = self._conditional_decode(
                encoder_features,
                x_m,
                alpha=strength,
                collect_details=collect_details,
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
                "pci": {
                    "alpha": 0.0 if hard_baseline else strength,
                    "anchor_mode": self.anchor_mode,
                    "center_xi": self.center_xi,
                    "hard_baseline": hard_baseline,
                    "stage_terms": stage_details,
                },
            }
        return masks, pred


PCIMSHNet = PersistentConditionalIncrementMSHNet


__all__ = [
    "PersistentConditionalIncrementMSHNet",
    "PCIMSHNet",
    "seed_conditional_increment",
    "persistent_conditional_increment_step",
]
