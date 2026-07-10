"""Isolated B0/B1 prototype of the Scale-Interaction Exchange Decoder.

SIED does not add a learned decoder, gate, router, or terminal head.  At an
active MSHNet decoder stage it evaluates the *same* native decoder under four
anchored input coalitions, performs the exact two-input Mobius decomposition,
and exchanges independent current-scale response for joint response::

    d_hat = q11 + alpha * (interaction - current_main).

The default prototype activates only ``decoder_0`` and uses a stop-gradient
spatial-mean background reference.  ``alpha == 0`` (or an empty active-stage
set) takes a hard native-MSHNet path and never evaluates an anchored branch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from model.MSHNet import MSHNet, ResNet


_VALID_STAGES = (0, 1, 2, 3)
_VALID_ANCHORS = ("zero", "mean")


def _anchored_input(value: Tensor, mode: str) -> Tensor:
    """Return the absence anchor used by a decoder coalition.

    ``mean`` is a stop-gradient spatial mean.  It retains each channel's DC
    level while removing spatial evidence, and is intended only as the
    zero-anchor sensitivity control described in the SIED design.
    """

    if mode == "zero":
        return torch.zeros_like(value)
    if mode == "mean":
        mean = value.detach().mean(dim=(-2, -1), keepdim=True)
        return mean.expand_as(value)
    raise ValueError(
        "anchor_mode must be one of %s, got %r" % (_VALID_ANCHORS, mode)
    )


def decoder_mobius_decomposition(
    decoder: nn.Module,
    current: Tensor,
    inherited: Tensor,
    *,
    anchor_mode: str = "zero",
) -> dict[str, Tensor]:
    """Evaluate one shared decoder under the four two-input coalitions.

    The decoder must accept the same channel concatenation as an original
    MSHNet decoder stage.  No tensor in the factual path is detached.  Only
    the optional spatial-mean anchors are stop-gradient controls.
    """

    current_anchor = _anchored_input(current, anchor_mode)
    inherited_anchor = _anchored_input(inherited, anchor_mode)

    # q11 is deliberately evaluated first: it is the factual native decoder
    # call and is also the exact state returned when the exchange is disabled.
    q11 = decoder(torch.cat([current, inherited], dim=1))
    q10 = decoder(torch.cat([current, inherited_anchor], dim=1))
    q01 = decoder(torch.cat([current_anchor, inherited], dim=1))
    q00 = decoder(torch.cat([current_anchor, inherited_anchor], dim=1))

    current_main = q10 - q00
    inherited_main = q01 - q00
    interaction = (q11 - q10) - (q01 - q00)

    return {
        "q11": q11,
        "q10": q10,
        "q01": q01,
        "q00": q00,
        "current_main": current_main,
        "inherited_main": inherited_main,
        "interaction": interaction,
    }


class ScaleInteractionExchangeMSHNet(MSHNet):
    """MSHNet with an isolated, weight-shared SIED transition prototype.

    Args:
        input_channels: Number of input image channels.
        alpha: Fixed interaction-exchange strength in ``[0, 1]``.  In this
            B0/B1 prototype it is deliberately a Python float for a
            pre-registered sweep, not a learnable parameter.
        active_stages: Decoder indices on which to perform coalition lifting.
            The B1 default is only the finest stage, ``(0,)``.
        anchor_mode: Stop-gradient spatial ``"mean"`` for the background-
            centered candidate, or ``"zero"`` for the OOD sensitivity control.
        freeze_bn_statistics: Keep every inherited BatchNorm in evaluation
            mode even when the rest of the model is in training mode.  Affine
            BatchNorm parameters remain differentiable.
        block: Native MSHNet block factory.

    This class intentionally registers no extra parameter or buffer, so an
    MSHNet state dict remains strictly loadable.  A later trainable-alpha
    experiment must be introduced separately after the frozen B1 gate.
    """

    def __init__(
        self,
        input_channels: int,
        *,
        alpha: float = 0.0,
        active_stages: Sequence[int] = (0,),
        anchor_mode: str = "mean",
        freeze_bn_statistics: bool = True,
        block=ResNet,
    ) -> None:
        super().__init__(input_channels, block=block)
        self.alpha = 0.0
        self.active_stages: tuple[int, ...] = ()
        self.anchor_mode = "zero"
        self.freeze_bn_statistics = bool(freeze_bn_statistics)

        self.set_alpha(alpha)
        self.set_active_stages(active_stages)
        self.set_anchor_mode(anchor_mode)
        if self.freeze_bn_statistics:
            self._freeze_batch_norm_statistics()

    def set_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [0, 1]")
        self.alpha = alpha

    def set_active_stages(self, active_stages: Sequence[int]) -> None:
        stages = tuple(sorted({int(stage) for stage in active_stages}))
        invalid = [stage for stage in stages if stage not in _VALID_STAGES]
        if invalid:
            raise ValueError(
                "active_stages must be drawn from %s, got %s"
                % (_VALID_STAGES, invalid)
            )
        self.active_stages = stages

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

    def _decode(
        self,
        encoder_features: tuple[Tensor, Tensor, Tensor, Tensor],
        middle: Tensor,
        *,
        alpha: float,
        collect_details: bool,
    ) -> tuple[dict[int, Tensor], dict[int, dict[str, Tensor]]]:
        state = middle
        decoder_states: dict[int, Tensor] = {}
        stage_terms: dict[int, dict[str, Tensor]] = {}

        for stage in (3, 2, 1, 0):
            current = encoder_features[stage]
            inherited = self.up(state)
            decoder = getattr(self, "decoder_%d" % stage)

            if alpha != 0.0 and stage in self.active_stages:
                terms = decoder_mobius_decomposition(
                    decoder,
                    current,
                    inherited,
                    anchor_mode=self.anchor_mode,
                )
                exchange = terms["interaction"] - terms["current_main"]
                state = terms["q11"] + alpha * exchange
                if collect_details:
                    stage_terms[stage] = {
                        **terms,
                        "exchange": exchange,
                        "state": state,
                    }
            else:
                # This is the only decoder call at inactive stages and along
                # the alpha-zero hard path.
                state = decoder(torch.cat([current, inherited], dim=1))
                if collect_details:
                    stage_terms[stage] = {
                        "q11": state,
                        "state": state,
                    }

            decoder_states[stage] = state

        return decoder_states, stage_terms

    def _warm_outputs(
        self, decoder_states: dict[int, Tensor]
    ) -> tuple[list[Tensor], Tensor, Tensor]:
        masks = [
            self.output_0(decoder_states[0]),
            self.output_1(decoder_states[1]),
            self.output_2(decoder_states[2]),
            self.output_3(decoder_states[3]),
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

        hard_baseline = strength == 0.0 or not self.active_stages
        if hard_baseline and not (return_dict or return_details):
            # Exact native forward: no coalition anchor or counterfactual
            # decoder call is constructed in this branch.
            return super().forward(x, warm_flag)

        x_e0, x_e1, x_e2, x_e3, x_m = self._encode(x)
        decoder_states, stage_terms = self._decode(
            (x_e0, x_e1, x_e2, x_e3),
            x_m,
            alpha=0.0 if hard_baseline else strength,
            collect_details=return_dict or return_details,
        )

        if warm_flag:
            masks, scale_logits, pred = self._warm_outputs(decoder_states)
        else:
            masks = []
            scale_logits = None
            pred = self.output_0(decoder_states[0])

        if return_dict or return_details:
            output: dict[str, Any] = {
                "masks": masks,
                "pred": pred,
                "scale_logits_full": scale_logits,
                "decoder_features": tuple(
                    decoder_states[stage] for stage in (0, 1, 2, 3)
                ),
                "sied": {
                    "alpha": 0.0 if hard_baseline else strength,
                    "active_stages": self.active_stages,
                    "anchor_mode": self.anchor_mode,
                    "hard_baseline": hard_baseline,
                    "stage_terms": stage_terms,
                },
            }
            return output

        return masks, pred


# Short alias for experiment scripts without changing the public MSHNet name.
SIEDMSHNet = ScaleInteractionExchangeMSHNet


__all__ = [
    "ScaleInteractionExchangeMSHNet",
    "SIEDMSHNet",
    "decoder_mobius_decomposition",
]
