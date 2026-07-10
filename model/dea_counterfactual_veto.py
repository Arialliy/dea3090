"""Counterfactual Evidence Veto controls built on the full MSHNet path.

The controls preserve MSHNet's encoder, hierarchical decoder, side heads, and
direct final convolution.  They can only attenuate exact per-scale fusion
contributions; no free residual is available.  These models are deliberately
named controls because the mechanism is a constrained dynamic scale gate.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from model.MSHNet import MSHNet, ResNet
from model.dea_evidence import ExactScaleContributionDecomposer


class CounterfactualEvidenceVeto(nn.Module):
    """Shared local veto operator over exact MSHNet scale contributions."""

    def __init__(
        self,
        *,
        context_channels: int = 16,
        active_scales: Sequence[int] = (0, 1, 2, 3),
        kernel_size: int = 7,
        initial_bias: float = -6.0,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        scales = tuple(int(scale) for scale in active_scales)
        if not scales or len(set(scales)) != len(scales):
            raise ValueError("active_scales must be non-empty and unique")
        if any(scale < 0 or scale >= 4 for scale in scales):
            raise ValueError("active_scales must be selected from 0,1,2,3")
        if context_channels <= 0:
            raise ValueError("context_channels must be positive")

        self.context_channels = int(context_channels)
        self.active_scales = scales
        self.veto_predictor = nn.Conv2d(
            self.context_channels + 2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )
        nn.init.zeros_(self.veto_predictor.weight)
        nn.init.constant_(self.veto_predictor.bias, float(initial_bias))

    def forward(
        self,
        *,
        z_base: Tensor,
        contributions: Tensor,
        decoder_feature_0: Tensor,
        veto_strength: float = 1.0,
    ) -> dict[str, Tensor | tuple[int, ...]]:
        if z_base.ndim != 4 or z_base.shape[1] != 1:
            raise ValueError("z_base must have shape [B,1,H,W]")
        if contributions.ndim != 4 or contributions.shape[1] != 4:
            raise ValueError("contributions must have shape [B,4,H,W]")
        if (
            decoder_feature_0.ndim != 4
            or decoder_feature_0.shape[1] != self.context_channels
        ):
            raise ValueError(
                "decoder_feature_0 must have shape [B,%d,H,W]"
                % self.context_channels
            )
        if z_base.shape[0] != contributions.shape[0]:
            raise ValueError("z_base and contributions batch sizes differ")
        if z_base.shape[-2:] != contributions.shape[-2:]:
            raise ValueError("z_base and contributions spatial sizes differ")
        if decoder_feature_0.shape[0] != z_base.shape[0]:
            raise ValueError("decoder context batch size differs from z_base")
        if decoder_feature_0.shape[-2:] != z_base.shape[-2:]:
            raise ValueError("decoder context spatial size differs from z_base")
        strength = float(veto_strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("veto_strength must be in [0,1]")

        veto_channels = [torch.zeros_like(z_base) for _ in range(4)]
        without_scale = z_base - contributions

        if strength > 0.0:
            for scale in self.active_scales:
                contribution = contributions[:, scale : scale + 1]
                veto_input = torch.cat(
                    [
                        contribution,
                        without_scale[:, scale : scale + 1],
                        decoder_feature_0,
                    ],
                    dim=1,
                )
                veto_channels[scale] = strength * torch.sigmoid(
                    self.veto_predictor(veto_input)
                )

        vetoes = torch.cat(veto_channels, dim=1)
        delta = -(vetoes * contributions).sum(dim=1, keepdim=True)
        # The explicit bypass is the strict baseline-identity path used by B0.
        prediction = z_base if strength == 0.0 else z_base + delta
        return {
            "pred": prediction,
            "z_base": z_base,
            "contributions": contributions,
            "without_scale": without_scale,
            "vetoes": vetoes,
            "delta": delta,
            "active_scales": self.active_scales,
        }


class CounterfactualVetoMSHNet(MSHNet):
    """MSHNet with a frozen-backbone counterfactual-veto control."""

    BASELINE_MISSING_PREFIXES = (
        "veto_head.",
        # Public MSHNet checkpoints may predate this repository's DEA-lite head.
        "decidability_head.",
    )
    BASELINE_UNEXPECTED_PREFIXES: tuple[str, ...] = ()

    def __init__(
        self,
        input_channels: int,
        *,
        active_scales: Sequence[int],
        kernel_size: int = 7,
        initial_bias: float = -6.0,
        veto_strength: float = 1.0,
        freeze_baseline: bool = True,
        block=ResNet,
    ) -> None:
        super().__init__(input_channels, block=block)
        self.contribution_decomposer = ExactScaleContributionDecomposer(
            scale_channels=4
        )
        self.veto_head = CounterfactualEvidenceVeto(
            context_channels=16,
            active_scales=active_scales,
            kernel_size=kernel_size,
            initial_bias=initial_bias,
        )
        self.veto_strength = 1.0
        self.set_veto_strength(veto_strength)
        self.baseline_frozen = bool(freeze_baseline)
        if self.baseline_frozen:
            self.freeze_baseline_parameters()

    @property
    def active_scales(self) -> tuple[int, ...]:
        return self.veto_head.active_scales

    def set_veto_strength(self, strength: float) -> None:
        strength = float(strength)
        if not 0.0 <= strength <= 1.0:
            raise ValueError("veto_strength must be in [0,1]")
        self.veto_strength = strength

    def freeze_baseline_parameters(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.veto_head.parameters():
            parameter.requires_grad_(True)
        self.baseline_frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "baseline_frozen", False):
            for name, module in self.named_children():
                if name == "veto_head":
                    module.train(mode)
                else:
                    module.eval()
        return self

    def _forward_features(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))
        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], dim=1))
        return x_d0, x_d1, x_d2, x_d3

    def _warm_outputs(
        self,
        features: tuple[Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[list[Tensor], Tensor]:
        x_d0, x_d1, x_d2, x_d3 = features
        masks = [
            self.output_0(x_d0),
            self.output_1(x_d1),
            self.output_2(x_d2),
            self.output_3(x_d3),
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
        return masks, scale_logits

    def forward(
        self,
        x: Tensor,
        warm_flag: bool = True,
        *,
        return_dict: bool = False,
        veto_strength: float | None = None,
    ):
        features = self._forward_features(x)
        x_d0 = features[0]
        if not warm_flag:
            pred = self.output_0(x_d0)
            if return_dict:
                return {
                    "masks": [],
                    "pred": pred,
                    "z_base": pred,
                    "scale_logits_full": None,
                    "cev": None,
                }
            return [], pred

        masks, scale_logits = self._warm_outputs(features)
        # Baseline metrics always use this native direct convolution.
        z_base = self.final(scale_logits)
        decomposition = self.contribution_decomposer(
            scale_logits=scale_logits,
            z_base=z_base,
            fusion_weight=self.final.weight,
            fusion_bias=self.final.bias,
            stride=self.final.stride,
            padding=self.final.padding,
            dilation=self.final.dilation,
        )
        strength = self.veto_strength if veto_strength is None else veto_strength
        cev = self.veto_head(
            z_base=z_base,
            contributions=decomposition["scale_contributions"],
            decoder_feature_0=x_d0,
            veto_strength=float(strength),
        )

        if return_dict:
            return {
                "masks": masks,
                "pred": cev["pred"],
                "z_base": z_base,
                "scale_logits_full": scale_logits,
                "decoder_feature_0": x_d0,
                "cev": cev,
            }
        return masks, cev["pred"]


class FineScaleVetoMSHNet(CounterfactualVetoMSHNet):
    """Control that can veto only the finest MSHNet contribution ``c0``."""

    def __init__(self, input_channels: int, **kwargs) -> None:
        super().__init__(input_channels, active_scales=(0,), **kwargs)


class SharedCEVMSHNet(CounterfactualVetoMSHNet):
    """Control sharing one veto predictor across all four scales."""

    def __init__(self, input_channels: int, **kwargs) -> None:
        super().__init__(input_channels, active_scales=(0, 1, 2, 3), **kwargs)


__all__ = [
    "CounterfactualEvidenceVeto",
    "CounterfactualVetoMSHNet",
    "FineScaleVetoMSHNet",
    "SharedCEVMSHNet",
]
