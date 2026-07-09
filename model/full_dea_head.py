from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FullDEAHeadV2(nn.Module):
    """Baseline-preserving Full DEA v2 head.

    The head operates at the MSHNet multi-scale fusion point. It starts close to
    the original fused logit and learns target residuals plus subtractive clutter
    suppression.
    """

    def __init__(self, hidden_channels: int = 32, scale_channels: int = 4):
        super().__init__()
        h = hidden_channels

        self.proj0 = ConvBNAct(16, h, kernel_size=1)
        self.proj1 = ConvBNAct(32, h, kernel_size=1)
        self.proj2 = ConvBNAct(64, h, kernel_size=1)
        self.proj3 = ConvBNAct(128, h, kernel_size=1)

        self.feature_fuse = nn.Sequential(
            ConvBNAct(h * 4, h * 2, kernel_size=3),
            ConvBNAct(h * 2, h, kernel_size=3),
        )

        scale_stat_channels = scale_channels + 4
        self.scale_fuse = nn.Sequential(
            ConvBNAct(scale_stat_channels, h // 2, kernel_size=3),
            ConvBNAct(h // 2, h // 2, kernel_size=3),
        )

        evidence_in_channels = h + h // 2
        self.evidence_head = nn.Sequential(
            ConvBNAct(evidence_in_channels, h, kernel_size=3),
            nn.Conv2d(h, 2, kernel_size=1),
        )

        target_in_channels = h + scale_channels + 3
        self.target_delta_head = nn.Sequential(
            ConvBNAct(target_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        clutter_in_channels = h + scale_channels + 3
        self.clutter_head = nn.Sequential(
            ConvBNAct(clutter_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        gate_in_channels = h + 10
        self.suppression_head = nn.Sequential(
            ConvBNAct(gate_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        self.log_alpha = nn.Parameter(torch.tensor(-1.0))
        self._init_close_to_baseline()

    def _init_close_to_baseline(self) -> None:
        nn.init.zeros_(self.target_delta_head[-1].weight)
        nn.init.zeros_(self.target_delta_head[-1].bias)

        nn.init.zeros_(self.clutter_head[-1].weight)
        nn.init.constant_(self.clutter_head[-1].bias, -4.0)

        nn.init.zeros_(self.suppression_head[-1].weight)
        nn.init.constant_(self.suppression_head[-1].bias, -4.0)

    @staticmethod
    def _scale_stats(scale_logits_full: torch.Tensor) -> torch.Tensor:
        scale_mean = scale_logits_full.mean(dim=1, keepdim=True)
        scale_max = scale_logits_full.max(dim=1, keepdim=True)[0]
        scale_min = scale_logits_full.min(dim=1, keepdim=True)[0]
        scale_var = scale_logits_full.var(dim=1, keepdim=True, unbiased=False)
        return torch.cat(
            [scale_logits_full, scale_mean, scale_max, scale_min, scale_var],
            dim=1,
        )

    def forward(
        self,
        x_d0: torch.Tensor,
        x_d1: torch.Tensor,
        x_d2: torch.Tensor,
        x_d3: torch.Tensor,
        scale_logits_full: torch.Tensor,
        z_base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        size = x_d0.shape[-2:]

        f0 = self.proj0(x_d0)
        f1 = F.interpolate(
            self.proj1(x_d1),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        f2 = F.interpolate(
            self.proj2(x_d2),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        f3 = F.interpolate(
            self.proj3(x_d3),
            size=size,
            mode="bilinear",
            align_corners=True,
        )
        fused_feature = self.feature_fuse(torch.cat([f0, f1, f2, f3], dim=1))

        scale_stats = self._scale_stats(scale_logits_full)
        scale_feature = self.scale_fuse(scale_stats)

        evidence_logits = self.evidence_head(
            torch.cat([fused_feature, scale_feature], dim=1)
        )
        target_evidence_logit, clutter_evidence_logit = torch.chunk(
            evidence_logits,
            chunks=2,
            dim=1,
        )
        target_evidence = torch.sigmoid(target_evidence_logit)
        clutter_evidence = torch.sigmoid(clutter_evidence_logit)

        target_gate = torch.sigmoid(target_evidence_logit - clutter_evidence_logit)
        clutter_gate = torch.sigmoid(clutter_evidence_logit - target_evidence_logit)

        target_input = torch.cat(
            [
                fused_feature * (1.0 + target_gate),
                scale_logits_full,
                z_base,
                target_evidence_logit,
                clutter_evidence_logit,
            ],
            dim=1,
        )
        target_delta = self.target_delta_head(target_input)
        z_target = z_base + target_delta

        clutter_input = torch.cat(
            [
                fused_feature * (1.0 + clutter_gate),
                scale_logits_full,
                z_base,
                target_evidence_logit,
                clutter_evidence_logit,
            ],
            dim=1,
        )
        z_clutter = self.clutter_head(clutter_input)

        scale_aux_stats = scale_stats[:, 4:, :, :]
        gate_input = torch.cat(
            [
                fused_feature,
                target_evidence_logit,
                clutter_evidence_logit,
                z_target,
                z_clutter,
                target_gate,
                clutter_gate,
                scale_aux_stats,
            ],
            dim=1,
        )
        suppression_logit = self.suppression_head(gate_input)
        suppression_gate = torch.sigmoid(suppression_logit)

        alpha = F.softplus(self.log_alpha) + 1e-6
        z_final = z_target - alpha * suppression_gate * F.softplus(z_clutter)

        return {
            "z_base": z_base,
            "scale_logits_full": scale_logits_full,
            "fused_feature": fused_feature,
            "target_evidence_logit": target_evidence_logit,
            "clutter_evidence_logit": clutter_evidence_logit,
            "target_evidence": target_evidence,
            "clutter_evidence": clutter_evidence,
            "target_gate": target_gate,
            "clutter_gate": clutter_gate,
            "target_delta": target_delta,
            "z_target": z_target,
            "z_clutter": z_clutter,
            "suppression_logit": suppression_logit,
            "suppression_gate": suppression_gate,
            "alpha": alpha.detach(),
            "y_real": z_target,
            "y_cf": z_clutter,
            "y_final": z_final,
            "z_final": z_final,
        }


FullDEAHead = FullDEAHeadV2
