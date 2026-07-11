from __future__ import annotations

import torch
import torch.nn as nn

from model.loss import SLSIoULoss


class PartialSLSIoULoss(nn.Module):
    """SLS over a known label domain, with exact all-valid degeneration."""

    def __init__(self, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)
        self.canonical = SLSIoULoss()

    def forward(
        self,
        pred_log: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        warm_epoch: int,
        epoch: int,
        with_shape: bool = True,
    ) -> torch.Tensor:
        if pred_log.shape != target.shape or target.shape != valid.shape:
            raise ValueError("pred_log, target, and valid shapes must match")

        # This branch is intentional: it makes the required baseline identity
        # bitwise exact instead of merely algebraically close.
        if bool(torch.all(valid == 1).detach().cpu()):
            return self.canonical(
                pred_log,
                target,
                warm_epoch,
                epoch,
                with_shape=with_shape,
            )

        pred = torch.sigmoid(pred_log)
        target = target.to(dtype=pred.dtype)
        valid = valid.to(dtype=pred.dtype)
        pred_valid = pred * valid
        target_valid = target * valid

        intersection_sum = (pred * target * valid).sum(dim=(1, 2, 3))
        pred_sum = pred_valid.sum(dim=(1, 2, 3))
        target_sum = target_valid.sum(dim=(1, 2, 3))
        distance = ((pred_sum - target_sum) / 2.0).square()
        alpha = (torch.minimum(pred_sum, target_sum) + distance) / (
            torch.maximum(pred_sum, target_sum) + distance + self.eps
        )
        iou = (intersection_sum + self.eps) / (
            pred_sum + target_sum - intersection_sum + self.eps
        )
        has_positive = target_sum > 0
        iou = torch.where(has_positive, iou, torch.zeros_like(iou))

        if epoch <= warm_epoch:
            return 1.0 - iou.mean()
        scaled_iou = torch.where(
            has_positive,
            alpha * iou,
            torch.zeros_like(iou),
        )
        loss = 1.0 - scaled_iou.mean()
        if with_shape:
            loss = loss + self._partial_location_loss(
                pred_valid,
                target_valid,
                valid,
            )
        return loss

    @staticmethod
    def _partial_location_loss(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        height, width = pred.shape[-2:]
        x_index = torch.arange(
            width,
            device=pred.device,
            dtype=pred.dtype,
        ).view(1, 1, 1, width) / width
        y_index = torch.arange(
            height,
            device=pred.device,
            dtype=pred.dtype,
        ).view(1, 1, height, 1) / height
        smooth = 1e-8

        pred_centerx = (x_index * pred).mean(dim=(1, 2, 3))
        pred_centery = (y_index * pred).mean(dim=(1, 2, 3))
        target_centerx = (x_index * target).mean(dim=(1, 2, 3))
        target_centery = (y_index * target).mean(dim=(1, 2, 3))
        angle = (4 / (torch.pi**2)) * torch.square(
            torch.atan(pred_centery / (pred_centerx + smooth))
            - torch.atan(target_centery / (target_centerx + smooth))
        )
        pred_length = torch.sqrt(
            pred_centerx.square() + pred_centery.square() + smooth
        )
        target_length = torch.sqrt(
            target_centerx.square() + target_centery.square() + smooth
        )
        length_ratio = torch.minimum(pred_length, target_length) / (
            torch.maximum(pred_length, target_length) + smooth
        )
        terms = 1.0 - length_ratio + angle

        # Partial labels with no known positive have meaningful background
        # scale supervision but no target location.  The all-valid case was
        # already delegated to the canonical loss above.
        has_positive = target.flatten(1).sum(dim=1) > 0
        has_known_pixel = valid.flatten(1).sum(dim=1) > 0
        active = has_positive & has_known_pixel
        return torch.where(active, terms, torch.zeros_like(terms)).mean()


__all__ = ["PartialSLSIoULoss"]
