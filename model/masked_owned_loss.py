from __future__ import annotations

import torch
import torch.nn as nn


class MaskedOwnedScaleIoULoss(nn.Module):
    """Scale-IoU loss over valid pixels only.

    The loss returns one value per sample so the caller can skip side heads with
    no owned target components.  ``valid`` is the important RODS state: target
    pixels owned by another side head must be invalid, not background.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.shape != target.shape or target.shape != valid.shape:
            raise ValueError("logits, target, and valid shapes must match")
        if weight is not None and weight.shape != logits.shape:
            raise ValueError("weight shape must match logits")

        prob = torch.sigmoid(logits)
        target = target.to(dtype=prob.dtype)
        valid = valid.to(dtype=prob.dtype)
        if weight is not None:
            valid = valid * weight.to(dtype=prob.dtype)

        pred = prob * valid
        truth = target * valid

        intersection = (pred * truth).sum(dim=(1, 2, 3))
        pred_sum = pred.sum(dim=(1, 2, 3))
        target_sum = truth.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - intersection

        distance = ((pred_sum - target_sum) / 2.0).square()
        alpha = (torch.minimum(pred_sum, target_sum) + distance) / (
            torch.maximum(pred_sum, target_sum) + distance + self.eps
        )
        iou = (intersection + self.eps) / (union + self.eps)
        return 1.0 - alpha * iou


__all__ = ["MaskedOwnedScaleIoULoss"]
