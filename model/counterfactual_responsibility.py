"""Decision-flip responsibility for exact multi-scale fusion contributions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def build_safe_background(target: Tensor, kernel_size: int = 15) -> Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("target must have shape [B,1,H,W]")
    protected = F.max_pool2d(
        (target > 0.5).to(dtype=target.dtype),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return (protected < 0.5).to(dtype=target.dtype)


def counterfactual_responsibility_suppression(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Suppress only scale contributions that cause safe-background flips.

    A scale is responsible at a pixel iff the native full decision is positive
    while deleting that scale makes it non-positive.  The decision masks are
    detached; gradients act only on the responsible exact contribution.
    Target pixels and their safety neighbourhood never enter the constraint.
    """

    if z_full.ndim != 4 or z_full.shape[1] != 1:
        raise ValueError("z_full must have shape [B,1,H,W]")
    if contributions.ndim != 4 or contributions.shape[1] != 4:
        raise ValueError("contributions must have shape [B,4,H,W]")
    if z_full.shape[0] != contributions.shape[0]:
        raise ValueError("z_full and contributions batch sizes differ")
    if z_full.shape[-2:] != contributions.shape[-2:]:
        raise ValueError("z_full and contributions spatial shapes differ")
    if target.shape != z_full.shape:
        raise ValueError("target and z_full shapes differ")

    safe_background = build_safe_background(target, safe_kernel)
    without_scale = z_full - contributions
    with torch.no_grad():
        responsibility = (
            (z_full.detach() > 0.0)
            & (without_scale.detach() <= 0.0)
            & (safe_background.expand_as(contributions) > 0.5)
        ).to(dtype=z_full.dtype)

    # A decision-flip contribution is necessarily positive.  Softplus gives a
    # stable, monotone pressure towards zero/negative evidence without reading
    # gradients through the discrete counterfactual assignment.
    penalty_map = F.softplus(contributions)
    responsible_count = responsibility.sum()
    loss = (penalty_map * responsibility).sum() / responsible_count.clamp_min(1.0)
    if bool((responsible_count == 0).detach().cpu()):
        loss = contributions.sum() * 0.0

    image_active = responsibility.flatten(1).sum(dim=1) > 0
    logs = {
        "responsibility_ratio": responsibility.mean().detach(),
        "responsible_count": responsible_count.detach(),
        "responsible_image_ratio": image_active.float().mean().detach(),
        "responsible_contribution_mean": (
            (contributions.detach() * responsibility).sum()
            / responsible_count.clamp_min(1.0)
        ),
        "safe_background_ratio": safe_background.mean().detach(),
    }
    return loss, logs


__all__ = [
    "build_safe_background",
    "counterfactual_responsibility_suppression",
]
