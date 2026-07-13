"""Order-statistic pooling controls for MSHNet resampling audits.

The leave-one-peak endpoint answers a precise counterfactual question: what
would a 2x2 max-pooling cell transmit if its single strongest site were
removed?  It is used first as a frozen-checkpoint diagnostic, not claimed as
the final model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def leave_one_peak_pool2d(x: Tensor, alpha: float = 1.0) -> Tensor:
    """Interpolate between 2x2 max pooling and its leave-one-peak outcome.

    ``alpha=0`` is ordinary max pooling. ``alpha=1`` transmits the second
    largest value in every non-overlapping 2x2 cell and channel.
    """

    if x.ndim != 4:
        raise ValueError("x must be a BCHW tensor")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("spatial dimensions must be divisible by two")
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")

    batch, channels, height, width = x.shape
    cells = F.unfold(x, kernel_size=2, stride=2)
    cells = cells.view(batch, channels, 4, height // 2, width // 2)
    largest = torch.topk(cells, k=2, dim=2, sorted=True).values
    maximum = largest[:, :, 0]
    leave_one_peak = largest[:, :, 1]
    return maximum + alpha * (leave_one_peak - maximum)


def channel_consensus_pool2d(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Keep a cell maximum only to the extent that channels agree on its site.

    Each channel defines a soft spatial ownership distribution over the four
    sites after within-cell standardization.  Their mean is the cell-level
    consensus.  Chance agreement (1/4) maps to the leave-one-peak endpoint;
    perfect agreement maps to ordinary max pooling.  There are no learned
    weights or stage-specific constants.
    """

    if x.ndim != 4:
        raise ValueError("x must be a BCHW tensor")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("spatial dimensions must be divisible by two")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    batch, channels, height, width = x.shape
    cells = F.unfold(x, kernel_size=2, stride=2)
    cells = cells.view(batch, channels, 4, height // 2, width // 2)
    mean = cells.mean(dim=2, keepdim=True)
    std = cells.std(dim=2, keepdim=True, unbiased=False)
    ownership = torch.softmax((cells - mean) / (std + eps), dim=2)
    consensus = ownership.mean(dim=1, keepdim=True)
    agreement = (ownership * consensus).sum(dim=2)
    survival = ((agreement - 0.25) / 0.75).clamp(0.0, 1.0)

    largest = torch.topk(cells, k=2, dim=2, sorted=True).values
    maximum = largest[:, :, 0]
    leave_one_peak = largest[:, :, 1]
    return leave_one_peak + survival * (maximum - leave_one_peak)


def support_persistence_pool2d(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Parameter-free persistence transport with a factual-survival prior.

    The gate is the equal-weight mean of factual survival (one) and observed
    cross-channel spatial persistence.  Consequently, no cell can delete more
    than half of its strongest-site-exclusive evidence, while perfect support
    remains exact max pooling.
    """

    if x.ndim != 4:
        raise ValueError("x must be a BCHW tensor")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("spatial dimensions must be divisible by two")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    batch, channels, height, width = x.shape
    cells = F.unfold(x, kernel_size=2, stride=2)
    cells = cells.view(batch, channels, 4, height // 2, width // 2)
    mean = cells.mean(dim=2, keepdim=True)
    std = cells.std(dim=2, keepdim=True, unbiased=False)
    ownership = torch.softmax((cells - mean) / (std + eps), dim=2)
    consensus = ownership.mean(dim=1, keepdim=True)
    agreement = (ownership * consensus).sum(dim=2)
    persistence = ((agreement - 0.25) / 0.75).clamp(0.0, 1.0)
    gate = 0.5 * (1.0 + persistence)

    largest = torch.topk(cells, k=2, dim=2, sorted=True).values
    maximum = largest[:, :, 0]
    leave_one_peak = largest[:, :, 1]
    return leave_one_peak + gate * (maximum - leave_one_peak)


def leave_one_channel_influence_pool2d(
    x: Tensor, eps: float = 1e-6, *, return_state: bool = False
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Remove only a channel's exact self-influence on spatial consensus.

    A channel can make its own strongest site look population-supported merely
    because the population mean includes that channel's vote.  We measure this
    circular support by deleting the channel from the consensus and taking the
    resulting agreement drop.  The drop is an exact leave-one-out influence,
    not a tuned deletion coefficient, and is bounded by ``1 / channels``.
    """

    if x.ndim != 4:
        raise ValueError("x must be a BCHW tensor")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("spatial dimensions must be divisible by two")
    if x.shape[1] < 2:
        raise ValueError("leave-one-channel influence requires at least two channels")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    batch, channels, height, width = x.shape
    cells = F.unfold(x, kernel_size=2, stride=2)
    cells = cells.view(batch, channels, 4, height // 2, width // 2)
    mean = cells.mean(dim=2, keepdim=True)
    std = cells.std(dim=2, keepdim=True, unbiased=False)
    ownership = torch.softmax((cells - mean) / (std + eps), dim=2)
    population = ownership.mean(dim=1, keepdim=True)
    leave_one_out_population = (
        channels * population - ownership
    ) / float(channels - 1)
    agreement_with_self = (ownership * population).sum(dim=2)
    agreement_without_self = (ownership * leave_one_out_population).sum(dim=2)
    self_influence = (agreement_with_self - agreement_without_self).clamp(
        0.0, 1.0 / float(channels)
    )

    largest = torch.topk(cells, k=2, dim=2, sorted=True).values
    maximum = largest[:, :, 0]
    deleted_maximum = largest[:, :, 1]
    exclusive = maximum - deleted_maximum
    output = maximum - self_influence * exclusive
    if not return_state:
        return output
    return output, {
        "maximum": maximum,
        "deleted_maximum": deleted_maximum,
        "single_site_ownership": exclusive,
        "agreement_with_self": agreement_with_self,
        "agreement_without_self": agreement_without_self,
        "self_influence": self_influence,
        "survival_gate": 1.0 - self_influence,
    }


def counterfactual_self_support_pool2d(
    x: Tensor, eps: float = 1e-6, *, return_state: bool = False
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Remove the fraction of apparent support attributable only to oneself.

    Unlike the absolute leave-one-channel agreement drop, the ratio below is
    invariant to the overall concentration of a cell's ownership distribution:

    ``rho = positive(agreement_with_self - agreement_without_self) /
    agreement_with_self``.

    Thus ``rho`` has a direct counterfactual meaning: the fraction of the
    channel's apparent population support that disappears when its own vote is
    excluded.  No stage coefficient or learned gate is introduced.
    """

    if x.ndim != 4:
        raise ValueError("x must be a BCHW tensor")
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("spatial dimensions must be divisible by two")
    if x.shape[1] < 2:
        raise ValueError("counterfactual self-support requires at least two channels")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    batch, channels, height, width = x.shape
    cells = F.unfold(x, kernel_size=2, stride=2)
    cells = cells.view(batch, channels, 4, height // 2, width // 2)
    mean = cells.mean(dim=2, keepdim=True)
    std = cells.std(dim=2, keepdim=True, unbiased=False)
    ownership = torch.softmax((cells - mean) / (std + eps), dim=2)
    population = ownership.mean(dim=1, keepdim=True)
    leave_one_out_population = (
        channels * population - ownership
    ) / float(channels - 1)
    agreement_with_self = (ownership * population).sum(dim=2)
    agreement_without_self = (ownership * leave_one_out_population).sum(dim=2)
    self_support = (
        (agreement_with_self - agreement_without_self).clamp_min(0.0)
        / (agreement_with_self + eps)
    ).clamp(0.0, 1.0)

    largest = torch.topk(cells, k=2, dim=2, sorted=True).values
    maximum = largest[:, :, 0]
    deleted_maximum = largest[:, :, 1]
    exclusive = maximum - deleted_maximum
    output = maximum - self_support * exclusive
    if not return_state:
        return output
    return output, {
        "maximum": maximum,
        "deleted_maximum": deleted_maximum,
        "single_site_ownership": exclusive,
        "agreement_with_self": agreement_with_self,
        "agreement_without_self": agreement_without_self,
        "counterfactual_self_support": self_support,
        "survival_gate": 1.0 - self_support,
    }


__all__ = [
    "channel_consensus_pool2d",
    "counterfactual_self_support_pool2d",
    "leave_one_channel_influence_pool2d",
    "leave_one_peak_pool2d",
    "support_persistence_pool2d",
]
