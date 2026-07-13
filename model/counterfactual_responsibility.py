"""Decision-flip responsibility for exact multi-scale fusion contributions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter


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


def build_responsibility_mask(
    z_full: Tensor,
    contributions: Tensor,
    safe_background: Tensor,
) -> Tensor:
    """Return the detached exact scale-deletion decision-flip mask."""

    without_scale = z_full - contributions
    with torch.no_grad():
        return (
            (z_full.detach() > 0.0)
            & (without_scale.detach() <= 0.0)
            & (safe_background.expand_as(contributions) > 0.5)
        ).to(dtype=z_full.dtype)


def _validate_inputs(z_full: Tensor, contributions: Tensor, target: Tensor) -> None:
    if z_full.ndim != 4 or z_full.shape[1] != 1:
        raise ValueError("z_full must have shape [B,1,H,W]")
    if contributions.ndim != 4 or contributions.shape[1] < 1:
        raise ValueError("contributions must have shape [B,S,H,W] with S >= 1")
    if z_full.shape[0] != contributions.shape[0]:
        raise ValueError("z_full and contributions batch sizes differ")
    if z_full.shape[-2:] != contributions.shape[-2:]:
        raise ValueError("z_full and contributions spatial shapes differ")
    if target.shape != z_full.shape:
        raise ValueError("target and z_full shapes differ")


def _normalized_selected_penalty(
    penalty_map: Tensor,
    selected: Tensor,
    safe_background: Tensor,
    normalization: str,
) -> tuple[Tensor, dict[str, Tensor]]:
    if normalization not in ("event", "safe_density", "unique_pixel"):
        raise ValueError(
            "normalization must be event, safe_density, or unique_pixel"
        )
    event_count = selected.sum()
    degree = selected.sum(dim=1, keepdim=True)
    unique_mask = degree > 0
    unique_count = unique_mask.sum()
    unique_count_float = unique_count.to(dtype=penalty_map.dtype)
    numerator = (penalty_map * selected).sum()
    if normalization == "event":
        denominator = event_count.clamp_min(1.0)
        loss = numerator / denominator
    elif normalization == "safe_density":
        denominator = safe_background.sum().clamp_min(1.0)
        loss = numerator / denominator
    else:
        per_pixel = (penalty_map * selected).sum(dim=1, keepdim=True) / degree.clamp_min(
            1.0
        )
        denominator = unique_count_float.clamp_min(1.0)
        loss = (per_pixel * unique_mask).sum() / denominator
    if bool((event_count == 0).detach().cpu()):
        loss = penalty_map.sum() * 0.0
    logs = {
        "normalization_denominator": denominator.detach(),
        "unique_responsible_pixels": unique_count_float.detach(),
        "responsibility_mean_degree": (
            event_count / unique_count_float.clamp_min(1.0)
        ).detach(),
        "normalization_event": penalty_map.new_tensor(
            float(normalization == "event")
        ),
        "normalization_safe_density": penalty_map.new_tensor(
            float(normalization == "safe_density")
        ),
        "normalization_unique_pixel": penalty_map.new_tensor(
            float(normalization == "unique_pixel")
        ),
    }
    return loss, logs


def _match_contribution_gradient_l2(
    loss: Tensor,
    contributions: Tensor,
    reference: Tensor,
    selected: Tensor,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Match the control's raw contribution-gradient L2 to SDRR events."""

    with torch.no_grad():
        sigmoid = torch.sigmoid(contributions.detach())
        reference_norm = torch.sqrt((sigmoid.square() * reference).sum())
        selected_norm = torch.sqrt((sigmoid.square() * selected).sum())
        if bool((reference_norm > 0).detach().cpu()) and bool(
            (selected_norm > 0).detach().cpu()
        ):
            scale = reference_norm / selected_norm
        else:
            scale = contributions.new_zeros(())
    return loss * scale, {
        "control_contribution_gradient_l2_scale": scale.detach(),
        "reference_contribution_gradient_l2": reference_norm.detach(),
        "selected_contribution_gradient_l2_before_scale": selected_norm.detach(),
    }


def match_parameter_gradient_l2(
    reference_loss: Tensor,
    control_loss: Tensor,
    parameters: tuple[Parameter, ...],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Scale a control to the reference regularizer's shared-parameter norm."""

    reference_gradients = torch.autograd.grad(
        reference_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    control_gradients = torch.autograd.grad(
        control_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    reference_squared = reference_loss.new_zeros(())
    control_squared = control_loss.new_zeros(())
    for reference_gradient, control_gradient in zip(
        reference_gradients, control_gradients
    ):
        if reference_gradient is not None:
            reference_squared = reference_squared + reference_gradient.detach().square().sum()
        if control_gradient is not None:
            control_squared = control_squared + control_gradient.detach().square().sum()
    reference_norm = torch.sqrt(reference_squared)
    control_norm = torch.sqrt(control_squared)
    match_valid = (
        torch.isfinite(reference_norm)
        & torch.isfinite(control_norm)
        & (reference_norm > 0)
        & (control_norm > torch.finfo(control_norm.dtype).eps)
    )
    if bool(match_valid.detach().cpu()):
        scale = (reference_norm / control_norm).detach()
        matched_loss = control_loss * scale
    else:
        scale = control_loss.new_zeros(())
        # Multiplying an already non-finite loss by zero still yields NaN.
        # A failed match is a control-design failure, not a reason to corrupt
        # the canonical optimization path, so fail closed with a constant zero.
        matched_loss = control_loss.new_zeros(())
    return matched_loss, {
        "control_shared_gradient_l2_scale": scale,
        "reference_shared_gradient_l2": reference_norm.detach(),
        "control_shared_gradient_l2_before_scale": control_norm.detach(),
        "control_shared_gradient_l2_match_valid": match_valid.to(
            dtype=control_loss.dtype
        ).detach(),
    }


def counterfactual_responsibility_suppression(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    normalization: str = "event",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Suppress only scale contributions that cause safe-background flips.

    A scale is responsible at a pixel iff the native full decision is positive
    while deleting that scale makes it non-positive.  The decision masks are
    detached; gradients act only on the responsible exact contribution.
    Target pixels and their safety neighbourhood never enter the constraint.
    """

    _validate_inputs(z_full, contributions, target)

    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )

    # A decision-flip contribution is necessarily positive.  Softplus gives a
    # stable, monotone pressure towards zero/negative evidence without reading
    # gradients through the discrete counterfactual assignment.
    penalty_map = F.softplus(contributions)
    responsible_count = responsibility.sum()
    loss, normalization_logs = _normalized_selected_penalty(
        penalty_map, responsibility, safe_background, normalization
    )

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
        **normalization_logs,
    }
    return loss, logs


def responsibility_conserving_gradient_routing(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    normalization: str = "unique_pixel",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Route a false-decision gradient only to exact responsible sources.

    At each unique responsible pixel, the forward auxiliary value is exactly
    ``softplus(z_full)``.  Its detached derivative ``sigmoid(z_full)`` is split
    uniformly over the deletion-pivotal native contributions.  Consequently,
    the per-pixel contribution-coordinate gradient budget is conserved while
    every non-responsible contribution receives exactly zero auxiliary
    gradient.  The construction is a first-order surrogate; it adds no model
    parameter and does not change the deployed prediction graph.
    """

    _validate_inputs(z_full, contributions, target)
    if normalization not in ("unique_pixel", "safe_density"):
        raise ValueError(
            "RCR normalization must be unique_pixel or safe_density"
        )
    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )
    degree = responsibility.sum(dim=1, keepdim=True)
    unique_mask = degree > 0
    unique_count = unique_mask.sum().to(dtype=z_full.dtype)
    routing_weight = responsibility / degree.clamp_min(1.0)

    decision = z_full.detach()
    decision_value = F.softplus(decision)
    decision_slope = torch.sigmoid(decision)
    routed_surrogate = decision_value + decision_slope * (
        contributions - contributions.detach()
    )
    per_pixel = (routing_weight * routed_surrogate).sum(
        dim=1, keepdim=True
    )
    if normalization == "unique_pixel":
        denominator = unique_count.clamp_min(1.0)
    else:
        denominator = safe_background.sum().clamp_min(1.0)
    loss = (per_pixel * unique_mask).sum() / denominator
    if bool((unique_count == 0).detach().cpu()):
        loss = contributions.sum() * 0.0

    responsible_count = responsibility.sum()
    return loss, {
        "responsibility_ratio": responsibility.mean().detach(),
        "responsible_count": responsible_count.detach(),
        "responsible_image_ratio": (
            responsibility.flatten(1).any(dim=1).float().mean().detach()
        ),
        "unique_responsible_pixels": unique_count.detach(),
        "responsibility_mean_degree": (
            responsible_count / unique_count.clamp_min(1.0)
        ).detach(),
        "routed_decision_value_mean": (
            (decision_value * unique_mask).sum()
            / unique_count.clamp_min(1.0)
        ).detach(),
        "routed_decision_slope_mean": (
            (decision_slope * unique_mask).sum()
            / unique_count.clamp_min(1.0)
        ).detach(),
        "routing_weight_sum_error": (
            (
                routing_weight.sum(dim=1, keepdim=True)
                - unique_mask.to(dtype=z_full.dtype)
            )
            .abs()
            .max()
        ).detach(),
        "normalization_denominator": denominator.detach(),
        "normalization_unique_pixel": z_full.new_tensor(
            float(normalization == "unique_pixel")
        ),
        "normalization_safe_density": z_full.new_tensor(
            float(normalization == "safe_density")
        ),
        "safe_background_ratio": safe_background.mean().detach(),
        "responsibility_conserving_routing": z_full.new_ones(()),
    }


def responsibility_density_risk(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Measure responsible-evidence risk per safe-background location.

    This is the exact product of responsibility-event prevalence and the
    conditional mean responsible-evidence severity.  Computing the loss as a
    direct ratio of sums preserves the established safe-density implementation
    while the logged factorization makes the statistical meaning auditable.
    """

    _validate_inputs(z_full, contributions, target)
    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )
    penalty = F.softplus(contributions)
    responsible_count = responsibility.sum()
    safe_count = safe_background.sum()
    penalty_sum = (penalty * responsibility).sum()
    loss = penalty_sum / safe_count.clamp_min(1.0)
    if bool((responsible_count == 0).detach().cpu()):
        loss = penalty.sum() * 0.0

    prevalence = responsible_count / safe_count.clamp_min(1.0)
    conditional_severity = penalty_sum / responsible_count.clamp_min(1.0)
    factorized = prevalence * conditional_severity
    return loss, {
        "responsibility_ratio": responsibility.mean().detach(),
        "responsible_count": responsible_count.detach(),
        "responsible_image_ratio": (
            responsibility.flatten(1).any(dim=1).float().mean().detach()
        ),
        "responsibility_prevalence": prevalence.detach(),
        "conditional_responsibility_severity": conditional_severity.detach(),
        "responsibility_density_factorized": factorized.detach(),
        "responsibility_density_factorization_error": (
            loss.detach() - factorized.detach()
        ).abs(),
        "normalization_denominator": safe_count.detach(),
        "safe_background_ratio": safe_background.mean().detach(),
        "responsibility_density_risk": z_full.new_ones(()),
    }


def matched_random_responsibility_suppression(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    salt: int = 0,
    normalization: str = "event",
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-scale event-budget control for scale-deletion responsibility.

    For every image and scale, this control computes the number of genuine
    deletion-flip events, then selects the same number of *non-responsible*
    positive contributions from the same safe-background positive-decision
    domain.  Selection uses a stateless integer hash, so it neither consumes
    the training RNG nor changes data/model randomness.  This does *not*
    match unique pixels, degree, margins, or gradient norm and must therefore
    be reported as an unmatched scale-budget control rather than a complete
    matched-random attribution control.
    """

    _validate_inputs(z_full, contributions, target)
    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )
    with torch.no_grad():
        eligible = (
            (z_full.detach() > 0.0).expand_as(contributions)
            & (contributions.detach() > 0.0)
            & (safe_background.expand_as(contributions) > 0.5)
            & (responsibility < 0.5)
        )
        selected = torch.zeros_like(responsibility)
        modulus = 2_147_483_647
        salt_mod = int(salt) % modulus
        for batch_index in range(contributions.shape[0]):
            for scale_index in range(contributions.shape[1]):
                requested = int(
                    responsibility[batch_index, scale_index].sum().item()
                )
                if requested == 0:
                    continue
                candidates = torch.nonzero(
                    eligible[batch_index, scale_index].flatten(),
                    as_tuple=False,
                ).flatten()
                count = min(requested, int(candidates.numel()))
                if count == 0:
                    continue
                scores = torch.remainder(
                    candidates.to(torch.int64) * 1_103_515_245
                    + salt_mod * 12_345
                    + (batch_index * 4 + scale_index + 1) * 97_531,
                    modulus,
                )
                chosen = candidates[torch.argsort(scores)[:count]]
                selected[batch_index, scale_index].view(-1)[chosen] = 1.0

    reference_count = responsibility.sum()
    selected_count = selected.sum()
    penalty_map = F.softplus(contributions)
    loss, normalization_logs = _normalized_selected_penalty(
        penalty_map, selected, safe_background, normalization
    )
    loss, gradient_match_logs = _match_contribution_gradient_l2(
        loss, contributions, responsibility, selected
    )

    reference_active = responsibility.flatten(1).sum(dim=1) > 0
    logs = {
        "responsibility_ratio": responsibility.mean().detach(),
        "responsible_count": reference_count.detach(),
        "responsible_image_ratio": reference_active.float().mean().detach(),
        "control_selected_count": selected_count.detach(),
        "control_selected_ratio": selected.mean().detach(),
        "control_shortage_count": (reference_count - selected_count).detach(),
        "control_budget_match_ratio": (
            selected_count / reference_count.clamp_min(1.0)
        ).detach(),
        "control_contribution_mean": (
            (contributions.detach() * selected).sum()
            / selected_count.clamp_min(1.0)
        ),
        "safe_background_ratio": safe_background.mean().detach(),
        "scale_budget_random_control": z_full.new_ones(()),
        **normalization_logs,
        **gradient_match_logs,
    }
    return loss, logs


def same_pixel_random_scale_suppression(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    salt: int = 0,
    normalization: str = "event",
) -> tuple[Tensor, dict[str, Tensor]]:
    """M4 control: keep responsibility pixels/degree, randomize scale identity."""

    _validate_inputs(z_full, contributions, target)
    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )
    with torch.no_grad():
        selected = torch.zeros_like(responsibility)
        # Each randomized scale receives the derivative magnitude of the true
        # responsible event that it replaces.  This keeps event, pixel and
        # per-event contribution-gradient budgets fixed while changing only
        # the native scale/Jacobian receiving that gradient.  Applying
        # softplus directly to an arbitrary negative scale would have a nearly
        # zero derivative and require an unstable 1/sigmoid rescaling.
        selected_reference_slope = torch.zeros_like(contributions)
        selected_reference_value = torch.zeros_like(contributions)
        modulus = 2_147_483_647
        salt_mod = int(salt) % modulus
        scale_count = contributions.shape[1]
        flat_contributions = contributions.detach().reshape(
            contributions.shape[0], scale_count, -1
        )
        for batch_index in range(contributions.shape[0]):
            active_pixels = torch.nonzero(
                responsibility[batch_index].any(dim=0).flatten(),
                as_tuple=False,
            ).flatten()
            for pixel in active_pixels.tolist():
                true_scales = responsibility[batch_index].reshape(
                    scale_count, -1
                )[:, pixel].bool()
                true_indices = torch.nonzero(
                    true_scales, as_tuple=False
                ).flatten()
                requested = int(true_scales.sum().item())
                candidates = torch.nonzero(~true_scales, as_tuple=False).flatten()
                count = min(requested, int(candidates.numel()))
                if count == 0:
                    continue
                scores = torch.remainder(
                    candidates.to(torch.int64) * 1_103_515_245
                    + salt_mod * 12_345
                    + (batch_index + 1) * 97_531
                    + (pixel + 1) * 433_494_437,
                    modulus,
                )
                chosen = candidates[torch.argsort(scores)[:count]]
                chosen_true = true_indices[:count]
                selected[batch_index].reshape(scale_count, -1)[
                    chosen, pixel
                ] = 1.0
                true_values = flat_contributions[
                    batch_index, chosen_true, pixel
                ]
                selected_reference_slope[batch_index].reshape(
                    scale_count, -1
                )[chosen, pixel] = torch.sigmoid(true_values)
                selected_reference_value[batch_index].reshape(
                    scale_count, -1
                )[chosen, pixel] = F.softplus(true_values)

    reference_count = responsibility.sum()
    selected_count = selected.sum()
    reference_pixels = responsibility.any(dim=1, keepdim=True).sum().to(z_full.dtype)
    selected_pixels = selected.any(dim=1, keepdim=True).sum().to(z_full.dtype)
    # First-order surrogate: the forward penalty equals the replaced true
    # event's softplus value, while the backward slope is routed to the random
    # native scale.  The detached maps keep selection out of autograd.
    penalty_map = selected_reference_value + selected_reference_slope * (
        contributions - contributions.detach()
    )
    loss, normalization_logs = _normalized_selected_penalty(
        penalty_map, selected, safe_background, normalization
    )
    reference_slope = torch.sigmoid(contributions.detach())
    reference_gradient_norm = torch.sqrt(
        (reference_slope.square() * responsibility).sum()
    )
    selected_gradient_norm = torch.sqrt(
        (selected_reference_slope.square() * selected).sum()
    )
    gradient_match_logs = {
        "control_contribution_gradient_l2_scale": contributions.new_ones(()),
        "reference_contribution_gradient_l2": reference_gradient_norm.detach(),
        "selected_contribution_gradient_l2_before_scale": (
            selected_gradient_norm.detach()
        ),
        "random_scale_reference_slope_surrogate": contributions.new_ones(()),
    }
    logs = {
        "responsibility_ratio": responsibility.mean().detach(),
        "responsible_count": reference_count.detach(),
        "responsible_image_ratio": (
            responsibility.flatten(1).any(dim=1).float().mean().detach()
        ),
        "control_selected_count": selected_count.detach(),
        "control_reference_pixels": reference_pixels.detach(),
        "control_selected_pixels": selected_pixels.detach(),
        "control_shortage_count": (reference_count - selected_count).detach(),
        "control_budget_match_ratio": (
            selected_count / reference_count.clamp_min(1.0)
        ).detach(),
        "control_pixel_match_ratio": (
            selected_pixels / reference_pixels.clamp_min(1.0)
        ).detach(),
        "control_contribution_mean": (
            (contributions.detach() * selected).sum()
            / selected_count.clamp_min(1.0)
        ),
        "safe_background_ratio": safe_background.mean().detach(),
        "same_pixel_random_scale_control": z_full.new_ones(()),
        **normalization_logs,
        **gradient_match_logs,
    }
    return loss, logs


def magnitude_matched_nonpivotal_suppression(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    normalization: str = "event",
) -> tuple[Tensor, dict[str, Tensor]]:
    """M3 control: match full margin/contribution but require no deletion flip."""

    _validate_inputs(z_full, contributions, target)
    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )
    without_scale = z_full - contributions
    with torch.no_grad():
        candidates_mask = (
            (z_full.detach() > 0.0).expand_as(contributions)
            & (without_scale.detach() > 0.0)
            & (contributions.detach() > 0.0)
            & (safe_background.expand_as(contributions) > 0.5)
        )
        selected = torch.zeros_like(responsibility)
        selected_flat = selected.reshape(
            selected.shape[0], selected.shape[1], -1
        )
        spatial_size = selected.shape[-2] * selected.shape[-1]
        z_gap_sum = z_full.new_zeros(())
        contribution_gap_sum = z_full.new_zeros(())
        deleted_margin_gap_sum = z_full.new_zeros(())
        flat_z = z_full[:, 0].flatten()
        for scale_index in range(contributions.shape[1]):
            references = torch.nonzero(
                responsibility[:, scale_index].flatten() > 0.5,
                as_tuple=False,
            ).flatten()
            available = torch.nonzero(
                candidates_mask[:, scale_index].flatten(),
                as_tuple=False,
            ).flatten()
            if references.numel() == 0 or available.numel() == 0:
                continue
            flat_contribution = contributions[:, scale_index].detach().flatten()
            z_scale = flat_z[available].std(unbiased=False).clamp_min(1e-6)
            contribution_scale = flat_contribution[available].std(
                unbiased=False
            ).clamp_min(1e-6)
            flat_deleted = without_scale[:, scale_index].detach().flatten()
            deleted_magnitude_scale = flat_deleted[available].abs().std(
                unbiased=False
            ).clamp_min(1e-6)
            for reference in references.tolist():
                if available.numel() == 0:
                    break
                z_gap = (flat_z[available] - flat_z[reference]).abs()
                contribution_gap = (
                    flat_contribution[available]
                    - flat_contribution[reference]
                ).abs()
                deleted_margin_gap = (
                    flat_deleted[available].abs()
                    - flat_deleted[reference].abs()
                ).abs()
                cost = (
                    z_gap / z_scale
                    + contribution_gap / contribution_scale
                    + deleted_margin_gap / deleted_magnitude_scale
                )
                chosen_position = int(torch.argmin(cost).item())
                chosen = available[chosen_position]
                chosen_batch = torch.div(
                    chosen, spatial_size, rounding_mode="floor"
                )
                chosen_pixel = torch.remainder(chosen, spatial_size)
                selected_flat[chosen_batch, scale_index, chosen_pixel] = 1.0
                z_gap_sum = z_gap_sum + z_gap[chosen_position]
                contribution_gap_sum = (
                    contribution_gap_sum + contribution_gap[chosen_position]
                )
                deleted_margin_gap_sum = (
                    deleted_margin_gap_sum
                    + deleted_margin_gap[chosen_position]
                )
                keep = torch.ones(
                    available.numel(), dtype=torch.bool, device=available.device
                )
                keep[chosen_position] = False
                available = available[keep]

    reference_count = responsibility.sum()
    selected_count = selected.sum()
    penalty_map = F.softplus(contributions)
    loss, normalization_logs = _normalized_selected_penalty(
        penalty_map, selected, safe_background, normalization
    )
    loss, gradient_match_logs = _match_contribution_gradient_l2(
        loss, contributions, responsibility, selected
    )
    logs = {
        "responsibility_ratio": responsibility.mean().detach(),
        "responsible_count": reference_count.detach(),
        "responsible_image_ratio": (
            responsibility.flatten(1).any(dim=1).float().mean().detach()
        ),
        "control_selected_image_ratio": (
            selected.flatten(1).any(dim=1).float().mean().detach()
        ),
        "control_selected_count": selected_count.detach(),
        "control_shortage_count": (reference_count - selected_count).detach(),
        "control_budget_match_ratio": (
            selected_count / reference_count.clamp_min(1.0)
        ).detach(),
        "control_mean_abs_full_logit_gap": (
            z_gap_sum / selected_count.clamp_min(1.0)
        ).detach(),
        "control_mean_abs_contribution_gap": (
            contribution_gap_sum / selected_count.clamp_min(1.0)
        ).detach(),
        "control_mean_abs_deleted_margin_gap": (
            deleted_margin_gap_sum / selected_count.clamp_min(1.0)
        ).detach(),
        "control_contribution_mean": (
            (contributions.detach() * selected).sum()
            / selected_count.clamp_min(1.0)
        ),
        "safe_background_ratio": safe_background.mean().detach(),
        "magnitude_matched_nonpivotal_control": z_full.new_ones(()),
        **normalization_logs,
        **gradient_match_logs,
        **{
            "control_scale%d_budget_match_ratio" % scale: (
                selected[:, scale].sum()
                / responsibility[:, scale].sum().clamp_min(1.0)
            ).detach()
            for scale in range(contributions.shape[1])
        },
    }
    return loss, logs


def all_safe_positive_fused_suppression(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    normalization: str = "event",
) -> tuple[Tensor, dict[str, Tensor]]:
    """M1 control: suppress every safe-background positive fused logit."""

    _validate_inputs(z_full, contributions, target)
    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )
    with torch.no_grad():
        selected = (
            (z_full.detach() > 0.0) & (safe_background > 0.5)
        ).to(z_full.dtype)
    loss, normalization_logs = _normalized_selected_penalty(
        F.softplus(z_full), selected, safe_background, normalization
    )
    return loss, {
        "responsible_count": responsibility.sum().detach(),
        "responsible_image_ratio": (
            responsibility.flatten(1).any(dim=1).float().mean().detach()
        ),
        "control_selected_count": selected.sum().detach(),
        "control_selected_image_ratio": (
            selected.flatten(1).any(dim=1).float().mean().detach()
        ),
        "safe_background_ratio": safe_background.mean().detach(),
        "all_safe_positive_fused_control": z_full.new_ones(()),
        **normalization_logs,
    }


def same_responsible_pixel_fused_suppression(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    normalization: str = "event",
) -> tuple[Tensor, dict[str, Tensor]]:
    """M2 control: keep pivotal pixels but discard responsible scale identity."""

    _validate_inputs(z_full, contributions, target)
    safe_background = build_safe_background(target, safe_kernel)
    responsibility = build_responsibility_mask(
        z_full, contributions, safe_background
    )
    with torch.no_grad():
        selected = responsibility.any(dim=1, keepdim=True).to(z_full.dtype)
    loss, normalization_logs = _normalized_selected_penalty(
        F.softplus(z_full), selected, safe_background, normalization
    )
    return loss, {
        "responsible_count": responsibility.sum().detach(),
        "responsible_image_ratio": (
            responsibility.flatten(1).any(dim=1).float().mean().detach()
        ),
        "control_selected_count": selected.sum().detach(),
        "control_selected_image_ratio": (
            selected.flatten(1).any(dim=1).float().mean().detach()
        ),
        "control_reference_pixels": selected.sum().detach(),
        "control_selected_pixels": selected.sum().detach(),
        "safe_background_ratio": safe_background.mean().detach(),
        "same_responsible_pixel_fused_control": z_full.new_ones(()),
        **normalization_logs,
    }


__all__ = [
    "build_safe_background",
    "build_responsibility_mask",
    "counterfactual_responsibility_suppression",
    "responsibility_conserving_gradient_routing",
    "responsibility_density_risk",
    "all_safe_positive_fused_suppression",
    "matched_random_responsibility_suppression",
    "match_parameter_gradient_l2",
    "magnitude_matched_nonpivotal_suppression",
    "same_pixel_random_scale_suppression",
    "same_responsible_pixel_fused_suppression",
]
