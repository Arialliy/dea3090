from __future__ import annotations

import torch
import torch.nn.functional as F

from model.loss import SoftIoULoss, build_safe_bg


def _masked_mean(
    loss_map: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    return (loss_map * mask).sum() / (mask.sum() + eps)


def _topk_safe_mask(
    score: torch.Tensor,
    safe_bg: torch.Tensor,
    ratio: float = 0.001,
    min_score: float = 0.45,
) -> torch.Tensor:
    b, _, h, w = score.shape
    score = score.detach() * safe_bg
    flat_score = score.view(b, -1)
    flat_safe = safe_bg.view(b, -1)
    k = max(1, int(h * w * ratio))

    masks = []
    for i in range(b):
        valid_idx = torch.nonzero(
            (flat_safe[i] > 0.5) & (flat_score[i] >= min_score),
            as_tuple=False,
        ).view(-1)
        m = torch.zeros_like(flat_safe[i])
        if valid_idx.numel() > 0:
            local_score = flat_score[i, valid_idx]
            local_k = min(k, valid_idx.numel())
            top_idx_local = torch.topk(local_score, k=local_k, largest=True).indices
            m[valid_idx[top_idx_local]] = 1.0
        masks.append(m)

    return torch.stack(masks, dim=0).view(b, 1, h, w)


def _limit_safe_mask_by_score(
    mask: torch.Tensor,
    score: torch.Tensor,
    max_ratio: float,
) -> torch.Tensor:
    if max_ratio <= 0:
        return mask

    b, _, h, w = mask.shape
    flat_mask = mask.view(b, -1)
    flat_score = score.detach().view(b, -1)
    max_k = max(1, int(h * w * max_ratio))

    limited = []
    for i in range(b):
        valid_idx = torch.nonzero(flat_mask[i] > 0.5, as_tuple=False).view(-1)
        out = torch.zeros_like(flat_mask[i])
        if valid_idx.numel() > 0:
            local_k = min(max_k, valid_idx.numel())
            local_score = flat_score[i, valid_idx]
            top_idx_local = torch.topk(local_score, k=local_k, largest=True).indices
            out[valid_idx[top_idx_local]] = 1.0
        limited.append(out)

    return torch.stack(limited, dim=0).view(b, 1, h, w)


def build_hard_clutter_label(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    safe_kernel: int = 15,
    topk_ratio: float = 0.001,
    topk_min_score: float = 0.45,
    max_hard_bg_ratio: float = 0.003,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = target.float()
    safe_bg = build_safe_bg(target, kernel_size=safe_kernel)

    z_base = full_dea_out["z_base"]
    z_target = full_dea_out["z_target"]
    scale_logits_full = full_dea_out["scale_logits_full"]

    with torch.no_grad():
        p_base = torch.sigmoid(z_base.detach())
        p_target = torch.sigmoid(z_target.detach())
        p_scale = torch.sigmoid(scale_logits_full.detach()).max(dim=1, keepdim=True)[0]
        hard_score = torch.maximum(torch.maximum(p_base, p_target), p_scale)

        hard_by_threshold = safe_bg * (
            (p_base > tau_base).float()
            + (p_target > tau_target).float()
            + (p_scale > tau_scale).float()
        )
        hard_by_threshold = (hard_by_threshold > 0).float()

        hard_by_topk = _topk_safe_mask(
            hard_score,
            safe_bg,
            ratio=topk_ratio,
            min_score=topk_min_score,
        )
        hard_bg = torch.maximum(hard_by_threshold, hard_by_topk) * safe_bg
        hard_bg = _limit_safe_mask_by_score(
            hard_bg,
            hard_score,
            max_ratio=max_hard_bg_ratio,
        )

    return hard_bg, safe_bg


def full_dea_aux_loss_v2(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    epoch: int,
    warm_epoch: int,
    seg_criterion=None,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    margin: float = 1.0,
    lambda_target_aux: float = 0.30,
    lambda_ev_target: float = 0.20,
    lambda_ev_clutter: float = 0.20,
    lambda_clutter_pred: float = 0.20,
    lambda_suppress_gate: float = 0.10,
    lambda_margin: float = 0.05,
    lambda_hard_bg_final: float = 0.10,
    lambda_suppress_order: float = 0.05,
    safe_kernel: int = 15,
    topk_ratio: float = 0.001,
    topk_min_score: float = 0.45,
    max_hard_bg_ratio: float = 0.003,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target.float()
    device = target.device

    z_base = full_dea_out["z_base"]
    z_target = full_dea_out["z_target"]
    z_final = full_dea_out["z_final"]
    z_clutter = full_dea_out["z_clutter"]
    e_t_logit = full_dea_out["target_evidence_logit"]
    e_c_logit = full_dea_out["clutter_evidence_logit"]
    suppression_logit = full_dea_out["suppression_logit"]

    hard_bg, safe_bg = build_hard_clutter_label(
        full_dea_out=full_dea_out,
        target=target,
        tau_base=tau_base,
        tau_target=tau_target,
        tau_scale=tau_scale,
        safe_kernel=safe_kernel,
        topk_ratio=topk_ratio,
        topk_min_score=topk_min_score,
        max_hard_bg_ratio=max_hard_bg_ratio,
    )
    valid_ev = torch.clamp(target + hard_bg, 0.0, 1.0)

    if seg_criterion is None:
        loss_target_aux = SoftIoULoss(z_target, target)
    else:
        loss_target_aux = seg_criterion(z_target, target, warm_epoch, epoch)

    loss_ev_t = _masked_mean(
        F.binary_cross_entropy_with_logits(e_t_logit, target, reduction="none"),
        valid_ev,
    )
    loss_ev_c = _masked_mean(
        F.binary_cross_entropy_with_logits(e_c_logit, hard_bg, reduction="none"),
        valid_ev,
    )
    loss_clutter_pred = _masked_mean(
        F.binary_cross_entropy_with_logits(z_clutter, hard_bg, reduction="none"),
        valid_ev,
    )
    loss_suppress_gate = _masked_mean(
        F.binary_cross_entropy_with_logits(
            suppression_logit,
            hard_bg,
            reduction="none",
        ),
        valid_ev,
    )

    loss_margin = _masked_mean(
        F.relu(margin - (e_t_logit - e_c_logit)),
        target,
    ) + _masked_mean(
        F.relu(margin - (e_c_logit - e_t_logit)),
        hard_bg,
    )

    loss_hard_bg_final = _masked_mean(
        F.binary_cross_entropy_with_logits(
            z_final,
            torch.zeros_like(z_final),
            reduction="none",
        ),
        hard_bg,
    )
    loss_suppress_order = _masked_mean(
        F.relu(z_final - z_base.detach()),
        hard_bg,
    )

    total = torch.tensor(0.0, device=device)
    total = total + lambda_target_aux * loss_target_aux
    total = total + lambda_ev_target * loss_ev_t
    total = total + lambda_ev_clutter * loss_ev_c
    total = total + lambda_clutter_pred * loss_clutter_pred
    total = total + lambda_suppress_gate * loss_suppress_gate
    total = total + lambda_margin * loss_margin
    total = total + lambda_hard_bg_final * loss_hard_bg_final
    total = total + lambda_suppress_order * loss_suppress_order

    target_sum = target.sum() + 1e-6
    hard_sum = hard_bg.sum() + 1e-6
    log_vars = {
        "full_dea_loss_target_aux": loss_target_aux.detach(),
        "full_dea_loss_ev_t": loss_ev_t.detach(),
        "full_dea_loss_ev_c": loss_ev_c.detach(),
        "full_dea_loss_clutter_pred": loss_clutter_pred.detach(),
        "full_dea_loss_suppress_gate": loss_suppress_gate.detach(),
        "full_dea_loss_margin": loss_margin.detach(),
        "full_dea_loss_hard_bg_final": loss_hard_bg_final.detach(),
        "full_dea_loss_suppress_order": loss_suppress_order.detach(),
        "hard_bg_ratio": hard_bg.detach().mean(),
        "safe_bg_ratio": safe_bg.detach().mean(),
        "target_evidence_on_gt": (
            torch.sigmoid(e_t_logit).detach() * target
        ).sum()
        / target_sum,
        "clutter_evidence_on_gt": (
            torch.sigmoid(e_c_logit).detach() * target
        ).sum()
        / target_sum,
        "target_evidence_on_hard_bg": (
            torch.sigmoid(e_t_logit).detach() * hard_bg
        ).sum()
        / hard_sum,
        "clutter_evidence_on_hard_bg": (
            torch.sigmoid(e_c_logit).detach() * hard_bg
        ).sum()
        / hard_sum,
        "suppression_on_gt": (
            torch.sigmoid(suppression_logit).detach() * target
        ).sum()
        / target_sum,
        "suppression_on_hard_bg": (
            torch.sigmoid(suppression_logit).detach() * hard_bg
        ).sum()
        / hard_sum,
        "alpha": full_dea_out["alpha"].detach(),
    }
    return total, log_vars


def full_dea_loss(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    lambda_cf: float = 0.1,
    lambda_bg: float = 0.05,
    lambda_sep: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return full_dea_aux_loss_v2(
        full_dea_out=full_dea_out,
        target=target,
        epoch=1,
        warm_epoch=0,
        lambda_target_aux=1.0,
        lambda_ev_target=lambda_sep,
        lambda_ev_clutter=lambda_sep,
        lambda_clutter_pred=lambda_cf,
        lambda_suppress_gate=lambda_bg,
        lambda_hard_bg_final=lambda_bg,
    )
