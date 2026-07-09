from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.full_dea_head import FullDEAHeadV2
from model.full_dea_loss import full_dea_aux_loss_v2


def make_out_and_target():
    torch.manual_seed(13)
    batch, size = 2, 32
    head = FullDEAHeadV2(hidden_channels=16)
    x_d0 = torch.randn(batch, 16, size, size)
    x_d1 = torch.randn(batch, 32, size // 2, size // 2)
    x_d2 = torch.randn(batch, 64, size // 4, size // 4)
    x_d3 = torch.randn(batch, 128, size // 8, size // 8)
    scale_logits_full = torch.randn(batch, 4, size, size)
    z_base = torch.randn(batch, 1, size, size)
    out = head(x_d0, x_d1, x_d2, x_d3, scale_logits_full, z_base)
    target = (torch.rand(batch, 1, size, size) > 0.97).float()
    return out, target


def test_clutter_prediction_changes_loss() -> None:
    out, target = make_out_and_target()
    loss_a, _ = full_dea_aux_loss_v2(
        out,
        target,
        epoch=1,
        warm_epoch=0,
        topk_ratio=0.01,
        topk_min_score=0.0,
    )

    changed = dict(out)
    changed["z_clutter"] = out["z_clutter"] + 1.0
    changed["z_final"] = (
        out["z_target"]
        - out["alpha"] * out["suppression_gate"] * torch.nn.functional.softplus(changed["z_clutter"])
    )
    loss_b, _ = full_dea_aux_loss_v2(
        changed,
        target,
        epoch=1,
        warm_epoch=0,
        topk_ratio=0.01,
        topk_min_score=0.0,
    )

    assert torch.abs(loss_a - loss_b) > 1e-5


def test_clutter_branch_receives_gradient() -> None:
    out, target = make_out_and_target()
    out["z_clutter"].retain_grad()

    loss, _ = full_dea_aux_loss_v2(
        out,
        target,
        epoch=1,
        warm_epoch=0,
        topk_ratio=0.01,
        topk_min_score=0.0,
    )
    loss.backward()

    assert out["z_clutter"].grad is not None
    assert torch.isfinite(out["z_clutter"].grad).all()
    assert out["z_clutter"].grad.abs().sum() > 0


if __name__ == "__main__":
    test_clutter_prediction_changes_loss()
    test_clutter_branch_receives_gradient()
