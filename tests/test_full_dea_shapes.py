from __future__ import annotations

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.full_dea_head import FullDEAHeadV2
from model.full_dea_loss import build_hard_clutter_label, full_dea_aux_loss_v2
from model.full_dea_mshnet import FullDEAMSHNet


def make_head_inputs(batch: int = 2, size: int = 32):
    torch.manual_seed(7)
    x_d0 = torch.randn(batch, 16, size, size)
    x_d1 = torch.randn(batch, 32, size // 2, size // 2)
    x_d2 = torch.randn(batch, 64, size // 4, size // 4)
    x_d3 = torch.randn(batch, 128, size // 8, size // 8)
    scale_logits_full = torch.randn(batch, 4, size, size)
    z_base = torch.randn(batch, 1, size, size)
    return x_d0, x_d1, x_d2, x_d3, scale_logits_full, z_base


def test_full_dea_v2_head_shapes_and_baseline_init() -> None:
    head = FullDEAHeadV2(hidden_channels=16)
    head.eval()
    inputs = make_head_inputs(size=32)

    with torch.no_grad():
        out = head(*inputs)

    expected = (2, 1, 32, 32)
    for key in [
        "target_evidence",
        "clutter_evidence",
        "target_gate",
        "clutter_gate",
        "target_delta",
        "z_target",
        "z_clutter",
        "suppression_gate",
        "z_final",
        "y_final",
    ]:
        assert out[key].shape == expected, (key, out[key].shape)

    z_base = inputs[-1]
    assert torch.mean(torch.abs(out["z_final"] - z_base)) < 1e-3


def test_full_dea_v2_loss_finite_and_hard_bg_bounded() -> None:
    head = FullDEAHeadV2(hidden_channels=16)
    out = head(*make_head_inputs(size=32))
    target = (torch.rand(2, 1, 32, 32) > 0.97).float()

    hard_bg, safe_bg = build_hard_clutter_label(
        out,
        target,
        topk_ratio=0.01,
        topk_min_score=0.0,
        max_hard_bg_ratio=0.005,
    )
    assert hard_bg.shape == target.shape
    assert safe_bg.shape == target.shape
    assert float(hard_bg.mean()) <= 0.006

    loss, logs = full_dea_aux_loss_v2(
        out,
        target,
        epoch=1,
        warm_epoch=0,
        topk_ratio=0.01,
        topk_min_score=0.0,
        max_hard_bg_ratio=0.005,
    )
    assert torch.isfinite(loss)
    assert logs["hard_bg_ratio"].ndim == 0


def test_full_dea_mshnet_wrapper_contract() -> None:
    torch.manual_seed(11)
    model = FullDEAMSHNet(input_channels=3)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x, warm_flag=True, return_dict=True)

    masks = out["masks"]
    assert [tuple(m.shape[-2:]) for m in masks] == [
        (64, 64),
        (32, 32),
        (16, 16),
        (8, 8),
    ]
    assert out["pred"].shape == (2, 1, 64, 64)
    assert out["z_base"].shape == (2, 1, 64, 64)
    assert out["scale_logits_full"].shape == (2, 4, 64, 64)
    assert out["full_dea"]["z_final"].shape == (2, 1, 64, 64)


if __name__ == "__main__":
    test_full_dea_v2_head_shapes_and_baseline_init()
    test_full_dea_v2_loss_finite_and_hard_bg_bounded()
    test_full_dea_mshnet_wrapper_contract()
