from __future__ import annotations

import os
import sys
from argparse import Namespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main import get_method_metadata, get_method_name, get_run_folder_name, validate_args


def make_args(**kwargs):
    args = Namespace(
        model_type="mshnet",
        init_from_baseline="",
        if_checkpoint=False,
        dea_lambda_single=0.0,
        dea_lambda_dec=0.0,
        dea_lambda_empty=0.0,
        full_dea_lambda=1.0,
        full_dea_ramp_epochs=30,
        full_dea_start_epoch=0,
        full_dea_freeze_backbone_epochs=0,
        full_dea_tau_base=0.45,
        full_dea_tau_target=0.45,
        full_dea_tau_scale=0.45,
        full_dea_topk_ratio=0.001,
        full_dea_topk_min_score=0.45,
        full_dea_max_hard_bg_ratio=0.003,
        full_dea_safe_kernel=15,
        dataset_dir="datasets/NUAA-SIRST",
        seed=20260706,
        deterministic=True,
    )
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


def test_full_dea_rejects_dea_lite_lambdas() -> None:
    args = make_args(model_type="full_dea", dea_lambda_single=0.01)
    with pytest.raises(ValueError):
        validate_args(args)


def test_full_dea_rejects_invalid_safe_kernel() -> None:
    args = make_args(model_type="full_dea", full_dea_safe_kernel=14)
    with pytest.raises(ValueError):
        validate_args(args)


def test_method_metadata_names_full_dea_v2() -> None:
    args = validate_args(make_args(model_type="full_dea"))
    assert get_method_name(args) == "FullDEA-v2"
    meta = get_method_metadata(args)
    assert meta["model_type"] == "full_dea"
    assert meta["method"] == "FullDEA-v2"


def test_run_folder_name_uses_method_name() -> None:
    args = validate_args(make_args(model_type="full_dea"))
    assert get_run_folder_name(args, "2026-07-09-22-00-00") == (
        "FullDEA-v2-2026-07-09-22-00-00"
    )
