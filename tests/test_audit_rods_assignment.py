from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from model.resolution_owned_supervision import ResolutionDecidableSupervisionGraph
from tools.audit_rods_assignment import (
    audit_dataset,
    finalize_counters,
    new_counters,
    update_counters,
)


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def make_args(dataset_dir: Path, **overrides) -> argparse.Namespace:
    values = {
        "dataset_dir": str(dataset_dir),
        "split_file": "",
        "split": "train",
        "methods": "interval,hard",
        "ownership_preferred_cells": 3.0,
        "ownership_sigma": 0.75,
        "ownership_min_decidability": 0.25,
        "ownership_interval_ratio": 0.5,
        "ownership_fallback": "side0",
        "max_samples": 0,
        "collapse_threshold": 0.95,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_audit_dataset_reports_multiscale_ownership(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "fixture"
    (dataset_dir / "img_idx").mkdir(parents=True)
    (dataset_dir / "img_idx" / "train_fixture.txt").write_text(
        "sample_a\nsample_b\n",
        encoding="utf-8",
    )
    sample_a = np.zeros((32, 32), dtype=bool)
    sample_a[1:5, 1:5] = True
    sample_a[18:30, 18:30] = True
    sample_b = np.zeros((32, 32), dtype=bool)
    sample_b[8:24, 8:24] = True
    write_mask(dataset_dir / "masks" / "sample_a.png", sample_a)
    write_mask(dataset_dir / "masks" / "sample_b.png", sample_b)

    report = audit_dataset(make_args(dataset_dir))

    interval = report["methods"]["interval"]
    assert report["sample_count"] == 2
    assert interval["component_count"] == 3
    assert interval["coarse_owned_components"] > 0
    assert len(interval["owned_ratio"]) == 4
    assert not interval["collapsed"]


def test_finalize_counters_flags_side0_collapse() -> None:
    instance_map = torch.zeros((1, 8, 8), dtype=torch.long)
    instance_map[0, 2, 2] = 1
    instance_map[0, 5, 5] = 2
    graph = ResolutionDecidableSupervisionGraph(mode="hard")
    counters = new_counters(num_sides=4)

    update_counters(counters, graph(instance_map))
    report = finalize_counters(counters, collapse_threshold=0.5)

    assert report["side0_only_ratio"] == 1.0
    assert report["coarse_owned_ratio"] == 0.0
    assert report["collapse_flags"]["no_coarse_owned_components"]
    assert report["collapsed"]
