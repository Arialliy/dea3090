from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_task_consistent_projection import audit_dataset


def test_projection_audit_reports_identity_and_coarse_loss(tmp_path: Path) -> None:
    dataset = tmp_path / "fixture"
    (dataset / "masks").mkdir(parents=True)
    (dataset / "img_idx").mkdir(parents=True)
    (dataset / "img_idx" / "train_fixture.txt").write_text(
        "sample\n",
        encoding="utf-8",
    )
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2, 2] = 255
    Image.fromarray(mask).save(dataset / "masks" / "sample.png")
    args = argparse.Namespace(
        dataset_dir=str(dataset),
        split_file="",
        split="train",
        strides="1,2",
        min_iou=0.5,
        max_centroid_distance=3.0,
        max_samples=0,
    )

    report = audit_dataset(args)

    assert report["component_count"] == 1
    assert report["per_stride"][0]["feasible_ratio"] == 1.0
    assert report["per_stride"][1]["feasible_ratio"] == 0.0
    assert report["per_stride"][1]["reason_counts"]["low_iou"] == 1
    assert report["nestedness"][0]["violation_count"] == 0
