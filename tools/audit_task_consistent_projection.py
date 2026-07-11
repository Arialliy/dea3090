#!/usr/bin/env python3
"""Audit task-consistent projected labels without training a new model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from skimage import measure


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model.task_consistent_supervision import (  # noqa: E402
    REASON_DISAPPEARED,
    REASON_FEASIBLE,
    REASON_LOCALIZATION,
    REASON_LOW_IOU,
    REASON_MERGED,
    REASON_SPLIT,
    TaskConsistentProjectionGraph,
)
from tools.audit_rods_assignment import (  # noqa: E402
    read_names,
    resolve_split_file,
)


REASON_NAMES = {
    REASON_FEASIBLE: "feasible",
    REASON_DISAPPEARED: "disappeared",
    REASON_SPLIT: "split",
    REASON_MERGED: "merged",
    REASON_LOW_IOU: "low_iou",
    REASON_LOCALIZATION: "localization",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit scene-level task consistency of projected labels."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-file", default="")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--strides", default="1,2,4,8")
    parser.add_argument("--min-iou", type=float, default=0.5)
    parser.add_argument("--max-centroid-distance", type=float, default=3.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--write", default="")
    return parser.parse_args(argv)


def load_instance_map(path: Path) -> torch.Tensor:
    mask = np.asarray(Image.open(path)) > 0
    labels = measure.label(mask, connectivity=2, background=0).astype(np.int64)
    return torch.from_numpy(labels)


def audit_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    split_file = resolve_split_file(dataset_dir, args.split, args.split_file)
    names = read_names(split_file, args.max_samples)
    strides = tuple(int(value.strip()) for value in args.strides.split(","))
    graph_builder = TaskConsistentProjectionGraph(
        strides=strides,
        min_iou=args.min_iou,
        max_centroid_distance=args.max_centroid_distance,
    )

    component_count = 0
    feasible_counts = [0 for _ in strides]
    iou_values: list[list[float]] = [[] for _ in strides]
    distance_values: list[list[float]] = [[] for _ in strides]
    reasons = [
        {name: 0 for name in REASON_NAMES.values()}
        for _ in strides
    ]
    for name in names:
        instance_map = load_instance_map(dataset_dir / "masks" / f"{name}.png")
        assignment = graph_builder(instance_map.unsqueeze(0))
        count = int(assignment.component_ids[0].numel())
        component_count += count
        if count == 0:
            continue
        feasible = assignment.feasible[0]
        score = assignment.recovery_iou[0]
        distance = assignment.centroid_distance[0]
        reason = assignment.reason_code[0]
        for side_index in range(len(strides)):
            feasible_counts[side_index] += int(feasible[:, side_index].sum().item())
            iou_values[side_index].extend(score[:, side_index].tolist())
            finite = torch.isfinite(distance[:, side_index])
            distance_values[side_index].extend(
                distance[finite, side_index].tolist()
            )
            for code, reason_name in REASON_NAMES.items():
                reasons[side_index][reason_name] += int(
                    (reason[:, side_index] == code).sum().item()
                )

    denominator = max(1, component_count)
    per_stride = []
    for side_index, stride in enumerate(strides):
        scores = np.asarray(iou_values[side_index], dtype=float)
        distances = np.asarray(distance_values[side_index], dtype=float)
        per_stride.append(
            {
                "stride": stride,
                "component_count": component_count,
                "feasible_count": feasible_counts[side_index],
                "feasible_ratio": feasible_counts[side_index] / denominator,
                "recovery_iou_mean": float(scores.mean()) if scores.size else None,
                "recovery_iou_median": float(np.median(scores)) if scores.size else None,
                "finite_centroid_distance_mean": (
                    float(distances.mean()) if distances.size else None
                ),
                "reason_counts": reasons[side_index],
            }
        )

    return {
        "schema_version": 1,
        "method": "task_consistent_projection_audit",
        "claim_scope": (
            "audits the configured max-pool/nearest-lift projector; "
            "does not estimate optimal output-space feasibility"
        ),
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "split_file": str(split_file),
        "sample_count": len(names),
        "component_count": component_count,
        "min_iou": float(args.min_iou),
        "max_centroid_distance": float(args.max_centroid_distance),
        "per_stride": per_stride,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Task-consistent projection audit",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Split: `{report['split']}`",
        f"- Samples: {report['sample_count']}",
        f"- Components: {report['component_count']}",
        f"- Claim scope: {report['claim_scope']}",
        "",
        "| Stride | Feasible | Ratio | Mean IoU | Median IoU | Merge | Split | Low IoU | Localization |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_stride"]:
        reasons = row["reason_counts"]
        lines.append(
            "| {stride} | {count} | {ratio:.4f} | {mean:.4f} | {median:.4f} | {merge} | {split} | {low} | {loc} |".format(
                stride=row["stride"],
                count=row["feasible_count"],
                ratio=row["feasible_ratio"],
                mean=row["recovery_iou_mean"] or 0.0,
                median=row["recovery_iou_median"] or 0.0,
                merge=reasons["merged"],
                split=reasons["split"],
                low=reasons["low_iou"],
                loc=reasons["localization"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_dataset(args)
    if args.write:
        output_path = Path(args.write)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.as_json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(build_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
