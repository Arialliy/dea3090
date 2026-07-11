#!/usr/bin/env python3
"""Audit RODS instance-to-side assignment on a dataset split.

This is a lightweight sanity gate for the RODS idea: it verifies whether the
geometry-defined supervision graph uses multiple side heads or collapses into
fine-head-only supervision.  It does not construct MSHNet or load checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
from skimage import measure
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from model.resolution_owned_supervision import (  # noqa: E402
    ResolutionDecidableSupervisionGraph,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RODS supervision ownership on raw mask components."
    )
    parser.add_argument("--dataset-dir", default=str(PROJECT_DIR / "datasets" / "NUAA-SIRST"))
    parser.add_argument("--split-file", default="")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--methods",
        default="interval,hard,area_only",
        help="Comma-separated graph modes to audit.",
    )
    parser.add_argument("--ownership-preferred-cells", type=float, default=3.0)
    parser.add_argument("--ownership-sigma", type=float, default=0.75)
    parser.add_argument("--ownership-min-decidability", type=float, default=0.25)
    parser.add_argument("--ownership-interval-ratio", type=float, default=0.5)
    parser.add_argument("--ownership-fallback", choices=("side0", "final_only"), default="side0")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--collapse-threshold", type=float, default=0.95)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--write", default="")
    return parser.parse_args(argv)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_split_file(dataset_dir: Path, split: str, split_file: str = "") -> Path:
    if split_file:
        path = Path(split_file)
        if not path.is_absolute():
            path = dataset_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    dataset_name = dataset_dir.name
    candidates = [
        dataset_dir / ("trainval.txt" if split == "train" else "test.txt"),
        dataset_dir / "img_idx" / f"{split}_{dataset_name}.txt",
    ]
    candidates.extend(sorted((dataset_dir / "img_idx").glob(f"{split}_*.txt")))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "could not find split file; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def read_names(path: Path, max_samples: int = 0) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if max_samples > 0:
        names = names[:max_samples]
    if not names:
        raise ValueError(f"empty split file: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate sample names in split file: {path}")
    return names


def load_instance_map(mask_path: Path) -> torch.Tensor:
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    labels = measure.label(
        mask.astype(np.uint8),
        connectivity=2,
        background=0,
    ).astype(np.int64)
    return torch.from_numpy(labels)


def new_counters(num_sides: int) -> dict[str, Any]:
    return {
        "sample_count": 0,
        "empty_sample_count": 0,
        "component_count": 0,
        "owned_counts": [0 for _ in range(num_sides)],
        "primary_counts": [0 for _ in range(num_sides)],
        "decidable_counts": [0 for _ in range(num_sides)],
        "coarse_owned_components": 0,
        "side0_only_components": 0,
        "fallback_unowned_components": 0,
    }


def update_counters(counters: dict[str, Any], assignment: Any) -> None:
    counters["sample_count"] += 1
    ids = assignment.component_ids[0]
    if ids.numel() == 0:
        counters["empty_sample_count"] += 1
        return

    responsibilities = assignment.responsibilities[0]
    primary_owner = assignment.primary_owner[0]
    decidability = assignment.decidability[0]
    counters["component_count"] += int(ids.numel())
    owned = responsibilities > 0
    decidable = decidability > 0
    for side_index in range(owned.shape[1]):
        counters["owned_counts"][side_index] += int(owned[:, side_index].sum().item())
        counters["decidable_counts"][side_index] += int(
            decidable[:, side_index].sum().item()
        )
        counters["primary_counts"][side_index] += int(
            (primary_owner == side_index).sum().item()
        )
    counters["coarse_owned_components"] += int(owned[:, 1:].any(dim=1).sum().item())
    counters["side0_only_components"] += int(
        (owned[:, 0] & ~owned[:, 1:].any(dim=1)).sum().item()
    )
    counters["fallback_unowned_components"] += int((primary_owner < 0).sum().item())


def finalize_counters(
    counters: dict[str, Any],
    *,
    collapse_threshold: float,
) -> dict[str, Any]:
    component_count = max(1, int(counters["component_count"]))
    owned_counts = counters["owned_counts"]
    primary_counts = counters["primary_counts"]
    decidable_counts = counters["decidable_counts"]
    side0_only_ratio = float(counters["side0_only_components"]) / component_count
    coarse_owned_ratio = float(counters["coarse_owned_components"]) / component_count
    side0_primary_ratio = float(primary_counts[0]) / component_count
    collapse_flags = {
        "side0_only_ratio_exceeds_threshold": side0_only_ratio >= collapse_threshold,
        "side0_primary_ratio_exceeds_threshold": side0_primary_ratio >= collapse_threshold,
        "no_coarse_owned_components": counters["coarse_owned_components"] == 0,
    }
    return {
        **counters,
        "owned_ratio": [float(value) / component_count for value in owned_counts],
        "primary_ratio": [float(value) / component_count for value in primary_counts],
        "decidable_ratio": [float(value) / component_count for value in decidable_counts],
        "side0_only_ratio": side0_only_ratio,
        "coarse_owned_ratio": coarse_owned_ratio,
        "side0_primary_ratio": side0_primary_ratio,
        "collapse_threshold": float(collapse_threshold),
        "collapse_flags": collapse_flags,
        "collapsed": any(collapse_flags.values()),
    }


def make_graph(args: argparse.Namespace, mode: str) -> ResolutionDecidableSupervisionGraph:
    return ResolutionDecidableSupervisionGraph(
        preferred_diameter_cells=args.ownership_preferred_cells,
        sigma=args.ownership_sigma,
        min_decidability=args.ownership_min_decidability,
        interval_ratio=args.ownership_interval_ratio,
        mode=mode,
        fallback=args.ownership_fallback,
    )


def audit_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    split_file = resolve_split_file(dataset_dir, args.split, args.split_file)
    names = read_names(split_file, args.max_samples)
    methods = parse_csv(args.methods)
    if not methods:
        raise ValueError("--methods cannot be empty")

    graphs = {method: make_graph(args, method) for method in methods}
    counters = {
        method: new_counters(len(graph.strides))
        for method, graph in graphs.items()
    }
    for name in names:
        instance_map = load_instance_map(dataset_dir / "masks" / f"{name}.png")
        batched = instance_map.unsqueeze(0)
        for method, graph in graphs.items():
            update_counters(counters[method], graph(batched))

    methods_report = {
        method: finalize_counters(
            values,
            collapse_threshold=args.collapse_threshold,
        )
        for method, values in counters.items()
    }
    return {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "split_file": str(split_file),
        "sample_count": len(names),
        "methods": methods_report,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RODS assignment audit",
        "",
        f"- Dataset: `{report['dataset_dir']}`",
        f"- Split: `{report['split']}`",
        f"- Split file: `{report['split_file']}`",
        f"- Samples: {report['sample_count']}",
        "",
        "| Method | Components | Side0 only | Coarse owned | Primary owner ratios | Owned ratios | Collapsed |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for method, item in report["methods"].items():
        primary = ", ".join(f"{value:.3f}" for value in item["primary_ratio"])
        owned = ", ".join(f"{value:.3f}" for value in item["owned_ratio"])
        lines.append(
            "| {method} | {components} | {side0:.3f} | {coarse:.3f} | {primary} | {owned} | {collapsed} |".format(
                method=method,
                components=item["component_count"],
                side0=item["side0_only_ratio"],
                coarse=item["coarse_owned_ratio"],
                primary=primary,
                owned=owned,
                collapsed="yes" if item["collapsed"] else "no",
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
