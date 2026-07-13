#!/usr/bin/env python3
"""Fail-closed comparison of each run's independently selected best checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch


METRIC_RE = re.compile(
    r"(?P<epoch>\d+)\s+-\s+IoU\s+(?P<iou>[0-9.]+)\s+-\s+"
    r"PD\s+(?P<pd>[0-9.]+)\s+-\s+FA\s+(?P<fa>[0-9.]+)"
)
LOG_ROUNDING_TOLERANCE = 5.1e-5


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise ValueError(f"missing run config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"run config has no args mapping: {path}")
    return args


def expected_evaluation_epochs(args: dict[str, Any]) -> list[int]:
    epochs = int(args["epochs"])
    interval = int(args["evaluation_interval"])
    if epochs < 1 or interval < 1:
        raise ValueError("epochs and evaluation_interval must be positive")
    expected = list(range(interval - 1, epochs, interval))
    if not bool(args.get("skip_final_evaluation", False)) and epochs - 1 not in expected:
        expected.append(epochs - 1)
    return expected


def _read_metric_rows(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise ValueError(f"missing metric log: {path}")
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_RE.search(line.replace("\t", " "))
        if match is None:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "iou": float(match.group("iou")),
                "pd": float(match.group("pd")),
                "fa": float(match.group("fa")),
            }
        )
    return rows


def summarize_independent_best(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    invalid = run_dir / "INVALID_RUN.json"
    if invalid.exists():
        raise ValueError(f"run is explicitly invalid: {invalid}")

    args = _read_run_config(run_dir)
    expected = expected_evaluation_epochs(args)
    rows = _read_metric_rows(run_dir / "epoch_metric.log")
    logged_epochs = [int(row["epoch"]) for row in rows]
    if logged_epochs != expected:
        raise ValueError(
            f"evaluation schedule mismatch for {run_dir}: "
            f"expected {expected}, got {logged_epochs}"
        )

    checkpoint_path = run_dir / "checkpoint_best_iou.pkl"
    if not checkpoint_path.is_file():
        raise ValueError(f"missing independently selected best checkpoint: {checkpoint_path}")
    checkpoint = _load_torch(checkpoint_path)
    required = ("epoch", "iou", "pd", "fa", "best_iou")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"best checkpoint is missing fields {missing}: {checkpoint_path}")

    epoch = int(checkpoint["epoch"])
    by_epoch = {int(row["epoch"]): row for row in rows}
    if epoch not in by_epoch:
        raise ValueError(f"best checkpoint epoch {epoch} was not an evaluation epoch")
    row = by_epoch[epoch]
    for key in ("iou", "pd", "fa"):
        if abs(float(checkpoint[key]) - float(row[key])) > LOG_ROUNDING_TOLERANCE:
            raise ValueError(
                f"checkpoint/log {key} mismatch at epoch {epoch}: "
                f"{checkpoint[key]} != {row[key]}"
            )
    if abs(float(checkpoint["iou"]) - float(checkpoint["best_iou"])) > 1e-12:
        raise ValueError("checkpoint iou and best_iou disagree")

    max_logged_iou = max(float(item["iou"]) for item in rows)
    if float(row["iou"]) != max_logged_iou:
        raise ValueError(
            f"checkpoint epoch {epoch} is not a best-IoU evaluation: "
            f"logged {row['iou']} < {max_logged_iou}"
        )

    return {
        "run_dir": str(run_dir),
        "variant": args.get("mshnet_variant"),
        "seed": int(args["seed"]),
        "dataset_dir": args.get("dataset_dir"),
        "evaluation_protocol": args.get("evaluation_protocol"),
        "train_split_sha256": args.get("train_split_sha256"),
        "test_split_sha256": args.get("test_split_sha256"),
        "epochs": int(args["epochs"]),
        "evaluation_interval": int(args["evaluation_interval"]),
        "evaluated_epochs": expected,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "best": {
            "epoch": epoch,
            "iou": float(checkpoint["iou"]),
            "pd": float(checkpoint["pd"]),
            "fa": float(checkpoint["fa"]),
        },
    }


def compare_runs(baseline_run: Path, candidate_run: Path) -> dict[str, Any]:
    baseline = summarize_independent_best(baseline_run)
    candidate = summarize_independent_best(candidate_run)
    paired_fields = (
        "seed",
        "dataset_dir",
        "evaluation_protocol",
        "train_split_sha256",
        "test_split_sha256",
        "epochs",
        "evaluation_interval",
        "evaluated_epochs",
    )
    mismatches = {
        field: {"baseline": baseline[field], "candidate": candidate[field]}
        for field in paired_fields
        if baseline[field] != candidate[field]
    }
    if mismatches:
        raise ValueError(f"runs do not form a paired protocol: {mismatches}")
    baseline_best = baseline["best"]
    candidate_best = candidate["best"]
    return {
        "selection_rule": "independent_per_run_best_iou_checkpoint",
        "same_epoch_required": False,
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {
            key: float(candidate_best[key]) - float(baseline_best[key])
            for key in ("iou", "pd", "fa")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare_runs(args.baseline_run, args.candidate_run)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
