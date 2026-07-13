#!/usr/bin/env python3
"""Fail-closed paired summary for formal MSHNet/SDRR validation runs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


LINE_RE = re.compile(
    r"(?P<epoch>\d+)\s+-\s+IoU\s+(?P<iou>[0-9.]+)\s+-\s+"
    r"PD\s+(?P<pd>[0-9.]+)\s+-\s+FA\s+(?P<fa>[0-9.]+)"
)


def parse_metric_log(path: Path, expected_epochs: int) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise ValueError(f"missing metric log: {path}")
    invalid_marker = path.parent / "INVALID_RUN.json"
    if invalid_marker.is_file():
        raise ValueError(
            "run is explicitly marked invalid and must not be summarized: "
            f"{invalid_marker}"
        )
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LINE_RE.search(line.replace("\t", " "))
        if match is None:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "IoU": float(match.group("iou")),
                "PD": float(match.group("pd")),
                "FA": float(match.group("fa")),
            }
        )
    epochs = [int(row["epoch"]) for row in rows]
    expected = list(range(expected_epochs))
    if epochs != expected:
        raise ValueError(
            f"incomplete or non-canonical epochs in {path}: "
            f"expected 0..{expected_epochs - 1}, got {epochs[:1]}..{epochs[-1:]}, "
            f"rows={len(epochs)}"
        )
    return rows


def best_iou_row(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    # max() deliberately keeps the first occurrence, matching checkpoint saving
    # under the trainer's strict `>` best-IoU condition.
    return dict(max(rows, key=lambda row: float(row["IoU"])))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _mean_metric_row(
    rows: list[dict[str, float | int]],
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty trajectory window")
    return {
        metric: _mean([float(row[metric]) for row in rows])
        for metric in ("IoU", "PD", "FA")
    }


def trajectory_summary(
    rows: list[dict[str, float | int]],
    intervention_epoch: int,
) -> dict[str, Any]:
    if not 0 <= intervention_epoch < len(rows):
        raise ValueError("intervention_epoch must index the formal trajectory")
    return {
        "final_epoch": dict(rows[-1]),
        "last_20_mean": _mean_metric_row(rows[-20:]),
        "last_50_mean": _mean_metric_row(rows[-50:]),
        "post_intervention_mean": _mean_metric_row(rows[intervention_epoch:]),
    }


def summarize(
    root: Path,
    dataset: str,
    seeds: list[int],
    expected_epochs: int,
    intervention_epoch: int | None = None,
    baseline_template: str = "formal_baseline_{dataset}_seed{seed}_e{epochs}",
    sdrr_template: str = "formal_sdrr_{dataset}_seed{seed}_e{epochs}",
) -> dict[str, Any]:
    if intervention_epoch is None:
        intervention_epoch = int(round(0.625 * expected_epochs))
        intervention_epoch = min(expected_epochs - 1, intervention_epoch)
    pairs: list[dict[str, Any]] = []
    for seed in seeds:
        method_rows: dict[str, dict[str, float | int]] = {}
        method_trajectories: dict[str, dict[str, Any]] = {}
        for method in ("baseline", "sdrr"):
            template = baseline_template if method == "baseline" else sdrr_template
            run = root / template.format(
                dataset=dataset,
                seed=seed,
                epochs=expected_epochs,
                method=method,
            )
            rows = parse_metric_log(run / "epoch_metric.log", expected_epochs)
            method_rows[method] = best_iou_row(rows)
            method_trajectories[method] = trajectory_summary(
                rows, intervention_epoch
            )
        baseline = method_rows["baseline"]
        sdrr = method_rows["sdrr"]
        pairs.append(
            {
                "seed": seed,
                "baseline": baseline,
                "sdrr": sdrr,
                "delta": {
                    metric: float(sdrr[metric]) - float(baseline[metric])
                    for metric in ("IoU", "PD", "FA")
                },
                "trajectory": method_trajectories,
                "trajectory_delta": {
                    window: {
                        metric: float(method_trajectories["sdrr"][window][metric])
                        - float(method_trajectories["baseline"][window][metric])
                        for metric in ("IoU", "PD", "FA")
                    }
                    for window in (
                        "final_epoch",
                        "last_20_mean",
                        "last_50_mean",
                        "post_intervention_mean",
                    )
                },
            }
        )

    aggregate: dict[str, Any] = {}
    for method in ("baseline", "sdrr"):
        aggregate[method] = {}
        for metric in ("IoU", "PD", "FA"):
            values = [float(pair[method][metric]) for pair in pairs]
            aggregate[method][metric] = {
                "mean": _mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    aggregate["paired_delta"] = {}
    for metric in ("IoU", "PD", "FA"):
        values = [float(pair["delta"][metric]) for pair in pairs]
        aggregate["paired_delta"][metric] = {
            "mean": _mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "positive_seeds": sum(value > 0.0 for value in values),
            "negative_seeds": sum(value < 0.0 for value in values),
        }
    trajectory_aggregate: dict[str, Any] = {}
    for window in (
        "final_epoch",
        "last_20_mean",
        "last_50_mean",
        "post_intervention_mean",
    ):
        trajectory_aggregate[window] = {}
        for method in ("baseline", "sdrr"):
            trajectory_aggregate[window][method] = {}
            for metric in ("IoU", "PD", "FA"):
                values = [
                    float(pair["trajectory"][method][window][metric])
                    for pair in pairs
                ]
                trajectory_aggregate[window][method][metric] = {
                    "mean": _mean(values),
                    "sample_sd": (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    ),
                }
        trajectory_aggregate[window]["paired_delta"] = {}
        for metric in ("IoU", "PD", "FA"):
            values = [
                float(pair["trajectory_delta"][window][metric])
                for pair in pairs
            ]
            trajectory_aggregate[window]["paired_delta"][metric] = {
                "mean": _mean(values),
                "sample_sd": (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                ),
                "positive_seeds": sum(value > 0.0 for value in values),
                "negative_seeds": sum(value < 0.0 for value in values),
            }
    return {
        "dataset": dataset,
        "selection": "per-run best internal-validation IoU",
        "expected_epochs": expected_epochs,
        "intervention_epoch": intervention_epoch,
        "seeds": seeds,
        "pairs": pairs,
        "aggregate": aggregate,
        "trajectory_aggregate": trajectory_aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("repro_runs"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--intervention-epoch", type=int)
    parser.add_argument(
        "--baseline-template",
        default="formal_baseline_{dataset}_seed{seed}_e{epochs}",
    )
    parser.add_argument(
        "--sdrr-template",
        default="formal_sdrr_{dataset}_seed{seed}_e{epochs}",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        args.root,
        args.dataset,
        args.seeds,
        args.epochs,
        intervention_epoch=args.intervention_epoch,
        baseline_template=args.baseline_template,
        sdrr_template=args.sdrr_template,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
