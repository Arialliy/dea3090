#!/usr/bin/env python3
"""Summarize RODS/control development-holdout runs.

This tool intentionally does not load checkpoints or construct models.  It can
run on a lightweight environment to inspect scheduler state, epoch logs, and
paired deltas against ``legacy_exact`` for the same dataset/seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.finalize_clean_baselines import (  # noqa: E402
    FinalizationError,
    parse_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize RODS/control jobs and paired deltas."
    )
    parser.add_argument("--batch-id", default="rods_controls_v1")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def best_row(rows: list[dict[str, float | int]]) -> dict[str, float | int] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["iou"]))


def load_result(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def summarize_job(job: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(job["run_dir"])
    result = load_result(Path(job["result_file"]))
    try:
        rows = parse_metrics(run_dir / "epoch_metric.log")
    except FinalizationError:
        rows = []

    if result is not None:
        status = "completed" if result.get("returncode") == 0 else "failed"
    elif rows:
        status = "running_or_interrupted"
    else:
        status = "pending"
    latest = rows[-1] if rows else None
    best = best_row(rows)
    return {
        "job_id": job["job_id"],
        "method": job["method"],
        "dataset": job["dataset"],
        "seed": job["seed"],
        "status": status,
        "epochs_recorded": len(rows),
        "latest": latest,
        "best": best,
        "returncode": result.get("returncode") if result else None,
        "run_dir": job["run_dir"],
    }


def method_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None}
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values)}


def build_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    jobs = [summarize_job(job) for job in manifest["jobs"]]
    best_by_key = {
        (job["method"], job["dataset"], job["seed"]): job["best"]
        for job in jobs
        if job["best"] is not None
    }

    paired_deltas = []
    for job in jobs:
        if job["method"] == "legacy_exact" or job["best"] is None:
            continue
        baseline = best_by_key.get(("legacy_exact", job["dataset"], job["seed"]))
        if baseline is None:
            continue
        best = job["best"]
        paired_deltas.append(
            {
                "method": job["method"],
                "dataset": job["dataset"],
                "seed": job["seed"],
                "delta_iou": float(best["iou"]) - float(baseline["iou"]),
                "delta_pd": float(best["pd"]) - float(baseline["pd"]),
                "delta_fa": float(best["fa"]) - float(baseline["fa"]),
            }
        )

    aggregate: dict[str, dict[str, Any]] = {}
    methods = sorted({job["method"] for job in jobs})
    datasets = sorted({job["dataset"] for job in jobs})
    for method in methods:
        aggregate[method] = {}
        for dataset in datasets:
            rows = [
                job["best"]
                for job in jobs
                if job["method"] == method
                and job["dataset"] == dataset
                and job["best"] is not None
            ]
            aggregate[method][dataset] = {
                "count": len(rows),
                "iou": method_stats([float(row["iou"]) for row in rows]),
                "pd": method_stats([float(row["pd"]) for row in rows]),
                "fa": method_stats([float(row["fa"]) for row in rows]),
            }

    delta_aggregate: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in paired_deltas}):
        delta_aggregate[method] = {}
        for dataset in datasets:
            rows = [
                row
                for row in paired_deltas
                if row["method"] == method and row["dataset"] == dataset
            ]
            delta_aggregate[method][dataset] = {
                "count": len(rows),
                "delta_iou": method_stats([float(row["delta_iou"]) for row in rows]),
                "delta_pd": method_stats([float(row["delta_pd"]) for row in rows]),
                "delta_fa": method_stats([float(row["delta_fa"]) for row in rows]),
            }

    return {
        "batch_id": manifest["batch_id"],
        "stage": manifest["stage"],
        "official_test_policy": manifest["official_test_policy"],
        "methods": manifest.get("methods", []),
        "jobs": jobs,
        "aggregate": aggregate,
        "paired_deltas_vs_legacy_exact": paired_deltas,
        "delta_aggregate_vs_legacy_exact": delta_aggregate,
    }


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RODS/control development-holdout summary",
        "",
        "> Scope guard: these are internal development-holdout results. Official test data is not iterated by this workflow.",
        "",
        f"- Batch: `{summary['batch_id']}`",
        f"- Stage: `{summary['stage']}`",
        f"- Official-test policy: {summary['official_test_policy']}",
        "",
        "## Job Status",
        "",
        "| Method | Dataset | Seed | Status | Epochs | Best IoU | Best PD | Best FA/M |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for job in summary["jobs"]:
        best = job["best"] or {}
        lines.append(
            "| {method} | {dataset} | {seed} | {status} | {epochs} | {iou} | {pd} | {fa} |".format(
                method=job["method"],
                dataset=job["dataset"],
                seed=job["seed"],
                status=job["status"],
                epochs=job["epochs_recorded"],
                iou=fmt(best.get("iou")),
                pd=fmt(best.get("pd")),
                fa=fmt(best.get("fa"), digits=3),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Delta Vs legacy_exact",
            "",
            "| Method | Dataset | Seed | Delta IoU | Delta PD | Delta FA/M |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["paired_deltas_vs_legacy_exact"]:
        lines.append(
            "| {method} | {dataset} | {seed} | {diou} | {dpd} | {dfa} |".format(
                method=row["method"],
                dataset=row["dataset"],
                seed=row["seed"],
                diou=fmt(row["delta_iou"]),
                dpd=fmt(row["delta_pd"]),
                dfa=fmt(row["delta_fa"], digits=3),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(report_root: Path, summary: dict[str, Any]) -> None:
    json_path = report_root / "rods_controls_summary.json"
    md_path = report_root / "rods_controls_summary.md"
    json_path.write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(build_markdown(summary), encoding="utf-8")


def main() -> int:
    args = parse_args()
    report_root = PROJECT_DIR / "repro_runs" / "rods" / args.batch_id
    manifest_path = report_root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    summary = build_summary(manifest)
    if args.require_complete:
        incomplete = [
            job["job_id"]
            for job in summary["jobs"]
            if job["status"] != "completed"
        ]
        if incomplete:
            raise SystemExit("incomplete jobs: " + ", ".join(incomplete))
    if args.write:
        write_outputs(report_root, summary)
    if args.as_json:
        print(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True))
    else:
        print(build_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
