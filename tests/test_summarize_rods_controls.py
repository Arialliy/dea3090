from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_rods_controls import build_markdown, build_summary


def write_metric_log(path: Path, rows: list[tuple[int, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "2026-07-11-00-00-00 - %04d\t - IoU %.4f\t - PD %.4f\t - FA %.4f"
        % row
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_rods_summary_reports_paired_delta(tmp_path: Path) -> None:
    jobs = []
    for method, rows in (
        ("legacy_exact", [(0, 0.5, 0.80, 20.0), (1, 0.6, 0.85, 18.0)]),
        ("rods_interval", [(0, 0.55, 0.81, 19.0), (1, 0.62, 0.86, 15.0)]),
    ):
        job_id = f"{method}__nuaa-sirst__seed_1"
        run_dir = tmp_path / "weights" / method
        result_file = tmp_path / "jobs" / f"{job_id}.json"
        write_metric_log(run_dir / "epoch_metric.log", rows)
        write_json(result_file, {"returncode": 0})
        jobs.append(
            {
                "job_id": job_id,
                "method": method,
                "dataset": "NUAA-SIRST",
                "seed": 1,
                "run_dir": str(run_dir),
                "result_file": str(result_file),
            }
        )

    summary = build_summary(
        {
            "batch_id": "fixture",
            "stage": "development_holdout_rods_controls",
            "official_test_policy": "loaded only for disjoint/hash audit; not iterated",
            "methods": ["legacy_exact", "rods_interval"],
            "jobs": jobs,
        }
    )

    delta = summary["paired_deltas_vs_legacy_exact"][0]
    assert delta["delta_iou"] == pytest.approx(0.02)
    assert delta["delta_pd"] == pytest.approx(0.01)
    assert delta["delta_fa"] == pytest.approx(-3.0)
    assert "Delta IoU" in build_markdown(summary)
