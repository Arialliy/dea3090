from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.compare_independent_best import compare_runs, summarize_independent_best


def _write_run(
    root: Path,
    name: str,
    rows: list[tuple[int, float, float, float]],
    best_index: int,
) -> Path:
    run = root / name
    run.mkdir()
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "args": {
                    "epochs": 30,
                    "evaluation_interval": 10,
                    "skip_final_evaluation": False,
                    "mshnet_variant": name,
                    "seed": 7,
                    "dataset_dir": "/dataset",
                    "evaluation_protocol": "official_train_test",
                    "train_split_sha256": "train-hash",
                    "test_split_sha256": "test-hash",
                }
            }
        ),
        encoding="utf-8",
    )
    (run / "epoch_metric.log").write_text(
        "".join(
            f"2026-01-01 - {epoch:04d} - IoU {iou:.4f} - PD {pd:.4f} - FA {fa:.4f}\n"
            for epoch, iou, pd, fa in rows
        ),
        encoding="utf-8",
    )
    epoch, iou, pd, fa = rows[best_index]
    torch.save(
        {
            "epoch": epoch,
            "iou": iou,
            "pd": pd,
            "fa": fa,
            "best_iou": iou,
            "net": {},
        },
        run / "checkpoint_best_iou.pkl",
    )
    return run


def test_compare_allows_each_run_to_select_a_different_best_epoch(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        [(9, 0.60, 0.90, 30.0), (19, 0.72, 0.94, 22.0), (29, 0.70, 0.95, 21.0)],
        best_index=1,
    )
    candidate = _write_run(
        tmp_path,
        "candidate",
        [(9, 0.61, 0.91, 28.0), (19, 0.71, 0.95, 20.0), (29, 0.73, 0.96, 18.0)],
        best_index=2,
    )

    result = compare_runs(baseline, candidate)

    assert result["selection_rule"] == "independent_per_run_best_iou_checkpoint"
    assert result["same_epoch_required"] is False
    assert result["baseline"]["best"]["epoch"] == 19
    assert result["candidate"]["best"]["epoch"] == 29
    assert result["delta_candidate_minus_baseline"]["iou"] == pytest.approx(0.01)


def test_comparison_rejects_a_checkpoint_that_is_not_the_run_best(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "wrong",
        [(9, 0.60, 0.90, 30.0), (19, 0.72, 0.94, 22.0), (29, 0.70, 0.95, 21.0)],
        best_index=2,
    )

    with pytest.raises(ValueError, match="not a best-IoU evaluation"):
        summarize_independent_best(run)


def test_comparison_rejects_unpaired_seeds(tmp_path: Path) -> None:
    baseline = _write_run(
        tmp_path,
        "baseline",
        [(9, 0.60, 0.90, 30.0), (19, 0.72, 0.94, 22.0), (29, 0.70, 0.95, 21.0)],
        best_index=1,
    )
    candidate = _write_run(
        tmp_path,
        "candidate",
        [(9, 0.61, 0.91, 28.0), (19, 0.71, 0.95, 20.0), (29, 0.73, 0.96, 18.0)],
        best_index=2,
    )
    config_path = candidate / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["args"]["seed"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="paired protocol"):
        compare_runs(baseline, candidate)
