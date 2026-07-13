from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools.compare_independent_best import (
    METHOD_META_ARG_FIELDS,
    compare_runs,
    summarize_independent_best,
)


def _split_hash(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode()).hexdigest()


def _write_run(
    root: Path,
    name: str,
    rows: list[tuple[int, float, float, float]],
    best_index: int,
) -> Path:
    run = root / name
    run.mkdir()
    dataset = root / "dataset"
    manifest_dir = dataset / "img_idx"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    train_names = ["train_a", "train_b"]
    test_names = ["test_a", "test_b"]
    train_text = "\n".join(train_names) + "\n"
    test_text = "\n".join(test_names) + "\n"
    (manifest_dir / "train_fixture.txt").write_text(train_text, encoding="utf-8")
    (manifest_dir / "test_fixture.txt").write_text(test_text, encoding="utf-8")
    (run / "split_train.txt").write_text(train_text, encoding="utf-8")
    (run / "split_test.txt").write_text(test_text, encoding="utf-8")
    run_args = {
        "epochs": 30,
        "evaluation_interval": 10,
        "skip_final_evaluation": False,
        "model_type": "mshnet",
        "mshnet_variant": name,
        "seed": 7,
        "dataset_dir": str(dataset),
        "evaluation_protocol": "official_train_test",
        "train_split_file": "img_idx/train_fixture.txt",
        "val_split_file": "",
        "test_split_file": "img_idx/test_fixture.txt",
        "train_split_sha256": _split_hash(train_names),
        "val_split_sha256": _split_hash(test_names),
        "test_split_sha256": _split_hash(test_names),
        "deep_supervision": "legacy_exact",
        "fusion_regularizer": "none",
        "aux_loss_weight": 0.8,
        "empty_side_policy": "skip",
        "dea_lambda_single": 0.0,
        "dea_lambda_dec": 0.0,
        "dea_lambda_empty": 0.0,
        "dea_ramp_epochs": 0,
        "dea_tau": 0.5,
        "lr": 0.05,
        "warm_epoch": 5,
        "batch_size": 4,
        "num_workers": 0,
        "base_size": 256,
        "crop_size": 256,
        "deterministic": True,
        "val_fraction": 0.2,
        "split_seed": 11,
        "init_from_baseline": "",
        "origin_baseline_checkpoint": "",
    }
    method_meta = {field: run_args[field] for field in METHOD_META_ARG_FIELDS}
    (run / "run_config.json").write_text(
        json.dumps(
            {
                "args": run_args,
                "method_meta": method_meta,
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
            "method_meta": method_meta,
        },
        run / "checkpoint_best_iou.pkl",
    )
    global_best_iou = max(row[1] for row in rows)
    torch.save(
        {
            "epoch": rows[-1][0],
            "iou": rows[-1][1],
            "pd": rows[-1][2],
            "fa": rows[-1][3],
            "best_iou": global_best_iou,
            "net": {},
            "method_meta": method_meta,
        },
        run / "checkpoint.pkl",
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

    with pytest.raises(ValueError, match="exact global best_iou"):
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
    config["method_meta"]["seed"] = 8
    config_path.write_text(json.dumps(config), encoding="utf-8")
    for filename in ("checkpoint_best_iou.pkl", "checkpoint.pkl"):
        checkpoint_path = candidate / filename
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["method_meta"]["seed"] = 8
        torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="paired protocol"):
        compare_runs(baseline, candidate)


def test_comparison_rejects_checkpoint_variant_relabelling(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "candidate",
        [(9, 0.60, 0.90, 30.0), (19, 0.72, 0.94, 22.0), (29, 0.70, 0.95, 21.0)],
        best_index=1,
    )
    checkpoint_path = run / "checkpoint_best_iou.pkl"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["method_meta"]["mshnet_variant"] = "deterministic"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="method_meta does not match"):
        summarize_independent_best(run)


def test_comparison_rejects_tampered_split_snapshot(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "candidate",
        [(9, 0.60, 0.90, 30.0), (19, 0.72, 0.94, 22.0), (29, 0.70, 0.95, 21.0)],
        best_index=1,
    )
    with (run / "split_test.txt").open("a", encoding="utf-8") as handle:
        handle.write("test_tampered\n")

    with pytest.raises(ValueError, match="snapshot/hash mismatch"):
        summarize_independent_best(run)


def test_comparison_rejects_training_configuration_mismatch(tmp_path: Path) -> None:
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
    config["args"]["lr"] = 0.01
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="paired protocol"):
        compare_runs(baseline, candidate)


def test_comparison_requires_the_formal_ten_epoch_schedule(tmp_path: Path) -> None:
    run = _write_run(
        tmp_path,
        "candidate",
        [(9, 0.60, 0.90, 30.0), (19, 0.72, 0.94, 22.0), (29, 0.70, 0.95, 21.0)],
        best_index=1,
    )
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["args"]["evaluation_interval"] = 20
    config["method_meta"]["evaluation_interval"] = 20
    config_path.write_text(json.dumps(config), encoding="utf-8")
    for filename in ("checkpoint_best_iou.pkl", "checkpoint.pkl"):
        checkpoint_path = run / filename
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["method_meta"]["evaluation_interval"] = 20
        torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="evaluation_interval=10"):
        summarize_independent_best(run)
