from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch

import tools.finalize_trace_stage0_baseline as finalizer


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def metric_line(epoch: int, iou: float, pd: float, fa: float) -> str:
    return (
        f"2026-07-13-00-00-00 - {epoch:04d}\t - IoU {iou:.4f}"
        f"\t - PD {pd:.4f}\t - FA {fa:.4f}"
    )


def method_metadata(job: dict) -> dict:
    return {
        "method": "MSHNet-Deterministic",
        "model_type": "mshnet",
        "mshnet_variant": "deterministic",
        "evaluation_protocol": "internal_holdout",
        "deep_supervision": "legacy_exact",
        "fusion_regularizer": "none",
        "deterministic": True,
        "evaluation_interval": finalizer.EXPECTED_EVALUATION_INTERVAL,
        "skip_final_evaluation": False,
        "init_from_baseline": "",
        "dea_lambda_single": 0.0,
        "dea_lambda_dec": 0.0,
        "dea_lambda_empty": 0.0,
        "seed": job["seed"],
        "run_label": job["job_id"],
        "split_seed": finalizer.EXPECTED_BATCH_ARGS["split_seed"],
        "dataset_dir": job["dataset_dir"],
        "train_split_file": job["train_file"],
        "val_split_file": "",
        "test_split_file": job["test_file"],
        "train_split_sha256": finalizer.EXPECTED_DATASET_HASHES["fit_sha256"],
        "val_split_sha256": finalizer.EXPECTED_DATASET_HASHES["val_sha256"],
        "test_split_sha256": finalizer.EXPECTED_DATASET_HASHES[
            "official_test_sha256"
        ],
    }


def run_args(job: dict) -> dict:
    return {
        "mode": "train",
        "model_type": "mshnet",
        "mshnet_variant": "deterministic",
        "evaluation_protocol": "internal_holdout",
        "deep_supervision": "legacy_exact",
        "fusion_regularizer": "none",
        "deterministic": True,
        "evaluation_interval": finalizer.EXPECTED_EVALUATION_INTERVAL,
        "skip_final_evaluation": False,
        "epochs": finalizer.EXPECTED_EPOCHS,
        "batch_size": finalizer.EXPECTED_BATCH_ARGS["batch_size"],
        "num_workers": finalizer.EXPECTED_BATCH_ARGS["num_workers"],
        "lr": finalizer.EXPECTED_BATCH_ARGS["lr"],
        "warm_epoch": finalizer.EXPECTED_BATCH_ARGS["warm_epoch"],
        "val_fraction": finalizer.EXPECTED_BATCH_ARGS["val_fraction"],
        "split_seed": finalizer.EXPECTED_BATCH_ARGS["split_seed"],
        "seed": job["seed"],
        "run_label": job["job_id"],
        "run_dir": job["run_dir"],
        "dataset_dir": job["dataset_dir"],
        "train_split_file": job["train_file"],
        "val_split_file": "",
        "test_split_file": job["test_file"],
        "train_split_sha256": finalizer.EXPECTED_DATASET_HASHES["fit_sha256"],
        "val_split_sha256": finalizer.EXPECTED_DATASET_HASHES["val_sha256"],
        "test_split_sha256": finalizer.EXPECTED_DATASET_HASHES[
            "official_test_sha256"
        ],
        "if_checkpoint": False,
        "checkpoint_dir": "",
        "reset_optimizer": False,
        "init_from_baseline": "",
        "origin_baseline_checkpoint": "",
        "multi_gpus": False,
        "gpu_ids": "",
        "dea_lambda_single": 0.0,
        "dea_lambda_dec": 0.0,
        "dea_lambda_empty": 0.0,
    }


def fake_source_validator(_manifest: dict, _project_dir: Path) -> dict:
    return {
        "canonical_source_commit": finalizer.EXPECTED_CANONICAL_SOURCE_COMMIT,
        "canonical_source_sha256": finalizer.EXPECTED_HISTORICAL_SOURCE_SHA256,
        "launch_repository_head": finalizer.EXPECTED_LAUNCH_REPOSITORY_HEAD,
        **finalizer.EXPECTED_SOURCE_SHA256,
    }


def fake_runtime_attestation_validator(_batch_dir: Path) -> dict:
    return {
        "path": "/fixture/runtime_attestation.json",
        "sha256": "a" * 64,
        "capture_started_at": "2026-07-13T01:30:00+00:00",
        "capture_finished_at": "2026-07-13T01:31:00+00:00",
        "capture_scope": "during-run, not launch-time",
        "source_dependency_file_count": 34,
        "source_dependency_aggregate_sha256": "b" * 64,
        "training_data_aggregates": {
            "fit": "c" * 64,
            "validation": "d" * 64,
            "official_train": "e" * 64,
        },
        "official_test_pixel_files_opened_by_baseline_attester": 0,
        "workers": {
            seed: {
                "pid": 1000 + index,
                "argv0": sys.executable,
                "exe": str(Path(sys.executable).resolve()),
            }
            for index, seed in enumerate(finalizer.EXPECTED_SEEDS)
        },
    }


def install_synthetic_dataset_contract(monkeypatch):
    """Replace private-data constants with a self-contained split fixture."""

    official_train = [f"train_{index:03d}" for index in range(10)]
    official_test = [f"test_{index:03d}" for index in range(7)]
    fit_names, val_names = finalizer.deterministic_split(
        official_train,
        finalizer.EXPECTED_BATCH_ARGS["split_seed"],
        finalizer.EXPECTED_BATCH_ARGS["val_fraction"],
    )
    counts = {
        "official_train_count": len(official_train),
        "fit_count": len(fit_names),
        "val_count": len(val_names),
        "official_test_count": len(official_test),
    }
    hashes = {
        "official_train_sha256": finalizer.split_hash(official_train),
        "fit_sha256": finalizer.split_hash(fit_names),
        "val_sha256": finalizer.split_hash(val_names),
        "official_test_sha256": finalizer.split_hash(official_test),
    }
    monkeypatch.setattr(finalizer, "EXPECTED_DATASET_COUNTS", counts)
    monkeypatch.setattr(finalizer, "EXPECTED_DATASET_HASHES", hashes)
    return official_train, official_test, fit_names, val_names


def make_complete_batch(tmp_path: Path, monkeypatch):
    project_dir = tmp_path / "project"
    dataset_dir = project_dir / "datasets" / finalizer.DATASET_NAME
    (dataset_dir / "img_idx").mkdir(parents=True)
    official_train, official_test, fit_names, val_names = (
        install_synthetic_dataset_contract(monkeypatch)
    )
    (dataset_dir / "img_idx" / f"train_{finalizer.DATASET_NAME}.txt").write_text(
        "\n".join(official_train) + "\n", encoding="utf-8"
    )
    (dataset_dir / "img_idx" / f"test_{finalizer.DATASET_NAME}.txt").write_text(
        "\n".join(official_test) + "\n", encoding="utf-8"
    )
    (project_dir / "main.py").write_text("# fixture\n", encoding="utf-8")
    batch_dir = (
        project_dir
        / "repro_runs"
        / "clean"
        / finalizer.EXPECTED_BATCH_ID
    )
    dataset_meta = {
        "dataset": finalizer.DATASET_NAME,
        "dataset_dir": str(dataset_dir.resolve()),
        "train_file": f"img_idx/train_{finalizer.DATASET_NAME}.txt",
        "test_file": f"img_idx/test_{finalizer.DATASET_NAME}.txt",
        **finalizer.EXPECTED_DATASET_COUNTS,
        **finalizer.EXPECTED_DATASET_HASHES,
    }
    jobs = []
    checkpoints: dict[Path, dict] = {}
    for seed_index, seed in enumerate(finalizer.EXPECTED_SEEDS):
        job_id = f"mshnet__{finalizer.DATASET_NAME.lower()}__seed_{seed}"
        run_dir = (
            project_dir
            / "weight"
            / "clean"
            / finalizer.EXPECTED_BATCH_ID
            / finalizer.DATASET_NAME
            / f"seed_{seed}"
        ).resolve()
        job = {
            "dataset": finalizer.DATASET_NAME,
            "dataset_dir": str(dataset_dir.resolve()),
            "job_id": job_id,
            "log_file": str((batch_dir / "logs" / f"{job_id}.log").resolve()),
            "result_file": str((batch_dir / "jobs" / f"{job_id}.json").resolve()),
            "run_dir": str(run_dir),
            "seed": seed,
            "test_file": f"img_idx/test_{finalizer.DATASET_NAME}.txt",
            "train_file": f"img_idx/train_{finalizer.DATASET_NAME}.txt",
        }
        jobs.append(job)
        run_dir.mkdir(parents=True)
        (run_dir / "split_train.txt").write_text(
            "\n".join(fit_names) + "\n", encoding="utf-8"
        )
        (run_dir / "split_val.txt").write_text(
            "\n".join(val_names) + "\n", encoding="utf-8"
        )
        metadata = method_metadata(job)
        write_json(
            run_dir / "run_config.json",
            {"args": run_args(job), "method_meta": metadata},
        )

        best_epoch = 199 + seed_index * 10
        best_iou = 0.7000 + seed_index * 0.0020
        best_pd = 0.9000 + seed_index * 0.0050
        best_fa = 30.0 - seed_index
        rows = []
        final_metrics = None
        for epoch in finalizer.EXPECTED_EVALUATION_EPOCHS:
            iou = 0.5000 + epoch / 10000.0
            pd = 0.8000 + seed_index * 0.0010
            fa = 50.0 - seed_index
            if epoch == best_epoch:
                iou, pd, fa = best_iou, best_pd, best_fa
            if epoch == finalizer.EXPECTED_EPOCHS - 1:
                final_metrics = (iou, pd, fa)
            rows.append(metric_line(epoch, iou, pd, fa))
        assert final_metrics is not None
        (run_dir / "epoch_metric.log").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )
        (run_dir / "metric.log").write_text(
            metric_line(best_epoch, best_iou, best_pd, best_fa) + "\n",
            encoding="utf-8",
        )

        state = {"weight": torch.ones(1)}
        best_path = run_dir / "checkpoint_best_iou.pkl"
        latest_path = run_dir / "checkpoint.pkl"
        best_path.touch()
        latest_path.touch()
        checkpoints[best_path] = {
            "net": state,
            "optimizer": {},
            "epoch": best_epoch,
            "iou": best_iou,
            "pd": best_pd,
            "fa": best_fa,
            "best_iou": best_iou,
            "method_meta": copy.deepcopy(metadata),
        }
        checkpoints[latest_path] = {
            "net": state,
            "optimizer": {},
            "epoch": finalizer.EXPECTED_EPOCHS - 1,
            "iou": final_metrics[0],
            "pd": final_metrics[1],
            "fa": final_metrics[2],
            "best_iou": best_iou,
            "method_meta": copy.deepcopy(metadata),
        }
        log_path = Path(job["log_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fixture clean training log\n", encoding="utf-8")
        write_json(
            Path(job["result_file"]),
            {
                "command": finalizer.expected_command(job, project_dir),
                "elapsed_seconds": 3600.0 + seed_index,
                "finished_at": f"2026-07-13T02:0{seed_index}:00+00:00",
                "gpu": seed_index,
                "job_id": job_id,
                "log_file": job["log_file"],
                "pid": 1000 + seed_index,
                "returncode": 0,
                "run_dir": job["run_dir"],
                "started_at": f"2026-07-13T01:0{seed_index}:00+00:00",
            },
        )

    manifest = {
        "args": finalizer.EXPECTED_BATCH_ARGS,
        "batch_id": finalizer.EXPECTED_BATCH_ID,
        "canonical_protocol": finalizer.EXPECTED_CANONICAL_PROTOCOL,
        "canonical_source_commit": finalizer.EXPECTED_CANONICAL_SOURCE_COMMIT,
        "created_at": "2026-07-13T00:00:00+00:00",
        "datasets": {finalizer.DATASET_NAME: dataset_meta},
        "jobs": jobs,
        "official_test_policy": finalizer.EXPECTED_TEST_POLICY,
        "provenance": {
            "repository_head": finalizer.EXPECTED_LAUNCH_REPOSITORY_HEAD,
            **finalizer.EXPECTED_SOURCE_SHA256,
        },
        "stage": finalizer.EXPECTED_STAGE,
    }
    write_json(batch_dir / "manifest.json", manifest)

    def checkpoint_loader(path: Path) -> dict:
        return checkpoints[path]

    kwargs = {
        "project_dir": project_dir,
        "checkpoint_loader": checkpoint_loader,
        "workspace_validator": fake_source_validator,
        "runtime_attestation_validator": fake_runtime_attestation_validator,
        "state_schema_loader": lambda: {"weight": ((1,), "torch.float32")},
    }
    return batch_dir, project_dir, checkpoints, kwargs


def assert_no_reports(batch_dir: Path) -> None:
    assert not (batch_dir / finalizer.OUTPUT_JSON).exists()
    assert not (batch_dir / finalizer.OUTPUT_MARKDOWN).exists()


def test_finalize_exact_single_dataset_batch_and_report_scope(tmp_path, monkeypatch):
    batch_dir, _, _, kwargs = make_complete_batch(tmp_path, monkeypatch)

    summary = finalizer.finalize_batch(batch_dir, **kwargs)

    assert summary["status"] == "complete_and_validated"
    assert summary["seeds"] == list(finalizer.EXPECTED_SEEDS)
    assert summary["selection"]["official_test_used_for_selection"] is False
    assert summary["official_test"]["evaluated"] is False
    assert summary["official_test"][
        "separate_task_definition_audit_inspected_test_masks"
    ] is True
    assert summary["aggregate"]["n"] == 3
    assert summary["aggregate"]["mean"]["iou"] == pytest.approx(0.702)
    assert summary["aggregate"]["sample_std"]["iou"] == pytest.approx(0.002)
    markdown = (batch_dir / finalizer.OUTPUT_MARKDOWN).read_text(encoding="utf-8")
    assert "internal NUAA-SIRST validation split" in markdown
    assert "must not claim that official-test annotations were globally sealed" in markdown
    json_payload = json.loads((batch_dir / finalizer.OUTPUT_JSON).read_text())
    assert json_payload["scope_guard"].startswith("Internal validation only")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        finalizer.finalize_batch(batch_dir, **kwargs)


def test_fail_closed_on_source_commit_before_writing(tmp_path, monkeypatch):
    batch_dir, _, _, kwargs = make_complete_batch(tmp_path, monkeypatch)
    manifest_path = batch_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["canonical_source_commit"] = "0" * 40
    write_json(manifest_path, manifest)

    with pytest.raises(finalizer.FinalizationError, match="source_commit"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)


def test_fail_closed_on_seed_set_and_split_hash(tmp_path, monkeypatch):
    batch_dir, _, _, kwargs = make_complete_batch(tmp_path, monkeypatch)
    manifest_path = batch_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["args"]["seeds"] = "1,2,3"
    write_json(manifest_path, manifest)

    with pytest.raises(finalizer.FinalizationError, match="frozen Stage-0 recipe"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)

    manifest["args"] = finalizer.EXPECTED_BATCH_ARGS
    manifest["datasets"][finalizer.DATASET_NAME]["fit_sha256"] = "0" * 64
    write_json(manifest_path, manifest)
    with pytest.raises(finalizer.FinalizationError, match="split hashes"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)


def test_fail_closed_on_resume_command_or_run_config(tmp_path, monkeypatch):
    batch_dir, _, _, kwargs = make_complete_batch(tmp_path, monkeypatch)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    first_job = manifest["jobs"][0]
    result_path = Path(first_job["result_file"])
    result = json.loads(result_path.read_text())
    result["command"].extend(["--if-checkpoint", "true"])
    write_json(result_path, result)

    with pytest.raises(finalizer.FinalizationError, match="fresh, no-resume"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)

    result["command"] = finalizer.expected_command(first_job, kwargs["project_dir"])
    write_json(result_path, result)
    config_path = Path(first_job["run_dir"]) / "run_config.json"
    config = json.loads(config_path.read_text())
    config["args"]["if_checkpoint"] = True
    write_json(config_path, config)
    with pytest.raises(finalizer.FinalizationError, match="canonical/fresh"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)


def test_fail_closed_on_incomplete_cadence_or_best_checkpoint(tmp_path, monkeypatch):
    batch_dir, _, checkpoints, kwargs = make_complete_batch(tmp_path, monkeypatch)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    first_job = manifest["jobs"][0]
    metric_path = Path(first_job["run_dir"]) / "epoch_metric.log"
    rows = metric_path.read_text().splitlines()
    metric_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(finalizer.FinalizationError, match="exactly 40"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)

    metric_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    best_path = Path(first_job["run_dir"]) / "checkpoint_best_iou.pkl"
    checkpoints[best_path]["iou"] = 0.1
    with pytest.raises(finalizer.FinalizationError, match="best_iou mismatch"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)


def test_fail_closed_on_run_config_checkpoint_metadata_disagreement(
    tmp_path, monkeypatch
):
    batch_dir, _, checkpoints, kwargs = make_complete_batch(tmp_path, monkeypatch)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    first_job = manifest["jobs"][0]
    best_path = Path(first_job["run_dir"]) / "checkpoint_best_iou.pkl"
    checkpoints[best_path]["method_meta"]["run_label"] = "wrong"

    with pytest.raises(finalizer.FinalizationError, match="metadata disagreement"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)


def test_fail_closed_on_non_finite_checkpoint_weight(tmp_path, monkeypatch):
    batch_dir, _, checkpoints, kwargs = make_complete_batch(tmp_path, monkeypatch)
    manifest = json.loads((batch_dir / "manifest.json").read_text())
    first_job = manifest["jobs"][0]
    best_path = Path(first_job["run_dir"]) / "checkpoint_best_iou.pkl"
    checkpoints[best_path]["net"]["weight"] = torch.tensor([float("nan")])

    with pytest.raises(finalizer.FinalizationError, match="non-finite values"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)


def test_atomic_pair_rolls_back_if_second_install_fails(tmp_path, monkeypatch):
    batch_dir, _, _, kwargs = make_complete_batch(tmp_path, monkeypatch)
    real_replace = finalizer.os.replace

    def fail_markdown_install(source, destination):
        if Path(destination).name == finalizer.OUTPUT_MARKDOWN:
            raise OSError("injected markdown install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(finalizer.os, "replace", fail_markdown_install)
    with pytest.raises(OSError, match="injected markdown"):
        finalizer.finalize_batch(batch_dir, **kwargs)
    assert_no_reports(batch_dir)


def test_real_frozen_source_and_state_schema_are_valid():
    manifest = {
        "provenance": {
            "repository_head": finalizer.EXPECTED_LAUNCH_REPOSITORY_HEAD,
            **finalizer.EXPECTED_SOURCE_SHA256,
        }
    }
    evidence = finalizer.validate_workspace_provenance(manifest, finalizer.PROJECT_DIR)
    assert evidence["canonical_source_commit"] == finalizer.EXPECTED_CANONICAL_SOURCE_COMMIT
    schema = finalizer.canonical_state_schema()
    assert len(schema) == 340
    assert schema["final.weight"] == ((1, 4, 3, 3), "torch.float32")
