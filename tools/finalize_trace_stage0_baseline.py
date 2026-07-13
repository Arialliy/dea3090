#!/usr/bin/env python3
"""Fail-closed finalizer for the single-dataset TRACE Stage-0 baseline.

This finalizer is intentionally narrower than ``finalize_clean_baselines.py``:
it accepts only the frozen NUAA-SIRST, three-seed batch used to establish the
canonical deterministic MSHNet baseline for TRACE.  It reads split *manifests*
for provenance, but never opens dataset images or masks and never evaluates a
model on the official test set.

All evidence is validated before either report is installed.  The JSON and
Markdown reports are then installed as one rollback-protected atomic pair.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any, Callable
import uuid

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.finalize_clean_baselines import (  # noqa: E402
    FinalizationError,
    expected_evaluation_epochs,
    load_checkpoint_cpu,
    normalized_path,
    parse_metrics,
    read_json,
    require_exact_int,
    require_mapping,
    require_number,
)


SCHEMA_VERSION = 1
EXPECTED_BATCH_ID = "trace_stage0_canonical_mshnet_nuaa_holdout_v1"
DATASET_NAME = "NUAA-SIRST"
EXPECTED_SEEDS = (20260711, 20260712, 20260713)
EXPECTED_EPOCHS = 400
EXPECTED_EVALUATION_INTERVAL = 10
EXPECTED_EVALUATION_EPOCHS = tuple(
    expected_evaluation_epochs(EXPECTED_EPOCHS, EXPECTED_EVALUATION_INTERVAL)
)
EXPECTED_STAGE = "development_holdout_baseline"
EXPECTED_TEST_POLICY = "loaded only for disjoint/hash audit; not iterated"
EXPECTED_CANONICAL_SOURCE_COMMIT = "46cdfd46802629da51f70124662af7335be74b56"
EXPECTED_LAUNCH_REPOSITORY_HEAD = "43c8c8367c21b64cae9e719868aaccda5cc6d329"
EXPECTED_HISTORICAL_SOURCE_SHA256 = (
    "2cb87bbc2c8cd6d7053df9ffb4c0ea7f01acf65c4d1750dd93ac639a94e44c0e"
)
EXPECTED_SOURCE_SHA256 = {
    "canonical_official_sha256": (
        "adf892d828f8795eb5987849a42e12870158fc8e801467dd41cb4c3dcf50769f"
    ),
    "canonical_deterministic_sha256": (
        "5f0c7425ede2d9dbac386ecfe6b7705e8c5143b31c589511e4c0389e3fe4d5a6"
    ),
    "training_entrypoint_sha256": (
        "611f7b2291637b2bb3af71d418ccf5b60b4818b7d609f6cda81721fde5382d7a"
    ),
}
EXPECTED_CANONICAL_PROTOCOL = {
    "model_type": "mshnet",
    "mshnet_variant": "deterministic",
    "evaluation_protocol": "internal_holdout",
    "deep_supervision": "legacy_exact",
    "fusion_regularizer": "none",
    "deterministic": True,
    "evaluation_interval": EXPECTED_EVALUATION_INTERVAL,
    "skip_final_evaluation": False,
    "checkpoint_resume": False,
}
EXPECTED_BATCH_ARGS = {
    "batch_id": EXPECTED_BATCH_ID,
    "batch_size": 4,
    "datasets": DATASET_NAME,
    "dry_run": False,
    "epochs": EXPECTED_EPOCHS,
    "gpus": "0,1,2",
    "lr": 0.05,
    "num_workers": 0,
    "resume": False,
    "seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS),
    "split_seed": 20260711,
    "val_fraction": 0.2,
    "warm_epoch": 5,
}
EXPECTED_DATASET_COUNTS = {
    "official_train_count": 213,
    "fit_count": 170,
    "val_count": 43,
    "official_test_count": 214,
}
EXPECTED_DATASET_HASHES = {
    "official_train_sha256": (
        "815dcca749f087f27f5dad4b447015aee70bd7ae7779d6fdd7d6efa6d5c6943f"
    ),
    "fit_sha256": (
        "2bc2eaae4b456dbcaf3eaa99aa5079287d41143449f7d17272a58a9ae96b88d6"
    ),
    "val_sha256": (
        "ffea874316e41558411d424b2fda531f14824dd68195bcfc351dd84079e89534"
    ),
    "official_test_sha256": (
        "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5"
    ),
}
EXPECTED_CANONICAL_PARAMETER_COUNT = 4_065_513
EXPECTED_RUNTIME_ATTESTATION_SHA256 = (
    "eae6f4af093893dffba4fdd5295dbb115fd93e80e9a5e6b951791ed4ecdca726"
)
EXPECTED_RUNTIME_SOURCE_AGGREGATE_SHA256 = (
    "cfb088c3021cdeb02959f9377a9cfca99724c7c1873d9e087407d4ed8b5c9989"
)
EXPECTED_RUNTIME_DATA_AGGREGATES = {
    "fit": "15473d2f812451aec6753cf2717d8b2750af7f7b442b1cca4f41321f44dc28c2",
    "validation": "651b7dc6eaff3d0af7d42a9baef90e38fe00cbf93c306fae165e3eed998f90eb",
    "official_train": "dd9f2fcd7631703b2675cb866be79135e343d8ecba8527c0e5b019d49055a21d",
}
OUTPUT_JSON = "trace_stage0_canonical_baseline_summary.json"
OUTPUT_MARKDOWN = "trace_stage0_canonical_baseline_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and finalize the frozen 1-dataset x 3-seed TRACE Stage-0 "
            "canonical deterministic MSHNet holdout batch."
        )
    )
    parser.add_argument("--batch-id", default=EXPECTED_BATCH_ID)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing report pair only after validation succeeds.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FinalizationError(f"missing file for SHA-256 audit: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise FinalizationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def split_hash(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def read_split(path: Path, label: str) -> list[str]:
    if not path.is_file():
        raise FinalizationError(f"missing {label}: {path}")
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise FinalizationError(f"cannot read {label} {path}: {exc}") from exc
    if not lines or any(not name.strip() or name != name.strip() for name in lines):
        raise FinalizationError(f"{label} must contain non-empty, stripped sample ids")
    if len(lines) != len(set(lines)):
        raise FinalizationError(f"{label} contains duplicate sample ids")
    return lines


def parse_utc_timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str) or not value:
        raise FinalizationError(f"{label} must be a non-empty ISO timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FinalizationError(f"{label} is not a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FinalizationError(f"{label} must include a UTC offset")
    return parsed.astimezone(dt.timezone.utc)


def _git_output(project_dir: Path, args: list[str], label: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=project_dir, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "output", b"")
        rendered = detail.decode("utf-8", errors="replace").strip()
        raise FinalizationError(f"cannot validate {label}: {rendered or exc}") from exc


def validate_workspace_provenance(
    manifest: dict[str, Any], project_dir: Path
) -> dict[str, Any]:
    """Validate both the historical source object and isolated local sources."""

    provenance = require_mapping(manifest.get("provenance"), "manifest.provenance")
    expected_provenance = {
        "repository_head": EXPECTED_LAUNCH_REPOSITORY_HEAD,
        **EXPECTED_SOURCE_SHA256,
    }
    if provenance != expected_provenance:
        raise FinalizationError(
            "manifest.provenance does not exactly match the frozen Stage-0 launch"
        )

    resolved_commit = _git_output(
        project_dir,
        ["rev-parse", f"{EXPECTED_CANONICAL_SOURCE_COMMIT}^{{commit}}"],
        "canonical source commit",
    ).decode("ascii", errors="strict").strip()
    if resolved_commit != EXPECTED_CANONICAL_SOURCE_COMMIT:
        raise FinalizationError("canonical source commit does not resolve exactly")
    historical_source = _git_output(
        project_dir,
        ["show", f"{EXPECTED_CANONICAL_SOURCE_COMMIT}:model/MSHNet.py"],
        "historical model/MSHNet.py",
    )
    historical_hash = hashlib.sha256(historical_source).hexdigest()
    if historical_hash != EXPECTED_HISTORICAL_SOURCE_SHA256:
        raise FinalizationError("historical canonical MSHNet source hash mismatch")

    local_sources = {
        "canonical_official_sha256": project_dir
        / "model"
        / "baselines"
        / "mshnet_official.py",
        "canonical_deterministic_sha256": project_dir
        / "model"
        / "baselines"
        / "mshnet_deterministic.py",
        "training_entrypoint_sha256": project_dir / "main.py",
    }
    actual_hashes = {key: sha256_file(path) for key, path in local_sources.items()}
    if actual_hashes != EXPECTED_SOURCE_SHA256:
        raise FinalizationError("current isolated canonical source hash mismatch")
    return {
        "canonical_source_commit": EXPECTED_CANONICAL_SOURCE_COMMIT,
        "canonical_source_sha256": historical_hash,
        "launch_repository_head": EXPECTED_LAUNCH_REPOSITORY_HEAD,
        **actual_hashes,
    }


def validate_runtime_attestation(batch_dir: Path) -> dict[str, Any]:
    """Validate the capture-time process/source/environment/data attestation."""

    path = batch_dir / "runtime_attestation.json"
    actual_sha256 = sha256_file(path)
    if actual_sha256 != EXPECTED_RUNTIME_ATTESTATION_SHA256:
        raise FinalizationError("runtime attestation artifact hash mismatch")
    payload = require_mapping(
        read_json(path, "runtime attestation"), "runtime attestation"
    )
    if payload.get("schema") != "trace-stage0-runtime-attestation/v1":
        raise FinalizationError("runtime attestation schema mismatch")
    if payload.get("status") != "PASS" or payload.get("batch_id") != EXPECTED_BATCH_ID:
        raise FinalizationError("runtime attestation did not pass for the frozen batch")

    capture = require_mapping(payload.get("capture"), "runtime attestation.capture")
    if (
        capture.get("capture_during_run") is not True
        or capture.get("launch_time_attestation") is not False
        or capture.get("all_three_workers_alive_at_start_and_end") is not True
    ):
        raise FinalizationError("runtime attestation scope/liveness mismatch")
    capture_started = parse_utc_timestamp(
        capture.get("started_at_utc"), "runtime attestation capture start"
    )
    capture_finished = parse_utc_timestamp(
        capture.get("finished_at_utc"), "runtime attestation capture finish"
    )
    if capture_finished < capture_started:
        raise FinalizationError("runtime attestation timestamps are reversed")

    repository = require_mapping(
        payload.get("repository"), "runtime attestation.repository"
    )
    if (
        repository.get("head") != EXPECTED_LAUNCH_REPOSITORY_HEAD
        or repository.get("canonical_source_commit")
        != EXPECTED_CANONICAL_SOURCE_COMMIT
    ):
        raise FinalizationError("runtime attestation repository mismatch")
    dependencies = require_mapping(
        payload.get("source_dependencies"),
        "runtime attestation.source_dependencies",
    )
    if (
        dependencies.get("file_count") != 34
        or dependencies.get("aggregate_sha256")
        != EXPECTED_RUNTIME_SOURCE_AGGREGATE_SHA256
    ):
        raise FinalizationError("runtime source dependency closure mismatch")

    dataset = require_mapping(
        payload.get("dataset_files"), "runtime attestation.dataset_files"
    )
    fit = require_mapping(dataset.get("fit"), "runtime attestation fit files")
    validation = require_mapping(
        dataset.get("validation"), "runtime attestation validation files"
    )
    observed_data = {
        "fit": fit.get("file_content_aggregate_sha256"),
        "validation": validation.get("file_content_aggregate_sha256"),
        "official_train": dataset.get(
            "official_train_file_content_aggregate_sha256"
        ),
    }
    if observed_data != EXPECTED_RUNTIME_DATA_AGGREGATES:
        raise FinalizationError("runtime training-data byte aggregates mismatch")
    official_test = require_mapping(
        dataset.get("official_test"), "runtime attestation official test"
    )
    test_access = require_mapping(
        payload.get("official_test_pixel_access"),
        "runtime attestation official-test pixel access",
    )
    if (
        official_test.get("image_or_mask_files_opened") != 0
        or test_access.get("status") != "NOT_PERFORMED"
        or test_access.get("opened_files") != 0
    ):
        raise FinalizationError("baseline runtime attestation accessed test pixels")

    processes = require_mapping(
        payload.get("processes"), "runtime attestation.processes"
    )
    workers_value = processes.get("workers")
    if not isinstance(workers_value, list) or len(workers_value) != 3:
        raise FinalizationError("runtime attestation must contain three workers")
    workers: dict[int, dict[str, Any]] = {}
    for index, value in enumerate(workers_value):
        worker = require_mapping(value, f"runtime attestation worker {index}")
        seed = require_exact_int(worker.get("seed"), f"attested worker {index}.seed")
        pid = require_exact_int(worker.get("pid"), f"attested worker {index}.pid")
        if (
            seed not in EXPECTED_SEEDS
            or seed in workers
            or pid <= 0
            or worker.get("command_matches_manifest") is not True
            or worker.get("nvidia_gpu_uuid_matches") is not True
        ):
            raise FinalizationError("runtime worker identity/evidence mismatch")
        cmdline = worker.get("cmdline")
        if not isinstance(cmdline, list) or not cmdline or not isinstance(cmdline[0], str):
            raise FinalizationError("runtime worker command line is missing")
        workers[seed] = {
            "pid": pid,
            "argv0": cmdline[0],
            "exe": worker.get("exe"),
        }
    if tuple(sorted(workers)) != EXPECTED_SEEDS:
        raise FinalizationError("runtime attestation seed set mismatch")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "capture_started_at": capture_started.isoformat(),
        "capture_finished_at": capture_finished.isoformat(),
        "capture_scope": "during-run, not launch-time",
        "source_dependency_file_count": 34,
        "source_dependency_aggregate_sha256": (
            EXPECTED_RUNTIME_SOURCE_AGGREGATE_SHA256
        ),
        "training_data_aggregates": dict(EXPECTED_RUNTIME_DATA_AGGREGATES),
        "official_test_pixel_files_opened_by_baseline_attester": 0,
        "workers": workers,
    }


def deterministic_split(
    official_train: list[str], split_seed: int, val_fraction: float
) -> tuple[list[str], list[str]]:
    ranked = sorted(
        official_train,
        key=lambda name: hashlib.sha256(
            f"{split_seed}\0{name}".encode("utf-8")
        ).digest(),
    )
    num_val = max(
        1,
        min(
            len(official_train) - 1,
            int(round(len(official_train) * val_fraction)),
        ),
    )
    val_set = set(ranked[:num_val])
    fit = [name for name in official_train if name not in val_set]
    val = [name for name in official_train if name in val_set]
    return fit, val


def validate_dataset(
    dataset_meta: dict[str, Any], project_dir: Path
) -> dict[str, Any]:
    dataset_dir = (project_dir / "datasets" / DATASET_NAME).resolve()
    expected_meta = {
        "dataset": DATASET_NAME,
        "dataset_dir": str(dataset_dir),
        "train_file": f"img_idx/train_{DATASET_NAME}.txt",
        "test_file": f"img_idx/test_{DATASET_NAME}.txt",
        **EXPECTED_DATASET_COUNTS,
        **EXPECTED_DATASET_HASHES,
    }
    if dataset_meta != expected_meta:
        raise FinalizationError(
            "manifest NUAA-SIRST counts, paths, or split hashes are not exactly frozen"
        )

    train_path = dataset_dir / expected_meta["train_file"]
    test_path = dataset_dir / expected_meta["test_file"]
    official_train = read_split(train_path, "official train split")
    official_test = read_split(test_path, "official test split (audit only)")
    if set(official_train).intersection(official_test):
        raise FinalizationError("official NUAA-SIRST train/test manifests overlap")
    fit, val = deterministic_split(
        official_train,
        split_seed=EXPECTED_BATCH_ARGS["split_seed"],
        val_fraction=EXPECTED_BATCH_ARGS["val_fraction"],
    )
    actual_counts = {
        "official_train_count": len(official_train),
        "fit_count": len(fit),
        "val_count": len(val),
        "official_test_count": len(official_test),
    }
    actual_hashes = {
        "official_train_sha256": split_hash(official_train),
        "fit_sha256": split_hash(fit),
        "val_sha256": split_hash(val),
        "official_test_sha256": split_hash(official_test),
    }
    if actual_counts != EXPECTED_DATASET_COUNTS:
        raise FinalizationError("current NUAA-SIRST split counts differ from frozen counts")
    if actual_hashes != EXPECTED_DATASET_HASHES:
        raise FinalizationError("current NUAA-SIRST split hashes differ from frozen hashes")
    return {
        "dataset_dir": dataset_dir,
        "official_train": official_train,
        "official_test": official_test,
        "fit": fit,
        "val": val,
        "counts": actual_counts,
        "hashes": actual_hashes,
    }


def validate_manifest(
    manifest: dict[str, Any], batch_dir: Path, project_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_top_keys = {
        "args",
        "batch_id",
        "canonical_protocol",
        "canonical_source_commit",
        "created_at",
        "datasets",
        "jobs",
        "official_test_policy",
        "provenance",
        "stage",
    }
    if set(manifest) != expected_top_keys:
        raise FinalizationError("manifest has an unexpected or incomplete top-level shape")
    if batch_dir.name != EXPECTED_BATCH_ID or manifest.get("batch_id") != EXPECTED_BATCH_ID:
        raise FinalizationError(f"only frozen batch {EXPECTED_BATCH_ID!r} is accepted")
    if manifest.get("stage") != EXPECTED_STAGE:
        raise FinalizationError(f"manifest.stage must be {EXPECTED_STAGE!r}")
    if manifest.get("official_test_policy") != EXPECTED_TEST_POLICY:
        raise FinalizationError("manifest official-test policy is not the frozen audit-only policy")
    if manifest.get("canonical_source_commit") != EXPECTED_CANONICAL_SOURCE_COMMIT:
        raise FinalizationError("manifest canonical_source_commit mismatch")
    if manifest.get("canonical_protocol") != EXPECTED_CANONICAL_PROTOCOL:
        raise FinalizationError("manifest canonical_protocol mismatch")
    if manifest.get("args") != EXPECTED_BATCH_ARGS:
        raise FinalizationError("manifest args do not exactly match the frozen Stage-0 recipe")
    created_at = parse_utc_timestamp(manifest.get("created_at"), "manifest.created_at")

    datasets = require_mapping(manifest.get("datasets"), "manifest.datasets")
    if set(datasets) != {DATASET_NAME}:
        raise FinalizationError(f"manifest must contain only {DATASET_NAME}")
    dataset_evidence = validate_dataset(
        require_mapping(datasets[DATASET_NAME], f"manifest.datasets.{DATASET_NAME}"),
        project_dir,
    )

    jobs_value = manifest.get("jobs")
    if not isinstance(jobs_value, list) or len(jobs_value) != len(EXPECTED_SEEDS):
        raise FinalizationError("manifest must contain exactly three NUAA-SIRST seed jobs")
    expected_job_keys = {
        "dataset",
        "dataset_dir",
        "job_id",
        "log_file",
        "result_file",
        "run_dir",
        "seed",
        "test_file",
        "train_file",
    }
    jobs: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for index, value in enumerate(jobs_value):
        job = require_mapping(value, f"manifest.jobs[{index}]")
        if set(job) != expected_job_keys:
            raise FinalizationError(f"manifest.jobs[{index}] has an invalid shape")
        seed = require_exact_int(job.get("seed"), f"manifest.jobs[{index}].seed")
        if seed not in EXPECTED_SEEDS or seed in seen_seeds:
            raise FinalizationError("manifest jobs do not contain the exact three frozen seeds")
        job_id = f"mshnet__{DATASET_NAME.lower()}__seed_{seed}"
        expected_job = {
            "dataset": DATASET_NAME,
            "dataset_dir": str(dataset_evidence["dataset_dir"]),
            "job_id": job_id,
            "log_file": str((batch_dir / "logs" / f"{job_id}.log").resolve()),
            "result_file": str((batch_dir / "jobs" / f"{job_id}.json").resolve()),
            "run_dir": str(
                (
                    project_dir
                    / "weight"
                    / "clean"
                    / EXPECTED_BATCH_ID
                    / DATASET_NAME
                    / f"seed_{seed}"
                ).resolve()
            ),
            "seed": seed,
            "test_file": f"img_idx/test_{DATASET_NAME}.txt",
            "train_file": f"img_idx/train_{DATASET_NAME}.txt",
        }
        if job != expected_job:
            raise FinalizationError(f"manifest job paths/identity mismatch for seed {seed}")
        seen_seeds.add(seed)
        jobs.append(job)
    if tuple(sorted(seen_seeds)) != EXPECTED_SEEDS:
        raise FinalizationError("manifest seed set is not exactly frozen")
    jobs.sort(key=lambda item: EXPECTED_SEEDS.index(int(item["seed"])))
    return {"created_at": created_at, **dataset_evidence}, jobs


def expected_command(
    job: dict[str, Any],
    project_dir: Path,
    python_executable: str | None = None,
) -> list[str]:
    return [
        python_executable or sys.executable,
        str((project_dir / "main.py").resolve()),
        "--mode",
        "train",
        "--model-type",
        "mshnet",
        "--mshnet-variant",
        "deterministic",
        "--evaluation-protocol",
        "internal_holdout",
        "--deep-supervision",
        "legacy_exact",
        "--fusion-regularizer",
        "none",
        "--evaluation-interval",
        str(EXPECTED_EVALUATION_INTERVAL),
        "--skip-final-evaluation",
        "false",
        "--dataset-dir",
        job["dataset_dir"],
        "--train-split-file",
        job["train_file"],
        "--test-split-file",
        job["test_file"],
        "--val-fraction",
        str(EXPECTED_BATCH_ARGS["val_fraction"]),
        "--split-seed",
        str(EXPECTED_BATCH_ARGS["split_seed"]),
        "--seed",
        str(job["seed"]),
        "--deterministic",
        "true",
        "--epochs",
        str(EXPECTED_EPOCHS),
        "--batch-size",
        str(EXPECTED_BATCH_ARGS["batch_size"]),
        "--num-workers",
        str(EXPECTED_BATCH_ARGS["num_workers"]),
        "--lr",
        str(EXPECTED_BATCH_ARGS["lr"]),
        "--warm-epoch",
        str(EXPECTED_BATCH_ARGS["warm_epoch"]),
        "--run-dir",
        job["run_dir"],
        "--run-label",
        job["job_id"],
    ]


def validate_job_result(
    result: dict[str, Any],
    job: dict[str, Any],
    manifest_created_at: dt.datetime,
    project_dir: Path,
    python_executable: str,
) -> dict[str, Any]:
    expected_keys = {
        "command",
        "elapsed_seconds",
        "finished_at",
        "gpu",
        "job_id",
        "log_file",
        "pid",
        "returncode",
        "run_dir",
        "started_at",
    }
    if set(result) != expected_keys:
        raise FinalizationError(f"job result {job['job_id']} has an invalid shape")
    if result.get("job_id") != job["job_id"]:
        raise FinalizationError(f"job result identity mismatch for {job['job_id']}")
    if require_exact_int(result.get("returncode"), "result.returncode") != 0:
        raise FinalizationError(f"job {job['job_id']} did not finish successfully")
    expected_gpu = EXPECTED_SEEDS.index(int(job["seed"]))
    if require_exact_int(result.get("gpu"), "result.gpu") != expected_gpu:
        raise FinalizationError(f"job {job['job_id']} ran on an unexpected GPU slot")
    if require_exact_int(result.get("pid"), "result.pid") <= 0:
        raise FinalizationError(f"job {job['job_id']} has an invalid pid")
    elapsed = require_number(result.get("elapsed_seconds"), "result.elapsed_seconds")
    if elapsed <= 0.0:
        raise FinalizationError(f"job {job['job_id']} has non-positive elapsed time")
    started = parse_utc_timestamp(result.get("started_at"), "result.started_at")
    finished = parse_utc_timestamp(result.get("finished_at"), "result.finished_at")
    if started < manifest_created_at or finished < started:
        raise FinalizationError(f"job {job['job_id']} timestamps are inconsistent")
    if normalized_path(result.get("run_dir"), "result.run_dir") != Path(job["run_dir"]):
        raise FinalizationError(f"job result run_dir mismatch for {job['job_id']}")
    if normalized_path(result.get("log_file"), "result.log_file") != Path(job["log_file"]):
        raise FinalizationError(f"job result log_file mismatch for {job['job_id']}")
    command = result.get("command")
    if command != expected_command(job, project_dir, python_executable):
        raise FinalizationError(
            f"job {job['job_id']} command is not the exact fresh, no-resume recipe"
        )
    log_path = Path(job["log_file"])
    if not log_path.is_file() or log_path.stat().st_size <= 0:
        raise FinalizationError(f"missing or empty scheduler log for {job['job_id']}")
    return {
        "gpu": expected_gpu,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": elapsed,
    }


def validate_run_config(
    config: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    if set(config) != {"args", "method_meta"}:
        raise FinalizationError(f"run_config shape mismatch for {job['job_id']}")
    args = require_mapping(config.get("args"), f"run_config {job['job_id']}.args")
    required_args = {
        "mode": "train",
        "model_type": "mshnet",
        "mshnet_variant": "deterministic",
        "evaluation_protocol": "internal_holdout",
        "deep_supervision": "legacy_exact",
        "fusion_regularizer": "none",
        "deterministic": True,
        "evaluation_interval": EXPECTED_EVALUATION_INTERVAL,
        "skip_final_evaluation": False,
        "epochs": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_ARGS["batch_size"],
        "num_workers": EXPECTED_BATCH_ARGS["num_workers"],
        "lr": EXPECTED_BATCH_ARGS["lr"],
        "warm_epoch": EXPECTED_BATCH_ARGS["warm_epoch"],
        "val_fraction": EXPECTED_BATCH_ARGS["val_fraction"],
        "split_seed": EXPECTED_BATCH_ARGS["split_seed"],
        "seed": job["seed"],
        "run_label": job["job_id"],
        "run_dir": job["run_dir"],
        "dataset_dir": job["dataset_dir"],
        "train_split_file": job["train_file"],
        "val_split_file": "",
        "test_split_file": job["test_file"],
        "train_split_sha256": EXPECTED_DATASET_HASHES["fit_sha256"],
        "val_split_sha256": EXPECTED_DATASET_HASHES["val_sha256"],
        "test_split_sha256": EXPECTED_DATASET_HASHES["official_test_sha256"],
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
    mismatches = [
        f"{key}={args.get(key)!r} (expected {expected!r})"
        for key, expected in required_args.items()
        if args.get(key) != expected
    ]
    if mismatches:
        raise FinalizationError(
            f"run_config is not canonical/fresh for {job['job_id']}: "
            + "; ".join(mismatches)
        )
    metadata = require_mapping(
        config.get("method_meta"), f"run_config {job['job_id']}.method_meta"
    )
    expected_metadata = {
        "method": "MSHNet-Deterministic",
        "model_type": "mshnet",
        "mshnet_variant": "deterministic",
        "evaluation_protocol": "internal_holdout",
        "deep_supervision": "legacy_exact",
        "fusion_regularizer": "none",
        "deterministic": True,
        "evaluation_interval": EXPECTED_EVALUATION_INTERVAL,
        "skip_final_evaluation": False,
        "init_from_baseline": "",
        "dea_lambda_single": 0.0,
        "dea_lambda_dec": 0.0,
        "dea_lambda_empty": 0.0,
        "seed": job["seed"],
        "run_label": job["job_id"],
        "split_seed": EXPECTED_BATCH_ARGS["split_seed"],
        "dataset_dir": job["dataset_dir"],
        "train_split_file": job["train_file"],
        "val_split_file": "",
        "test_split_file": job["test_file"],
        "train_split_sha256": EXPECTED_DATASET_HASHES["fit_sha256"],
        "val_split_sha256": EXPECTED_DATASET_HASHES["val_sha256"],
        "test_split_sha256": EXPECTED_DATASET_HASHES["official_test_sha256"],
    }
    metadata_mismatches = [
        key for key, expected in expected_metadata.items() if metadata.get(key) != expected
    ]
    if metadata_mismatches:
        raise FinalizationError(
            f"run_config method metadata mismatch for {job['job_id']}: "
            + ", ".join(metadata_mismatches)
        )
    return metadata


def canonical_state_schema() -> dict[str, tuple[tuple[int, ...], str]]:
    try:
        from model.baselines.mshnet_deterministic import MSHNet
    except ImportError as exc:
        raise FinalizationError(f"cannot import canonical deterministic MSHNet: {exc}") from exc
    model = MSHNet(3)
    if sum(parameter.numel() for parameter in model.parameters()) != EXPECTED_CANONICAL_PARAMETER_COUNT:
        raise FinalizationError("canonical deterministic MSHNet parameter count mismatch")
    return {
        key: (tuple(value.shape), str(value.dtype))
        for key, value in model.state_dict().items()
    }


def validate_state_schema(
    checkpoint: dict[str, Any],
    expected_schema: dict[str, tuple[tuple[int, ...], str]],
    label: str,
) -> None:
    state = require_mapping(checkpoint.get("net"), f"{label}.net")
    if set(state) != set(expected_schema):
        missing = sorted(set(expected_schema).difference(state))[:5]
        unexpected = sorted(set(state).difference(expected_schema))[:5]
        raise FinalizationError(
            f"{label} is not canonical MSHNet state schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, value in state.items():
        shape = tuple(getattr(value, "shape", ()))
        dtype = str(getattr(value, "dtype", ""))
        if (shape, dtype) != expected_schema[key]:
            raise FinalizationError(f"{label} tensor schema mismatch at {key}")
        if isinstance(value, torch.Tensor) and (
            value.is_floating_point() or value.is_complex()
        ):
            if not bool(torch.isfinite(value).all()):
                raise FinalizationError(f"{label} contains non-finite values at {key}")
    require_mapping(checkpoint.get("optimizer"), f"{label}.optimizer")


def _metrics_from_checkpoint(
    checkpoint: dict[str, Any], label: str
) -> tuple[int, float, float, float, float]:
    epoch = require_exact_int(checkpoint.get("epoch"), f"{label}.epoch")
    iou = require_number(checkpoint.get("iou"), f"{label}.iou")
    pd = require_number(checkpoint.get("pd"), f"{label}.pd")
    fa = require_number(checkpoint.get("fa"), f"{label}.fa")
    best_iou = require_number(checkpoint.get("best_iou"), f"{label}.best_iou")
    if not 0.0 <= iou <= 1.0 or not 0.0 <= pd <= 1.0 or fa < 0.0:
        raise FinalizationError(f"{label} has out-of-range metrics")
    return epoch, iou, pd, fa, best_iou


def metrics_match_row(
    values: tuple[float, float, float], row: dict[str, float | int]
) -> bool:
    return all(
        f"{actual:.4f}" == f"{float(row[key]):.4f}"
        for actual, key in zip(values, ("iou", "pd", "fa"))
    )


def validate_checkpoints(
    best: dict[str, Any],
    latest: dict[str, Any],
    rows: list[dict[str, float | int]],
    improvement_rows: list[dict[str, float | int]],
    job: dict[str, Any],
    run_metadata: dict[str, Any],
    state_schema: dict[str, tuple[tuple[int, ...], str]],
) -> dict[str, float | int]:
    best_label = f"best checkpoint {job['job_id']}"
    latest_label = f"latest checkpoint {job['job_id']}"
    best_metadata = require_mapping(best.get("method_meta"), f"{best_label}.method_meta")
    latest_metadata = require_mapping(
        latest.get("method_meta"), f"{latest_label}.method_meta"
    )
    if best_metadata != run_metadata or latest_metadata != run_metadata:
        raise FinalizationError(
            f"run_config/checkpoint method metadata disagreement for {job['job_id']}"
        )
    validate_state_schema(best, state_schema, best_label)
    validate_state_schema(latest, state_schema, latest_label)

    rows_by_epoch = {int(row["epoch"]): row for row in rows}
    if len(rows_by_epoch) != len(rows):
        raise FinalizationError(f"duplicate evaluation epochs for {job['job_id']}")
    best_epoch, best_iou, best_pd, best_fa, recorded_best = _metrics_from_checkpoint(
        best, best_label
    )
    if best_epoch not in rows_by_epoch:
        raise FinalizationError(f"best checkpoint epoch is not scheduled for {job['job_id']}")
    if not math.isclose(best_iou, recorded_best, rel_tol=0.0, abs_tol=1e-12):
        raise FinalizationError(f"best checkpoint best_iou mismatch for {job['job_id']}")
    if not metrics_match_row((best_iou, best_pd, best_fa), rows_by_epoch[best_epoch]):
        raise FinalizationError(f"best checkpoint disagrees with epoch log for {job['job_id']}")
    logged_best_iou = max(float(row["iou"]) for row in rows)
    if f"{best_iou:.4f}" != f"{logged_best_iou:.4f}":
        raise FinalizationError(f"checkpoint is not logged best IoU for {job['job_id']}")

    latest_epoch, latest_iou, latest_pd, latest_fa, latest_best = _metrics_from_checkpoint(
        latest, latest_label
    )
    if latest_epoch != EXPECTED_EPOCHS - 1:
        raise FinalizationError(f"latest checkpoint does not prove 400 epochs for {job['job_id']}")
    if not metrics_match_row(
        (latest_iou, latest_pd, latest_fa), rows_by_epoch[EXPECTED_EPOCHS - 1]
    ):
        raise FinalizationError(f"latest checkpoint disagrees with final log row for {job['job_id']}")
    if not math.isclose(latest_best, best_iou, rel_tol=0.0, abs_tol=1e-12):
        raise FinalizationError(f"latest and best checkpoints disagree for {job['job_id']}")

    if not improvement_rows:
        raise FinalizationError(f"empty metric.log for {job['job_id']}")
    improvement_epochs = [int(row["epoch"]) for row in improvement_rows]
    if improvement_epochs != sorted(set(improvement_epochs)):
        raise FinalizationError(f"metric.log epochs are duplicate/out of order for {job['job_id']}")
    if any(epoch not in EXPECTED_EVALUATION_EPOCHS for epoch in improvement_epochs):
        raise FinalizationError(f"metric.log contains unscheduled epoch for {job['job_id']}")
    if improvement_epochs[-1] != best_epoch or not metrics_match_row(
        (best_iou, best_pd, best_fa), improvement_rows[-1]
    ):
        raise FinalizationError(f"metric.log final improvement disagrees for {job['job_id']}")
    return {
        "seed": job["seed"],
        "best_epoch": best_epoch,
        "iou": best_iou,
        "pd": best_pd,
        "fa": best_fa,
        "final_epoch": latest_epoch,
        "final_iou": latest_iou,
        "final_pd": latest_pd,
        "final_fa": latest_fa,
    }


def metric_statistics(runs: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    sample_std: dict[str, float] = {}
    for key in ("best_epoch", "iou", "pd", "fa"):
        values = [float(run[key]) for run in runs]
        means[key] = statistics.mean(values)
        sample_std[key] = statistics.stdev(values)
    return means, sample_std


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TRACE Stage-0 canonical MSHNet baseline",
        "",
        "> **Scope guard:** all reported values are selected on the frozen internal "
        "NUAA-SIRST validation split. Within this baseline training/finalization "
        "process, the official test split was used only for ID/hash/disjointness "
        "audit; its images and masks were not iterated and it was not used for "
        "checkpoint selection or evaluation. A separate Stage-0 task-definition "
        "audit did inspect test annotations descriptively and is not baseline evidence.",
        "",
        f"- Batch: `{summary['batch_id']}`",
        f"- Historical canonical source commit: `{summary['source_provenance']['canonical_source_commit']}`",
        "- Model: canonical deterministic MSHNet (4,065,513 parameters)",
        "- Protocol: fresh 400-epoch runs, no resume/init, validation every 10 epochs",
        "- Evaluated checkpoints per run: 40 (epochs 9, 19, …, 399)",
        "- Selection: best internal-validation IoU only",
        "- Dispersion: sample SD across the three pre-registered seeds (n−1 denominator)",
        "- Runtime attestation: during-run (not launch-time), 34 local source "
        "dependencies and all 213 official-train image/mask pairs byte-hashed",
        "",
        "## Per-seed validation checkpoints",
        "",
        "| Seed | Best epoch | IoU | PD | FA/M |",
        "|---:|---:|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['seed']} | {run['best_epoch']} | {run['iou']:.6f} | "
            f"{run['pd']:.6f} | {run['fa']:.3f} |"
        )
    mean = summary["aggregate"]["mean"]
    std = summary["aggregate"]["sample_std"]
    lines.extend(
        [
            "",
            "## Three-seed aggregate",
            "",
            "| IoU mean ± SD | PD mean ± SD | FA/M mean ± SD | Best epoch mean ± SD |",
            "|---:|---:|---:|---:|",
            f"| {mean['iou']:.6f} ± {std['iou']:.6f} | "
            f"{mean['pd']:.6f} ± {std['pd']:.6f} | "
            f"{mean['fa']:.3f} ± {std['fa']:.3f} | "
            f"{mean['best_epoch']:.2f} ± {std['best_epoch']:.2f} |",
            "",
            "## Interpretation boundary",
            "",
            "This artifact finalizes only the internal-validation Stage-0 baseline. "
            "It is not an official-test result and must not be used as a paper main-table "
            "claim. This baseline process did not evaluate official test data; separately, "
            "the Stage-0 task-definition audit inspected test masks descriptively, so the "
            "project must not claim that official-test annotations were globally sealed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_pair_atomic(
    json_path: Path,
    json_text: str,
    markdown_path: Path,
    markdown_text: str,
    *,
    force: bool,
) -> None:
    outputs = (json_path, markdown_path)
    existing = [path for path in outputs if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "refusing to overwrite existing TRACE Stage-0 report: "
            + ", ".join(str(path) for path in existing)
        )
    token = uuid.uuid4().hex
    temporary = {
        json_path: json_path.with_name(f".{json_path.name}.{token}.tmp"),
        markdown_path: markdown_path.with_name(f".{markdown_path.name}.{token}.tmp"),
    }
    backups = {
        path: path.with_name(f".{path.name}.{token}.bak") for path in existing
    }
    installed: list[Path] = []
    try:
        for path, text in ((json_path, json_text), (markdown_path, markdown_text)):
            with temporary[path].open("x", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
        for path, backup in backups.items():
            os.replace(path, backup)
        for path in outputs:
            os.replace(temporary[path], path)
            installed.append(path)
    except Exception:
        for path in installed:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for path, backup in backups.items():
            if backup.exists():
                os.replace(backup, path)
        raise
    finally:
        for path in temporary.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        for backup in backups.values():
            try:
                backup.unlink()
            except FileNotFoundError:
                pass


def finalize_batch(
    batch_dir: Path,
    *,
    force: bool = False,
    project_dir: Path = PROJECT_DIR,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint_cpu,
    workspace_validator: Callable[
        [dict[str, Any], Path], dict[str, Any]
    ] = validate_workspace_provenance,
    runtime_attestation_validator: Callable[
        [Path], dict[str, Any]
    ] = validate_runtime_attestation,
    state_schema_loader: Callable[
        [], dict[str, tuple[tuple[int, ...], str]]
    ] = canonical_state_schema,
) -> dict[str, Any]:
    project_dir = project_dir.expanduser().resolve()
    batch_dir = batch_dir.expanduser().resolve()
    json_path = batch_dir / OUTPUT_JSON
    markdown_path = batch_dir / OUTPUT_MARKDOWN
    if any(path.exists() for path in (json_path, markdown_path)) and not force:
        raise FileExistsError("refusing to overwrite an existing TRACE Stage-0 report")

    manifest_path = batch_dir / "manifest.json"
    manifest = require_mapping(read_json(manifest_path, "manifest"), "manifest")
    dataset_evidence, jobs = validate_manifest(manifest, batch_dir, project_dir)
    source_provenance = workspace_validator(manifest, project_dir)
    runtime_attestation = runtime_attestation_validator(batch_dir)
    state_schema = state_schema_loader()

    runs: list[dict[str, Any]] = []
    for job in jobs:
        result_path = Path(job["result_file"])
        result = require_mapping(
            read_json(result_path, f"job result {job['job_id']}"),
            f"job result {job['job_id']}",
        )
        attested_worker = require_mapping(
            require_mapping(
                runtime_attestation.get("workers"),
                "runtime attestation workers",
            ).get(job["seed"]),
            f"runtime attestation worker seed {job['seed']}",
        )
        attested_argv0 = attested_worker.get("argv0")
        if not isinstance(attested_argv0, str) or not attested_argv0:
            raise FinalizationError(
                f"runtime attestation lacks argv0 for {job['job_id']}"
            )
        scheduler = validate_job_result(
            result,
            job,
            dataset_evidence["created_at"],
            project_dir,
            attested_argv0,
        )
        if require_exact_int(
            attested_worker.get("pid"), f"attested pid seed {job['seed']}"
        ) != require_exact_int(result.get("pid"), f"result pid seed {job['seed']}"):
            raise FinalizationError(
                f"runtime attestation/job-result pid mismatch for {job['job_id']}"
            )
        capture_started = parse_utc_timestamp(
            runtime_attestation.get("capture_started_at"),
            "runtime attestation capture start",
        )
        capture_finished = parse_utc_timestamp(
            runtime_attestation.get("capture_finished_at"),
            "runtime attestation capture finish",
        )
        job_started = parse_utc_timestamp(
            scheduler["started_at"], f"scheduler start {job['job_id']}"
        )
        job_finished = parse_utc_timestamp(
            scheduler["finished_at"], f"scheduler finish {job['job_id']}"
        )
        if not (
            job_started <= capture_started <= capture_finished <= job_finished
        ):
            raise FinalizationError(
                f"runtime attestation was not captured during {job['job_id']}"
            )
        run_dir = Path(job["run_dir"])
        fit_names = read_split(run_dir / "split_train.txt", "persisted fit split")
        val_names = read_split(run_dir / "split_val.txt", "persisted validation split")
        if fit_names != dataset_evidence["fit"] or val_names != dataset_evidence["val"]:
            raise FinalizationError(f"persisted split identity mismatch for {job['job_id']}")

        config_path = run_dir / "run_config.json"
        run_config = require_mapping(
            read_json(config_path, f"run_config {job['job_id']}"),
            f"run_config {job['job_id']}",
        )
        run_metadata = validate_run_config(run_config, job)
        rows = parse_metrics(run_dir / "epoch_metric.log")
        if len(rows) != len(EXPECTED_EVALUATION_EPOCHS):
            raise FinalizationError(
                f"{job['job_id']} must have exactly 40 evaluation rows; found {len(rows)}"
            )
        if tuple(int(row["epoch"]) for row in rows) != EXPECTED_EVALUATION_EPOCHS:
            raise FinalizationError(
                f"{job['job_id']} evaluation epochs must be exactly 9,19,...,399"
            )
        improvement_rows = parse_metrics(run_dir / "metric.log")

        best_path = run_dir / "checkpoint_best_iou.pkl"
        latest_path = run_dir / "checkpoint.pkl"
        best_checkpoint = require_mapping(
            checkpoint_loader(best_path), f"best checkpoint {job['job_id']}"
        )
        latest_checkpoint = require_mapping(
            checkpoint_loader(latest_path), f"latest checkpoint {job['job_id']}"
        )
        run = validate_checkpoints(
            best_checkpoint,
            latest_checkpoint,
            rows,
            improvement_rows,
            job,
            run_metadata,
            state_schema,
        )
        run.update(
            {
                "scheduler": scheduler,
                "run_dir": str(run_dir),
                "artifacts": {
                    "job_result": str(result_path),
                    "job_result_sha256": sha256_file(result_path),
                    "run_config": str(config_path),
                    "run_config_sha256": sha256_file(config_path),
                    "epoch_metric_log": str(run_dir / "epoch_metric.log"),
                    "epoch_metric_log_sha256": sha256_file(
                        run_dir / "epoch_metric.log"
                    ),
                    "improvement_log": str(run_dir / "metric.log"),
                    "improvement_log_sha256": sha256_file(run_dir / "metric.log"),
                    "best_checkpoint": str(best_path),
                    "best_checkpoint_sha256": sha256_file(best_path),
                    "latest_checkpoint": str(latest_path),
                    "latest_checkpoint_sha256": sha256_file(latest_path),
                },
            }
        )
        runs.append(run)
        del best_checkpoint, latest_checkpoint

    means, sample_std = metric_statistics(runs)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "batch_id": EXPECTED_BATCH_ID,
        "validated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete_and_validated",
        "stage": "TRACE Stage-0 canonical baseline",
        "method": "MSHNet-Deterministic",
        "model_type": "mshnet",
        "mshnet_variant": "deterministic",
        "canonical_parameter_count": EXPECTED_CANONICAL_PARAMETER_COUNT,
        "source_provenance": source_provenance,
        "runtime_attestation": runtime_attestation,
        "canonical_protocol": EXPECTED_CANONICAL_PROTOCOL,
        "fresh_start": {
            "resume": False,
            "checkpoint_initialization": False,
            "commands_exactly_validated": True,
        },
        "selection": {
            "scope": "NUAA-SIRST official-training-set internal validation holdout only",
            "criterion": "best validation IoU over 40 pre-scheduled checkpoints",
            "evaluation_epochs": list(EXPECTED_EVALUATION_EPOCHS),
            "official_test_used_for_selection": False,
        },
        "official_test": {
            "scope": "this baseline training and finalization process only",
            "status": (
                "not evaluated by this baseline; split IDs read only for hash and "
                "train/test disjointness audit"
            ),
            "images_or_masks_iterated": False,
            "evaluated": False,
            "used_for_checkpoint_selection": False,
            "separate_task_definition_audit_inspected_test_masks": True,
            "split_sha256": EXPECTED_DATASET_HASHES["official_test_sha256"],
        },
        "dataset": {
            "name": DATASET_NAME,
            "counts": dataset_evidence["counts"],
            "split_hashes": dataset_evidence["hashes"],
            "fit_validation_disjoint": True,
            "official_train_test_disjoint": True,
        },
        "seeds": list(EXPECTED_SEEDS),
        "runs": runs,
        "aggregate": {
            "n": len(runs),
            "mean": means,
            "sample_std": sample_std,
            "dispersion_definition": "sample standard deviation (n-1 denominator)",
        },
        "metrics": {
            "iou": "foreground intersection over union at logit threshold 0",
            "pd": "8-connected component detection with strict centroid distance < 3 px",
            "fa": "unmatched predicted foreground pixels per million image pixels",
        },
        "scope_guard": (
            "Internal validation only. Not an official-test result and not eligible "
            "for a paper main-table claim. The baseline process did not evaluate test "
            "data; a separate task-definition audit inspected test annotations."
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    json_text = json.dumps(summary, indent=2, sort_keys=False, allow_nan=False) + "\n"
    markdown_text = build_markdown(summary)
    write_report_pair_atomic(
        json_path,
        json_text,
        markdown_path,
        markdown_text,
        force=force,
    )
    return summary


def main() -> int:
    args = parse_args()
    if args.batch_id != EXPECTED_BATCH_ID:
        raise FinalizationError(f"--batch-id must be exactly {EXPECTED_BATCH_ID!r}")
    batch_dir = PROJECT_DIR / "repro_runs" / "clean" / EXPECTED_BATCH_ID
    summary = finalize_batch(batch_dir, force=args.force)
    print(
        f"validated {DATASET_NAME} x {len(summary['seeds'])} seeds; "
        f"wrote {batch_dir / OUTPUT_JSON} and {batch_dir / OUTPUT_MARKDOWN}"
    )
    print(
        "scope: internal validation only; this baseline did not evaluate official "
        "test data (a separate task audit inspected test masks)"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FinalizationError, FileExistsError, OSError) as exc:
        print(f"TRACE Stage-0 finalization refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
