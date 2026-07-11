#!/usr/bin/env python3
"""Run RODS and matched deep-supervision controls on fixed holdout splits.

The runner mirrors ``run_clean_baselines.py``: it uses official training
manifests for fitting, deterministic internal holdouts for model selection,
and only opens official test manifests for split provenance and leakage
audits performed by ``main.py``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from tools.run_clean_baselines import (  # noqa: E402
    DATASET_NAMES,
    parse_csv,
    validate_dataset,
    write_json,
)


METHODS = (
    "legacy_exact",
    "legacy_rescaled",
    "final_only",
    "side_no_location",
    "rods_interval",
    "rods_hard",
    "rods_random",
    "rods_area_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Schedule RODS deep-supervision controls on fixed GPUs."
    )
    parser.add_argument("--datasets", default=",".join(DATASET_NAMES))
    parser.add_argument("--seeds", default="20260711,20260712,20260713")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument("--deterministic", choices=("true", "false"), default="true")
    parser.add_argument("--aux-loss-weight", type=float, default=0.8)
    parser.add_argument("--ownership-preferred-cells", type=float, default=3.0)
    parser.add_argument("--ownership-sigma", type=float, default=0.75)
    parser.add_argument("--ownership-min-decidability", type=float, default=0.25)
    parser.add_argument("--ownership-interval-ratio", type=float, default=0.5)
    parser.add_argument("--ownership-fallback", choices=("side0", "final_only"), default="side0")
    parser.add_argument("--ownership-ignore-dilation", type=int, default=3)
    parser.add_argument("--empty-side-policy", choices=("skip", "background_only"), default="skip")
    parser.add_argument("--rods-log-interval", type=int, default=50)
    parser.add_argument("--batch-id", default="rods_controls_v1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[str], list[int], list[str], list[int]]:
    datasets = parse_csv(args.datasets, str)
    seeds = parse_csv(args.seeds, int)
    methods = parse_csv(args.methods, str)
    gpus = parse_csv(args.gpus, int)
    unknown_datasets = sorted(set(datasets).difference(DATASET_NAMES))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {unknown_datasets}; allowed={DATASET_NAMES}")
    unknown_methods = sorted(set(methods).difference(METHODS))
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}; allowed={METHODS}")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.aux_loss_weight < 0:
        raise ValueError("--aux-loss-weight must be non-negative")
    if args.ownership_sigma <= 0:
        raise ValueError("--ownership-sigma must be positive")
    if args.ownership_preferred_cells <= 0:
        raise ValueError("--ownership-preferred-cells must be positive")
    if args.ownership_ignore_dilation <= 0 or args.ownership_ignore_dilation % 2 == 0:
        raise ValueError("--ownership-ignore-dilation must be a positive odd integer")
    if args.rods_log_interval < 0:
        raise ValueError("--rods-log-interval must be non-negative")
    return datasets, seeds, methods, gpus


def build_command(args: argparse.Namespace, job: dict) -> list[str]:
    run_dir = Path(job["run_dir"])
    command = [
        sys.executable,
        str(PROJECT_DIR / "main.py"),
        "--mode", "train",
        "--model-type", "mshnet",
        "--deep-supervision", job["method"],
        "--dataset-dir", job["dataset_dir"],
        "--train-split-file", job["train_file"],
        "--test-split-file", job["test_file"],
        "--val-fraction", str(args.val_fraction),
        "--split-seed", str(args.split_seed),
        "--seed", str(job["seed"]),
        "--deterministic", args.deterministic,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--warm-epoch", str(args.warm_epoch),
        "--aux-loss-weight", str(args.aux_loss_weight),
        "--ownership-preferred-cells", str(args.ownership_preferred_cells),
        "--ownership-sigma", str(args.ownership_sigma),
        "--ownership-min-decidability", str(args.ownership_min_decidability),
        "--ownership-interval-ratio", str(args.ownership_interval_ratio),
        "--ownership-fallback", args.ownership_fallback,
        "--ownership-ignore-dilation", str(args.ownership_ignore_dilation),
        "--empty-side-policy", args.empty_side_policy,
        "--rods-log-interval", str(args.rods_log_interval),
        "--run-dir", str(run_dir),
        "--run-label", job["job_id"],
    ]
    checkpoint = run_dir / "checkpoint.pkl"
    if args.resume and checkpoint.is_file():
        command.extend(["--if-checkpoint", "true", "--checkpoint-dir", str(run_dir)])
    return command


def main() -> int:
    args = parse_args()
    datasets, seeds, methods, gpus = validate_args(args)
    dataset_meta = {
        name: validate_dataset(name, args.split_seed, args.val_fraction)
        for name in datasets
    }

    report_root = PROJECT_DIR / "repro_runs" / "rods" / args.batch_id
    weight_root = PROJECT_DIR / "weight" / "rods" / args.batch_id
    report_root.mkdir(parents=True, exist_ok=True)
    weight_root.mkdir(parents=True, exist_ok=True)

    jobs = []
    for method in methods:
        for seed in seeds:
            for dataset_name in datasets:
                job_id = f"{method}__{dataset_name.lower()}__seed_{seed}"
                job = {
                    "job_id": job_id,
                    "method": method,
                    "dataset": dataset_name,
                    "seed": seed,
                    "dataset_dir": dataset_meta[dataset_name]["dataset_dir"],
                    "train_file": dataset_meta[dataset_name]["train_file"],
                    "test_file": dataset_meta[dataset_name]["test_file"],
                    "run_dir": str(weight_root / method / dataset_name / f"seed_{seed}"),
                    "log_file": str(report_root / "logs" / f"{job_id}.log"),
                    "result_file": str(report_root / "jobs" / f"{job_id}.json"),
                }
                jobs.append(job)

    manifest = {
        "batch_id": args.batch_id,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "stage": "development_holdout_rods_controls",
        "official_test_policy": "loaded only for disjoint/hash audit; not iterated",
        "args": vars(args),
        "datasets": dataset_meta,
        "methods": list(methods),
        "jobs": jobs,
    }
    manifest_path = report_root / "manifest.json"
    if manifest_path.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(
            f"batch already exists: {manifest_path}; pass --resume or use a new --batch-id"
        )
    write_json(manifest_path, manifest)

    pending = []
    for job in jobs:
        result_path = Path(job["result_file"])
        if args.resume and result_path.is_file():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("returncode") == 0:
                print(f"skip completed {job['job_id']}", flush=True)
                continue
        job["command"] = build_command(args, job)
        pending.append(job)

    if args.dry_run:
        for index, job in enumerate(pending):
            gpu = gpus[index % len(gpus)]
            print(f"GPU {gpu}: " + " ".join(job["command"]))
        return 0

    active: dict[int, dict] = {}
    failures = []
    while pending or active:
        for gpu in gpus:
            if gpu in active or not pending:
                continue
            job = pending.pop(0)
            log_path = Path(job["log_file"])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            started_at = dt.datetime.now(dt.timezone.utc).isoformat()
            process = subprocess.Popen(
                job["command"],
                cwd=PROJECT_DIR,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] = {
                "job": job,
                "process": process,
                "log_handle": log_handle,
                "started_at": started_at,
                "started_monotonic": time.monotonic(),
            }
            print(f"start gpu={gpu} pid={process.pid} job={job['job_id']}", flush=True)

        time.sleep(2.0)
        for gpu, state in list(active.items()):
            process = state["process"]
            returncode = process.poll()
            if returncode is None:
                continue
            state["log_handle"].close()
            job = state["job"]
            payload = {
                "job_id": job["job_id"],
                "method": job["method"],
                "dataset": job["dataset"],
                "seed": job["seed"],
                "gpu": gpu,
                "pid": process.pid,
                "started_at": state["started_at"],
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - state["started_monotonic"],
                "returncode": returncode,
                "command": job["command"],
                "log_file": job["log_file"],
                "run_dir": job["run_dir"],
            }
            write_json(Path(job["result_file"]), payload)
            print(
                f"finish gpu={gpu} rc={returncode} job={job['job_id']} "
                f"elapsed={payload['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if returncode != 0:
                failures.append(job["job_id"])
            del active[gpu]

    if failures:
        print("failed jobs: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"all {len(jobs)} RODS/control jobs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
