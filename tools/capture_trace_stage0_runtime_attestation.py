#!/usr/bin/env python3
"""Capture a fail-closed, *runtime* attestation for the TRACE Stage-0 run.

This utility is deliberately separate from the launcher and training entrypoint.
It observes three already-running workers, hashes their capture-time source and
training-data dependencies, and writes one JSON document atomically.  It does
not claim that the observation happened at launch time, nor that a file could
not have changed between process launch and this capture.

The official-test split is treated as identifiers-only.  No official-test
image or mask path is opened by this program.
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Iterable


EXPECTED_BATCH_ID = "trace_stage0_canonical_mshnet_nuaa_holdout_v1"
EXPECTED_REPOSITORY_HEAD = "43c8c8367c21b64cae9e719868aaccda5cc6d329"
EXPECTED_SOURCE_COMMIT = "46cdfd46802629da51f70124662af7335be74b56"
EXPECTED_SEEDS = (20260711, 20260712, 20260713)
EXPECTED_GPUS = ("0", "1", "2")
EXPECTED_DATASET = "NUAA-SIRST"
EXPECTED_SPLIT = {
    "official_train_count": 213,
    "fit_count": 170,
    "val_count": 43,
    "official_test_count": 214,
    "official_train_sha256": "815dcca749f087f27f5dad4b447015aee70bd7ae7779d6fdd7d6efa6d5c6943f",
    "fit_sha256": "2bc2eaae4b456dbcaf3eaa99aa5079287d41143449f7d17272a58a9ae96b88d6",
    "val_sha256": "ffea874316e41558411d424b2fda531f14824dd68195bcfc351dd84079e89534",
    "official_test_sha256": "395eecd6bf0ed2a59f531de9145688597632c68f9d0933359aadcb93ec1a60b5",
}
EXPECTED_PROTOCOL = {
    "model_type": "mshnet",
    "mshnet_variant": "deterministic",
    "evaluation_protocol": "internal_holdout",
    "deep_supervision": "legacy_exact",
    "fusion_regularizer": "none",
    "deterministic": True,
    "evaluation_interval": 10,
    "skip_final_evaluation": False,
    "checkpoint_resume": False,
}
EXPECTED_RUN_ARGS = {
    "split_seed": 20260711,
    "val_fraction": 0.2,
    "epochs": 400,
    "batch_size": 4,
    "num_workers": 0,
    "lr": 0.05,
    "warm_epoch": 5,
    "resume": False,
    "dry_run": False,
}
SELECTED_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "CUBLAS_WORKSPACE_CONFIG",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)


class AttestationError(RuntimeError):
    """Raised whenever an attestation invariant cannot be proved."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AttestationError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_file_record(path: Path, *, relative_to: Path | None = None) -> dict:
    """Hash a regular file and reject an in-place edit or path replacement."""

    path = Path(path)
    require(path.is_file(), f"required regular file is missing: {path}")
    path_before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        fd_before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        fd_after = os.fstat(handle.fileno())
    path_after = path.stat()
    identity_before = (
        path_before.st_dev,
        path_before.st_ino,
        path_before.st_size,
        path_before.st_mtime_ns,
    )
    identity_fd_before = (
        fd_before.st_dev,
        fd_before.st_ino,
        fd_before.st_size,
        fd_before.st_mtime_ns,
    )
    identity_fd_after = (
        fd_after.st_dev,
        fd_after.st_ino,
        fd_after.st_size,
        fd_after.st_mtime_ns,
    )
    identity_after = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    require(
        identity_before == identity_fd_before == identity_fd_after == identity_after,
        f"file changed while it was being attested: {path}",
    )
    display_path = path
    if relative_to is not None:
        try:
            display_path = path.relative_to(relative_to)
        except ValueError as error:
            raise AttestationError(f"file escapes declared root: {path}") from error
    return {
        "path": display_path.as_posix(),
        "size_bytes": path_after.st_size,
        "sha256": digest.hexdigest(),
        "is_symlink": path.is_symlink(),
        "resolved_path": str(path.resolve()),
    }


def read_unique_names(path: Path) -> tuple[list[str], dict]:
    record = stable_file_record(path, relative_to=path.parent.parent)
    text = path.read_text(encoding="utf-8")
    names = [line.strip() for line in text.splitlines() if line.strip()]
    require(names, f"empty split file: {path}")
    require(len(names) == len(set(names)), f"duplicate id in split file: {path}")
    return names, record


def split_semantic_hash(names: Iterable[str]) -> str:
    return sha256_bytes(("\n".join(names) + "\n").encode("utf-8"))


def deterministic_fit_val(
    train_names: list[str], split_seed: int, val_fraction: float
) -> tuple[list[str], list[str]]:
    require(len(train_names) >= 2, "at least two official-train ids are required")
    require(0.0 < val_fraction < 1.0, "val_fraction must be in (0, 1)")
    ranked = sorted(
        train_names,
        key=lambda name: hashlib.sha256(
            f"{split_seed}\0{name}".encode("utf-8")
        ).digest(),
    )
    num_val = max(
        1,
        min(len(train_names) - 1, int(round(len(train_names) * val_fraction))),
    )
    val_set = set(ranked[:num_val])
    return (
        [name for name in train_names if name not in val_set],
        [name for name in train_names if name in val_set],
    )


def aggregate_sample_records(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        line = (
            f"{record['sample_id']}\0"
            f"{record['image']['sha256']}\0{record['image']['size_bytes']}\0"
            f"{record['mask']['sha256']}\0{record['mask']['size_bytes']}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def hash_training_sample(dataset_dir: Path, sample_id: str) -> dict:
    """Hash one official-*train* image/mask pair.

    This helper is intentionally called only with ids derived from the
    official-train split.  The official-test branch never constructs a pixel
    path.
    """

    require("/" not in sample_id and "\\" not in sample_id, f"unsafe id: {sample_id}")
    image_path = dataset_dir / "images" / f"{sample_id}.png"
    mask_path = dataset_dir / "masks" / f"{sample_id}.png"
    return {
        "sample_id": sample_id,
        "image": stable_file_record(image_path, relative_to=dataset_dir),
        "mask": stable_file_record(mask_path, relative_to=dataset_dir),
    }


def capture_dataset_files(manifest: dict) -> tuple[dict, dict[str, list[str]]]:
    dataset_meta = manifest["datasets"][EXPECTED_DATASET]
    dataset_dir = Path(dataset_meta["dataset_dir"]).resolve()
    train_split = dataset_dir / dataset_meta["train_file"]
    test_split = dataset_dir / dataset_meta["test_file"]
    train_names, train_split_record = read_unique_names(train_split)
    # Identifier-only access: this is the only official-test file opened.
    test_names, test_split_record = read_unique_names(test_split)
    require(not set(train_names).intersection(test_names), "train/test id overlap")

    args = manifest["args"]
    fit_names, val_names = deterministic_fit_val(
        train_names, int(args["split_seed"]), float(args["val_fraction"])
    )
    observed_split = {
        "official_train_count": len(train_names),
        "fit_count": len(fit_names),
        "val_count": len(val_names),
        "official_test_count": len(test_names),
        "official_train_sha256": split_semantic_hash(train_names),
        "fit_sha256": split_semantic_hash(fit_names),
        "val_sha256": split_semantic_hash(val_names),
        "official_test_sha256": split_semantic_hash(test_names),
    }
    require(observed_split == EXPECTED_SPLIT, "actual split does not match locked split")
    for key, expected in observed_split.items():
        require(dataset_meta.get(key) == expected, f"manifest dataset mismatch: {key}")

    fit_records = [hash_training_sample(dataset_dir, name) for name in fit_names]
    val_records = [hash_training_sample(dataset_dir, name) for name in val_names]
    official_records_by_id = {
        record["sample_id"]: record for record in fit_records + val_records
    }
    require(
        set(official_records_by_id) == set(train_names),
        "fit/val files do not partition official train",
    )
    official_records = [official_records_by_id[name] for name in train_names]
    payload = {
        "dataset": EXPECTED_DATASET,
        "dataset_dir": str(dataset_dir),
        "split_algorithm": "sha256(f'{split_seed}\\0{sample_id}'), source-order output",
        "split_metadata": observed_split,
        "official_train_split_file": train_split_record,
        "fit": {
            "count": len(fit_records),
            "file_content_aggregate_sha256": aggregate_sample_records(fit_records),
            "samples": fit_records,
        },
        "validation": {
            "count": len(val_records),
            "file_content_aggregate_sha256": aggregate_sample_records(val_records),
            "samples": val_records,
        },
        "official_train_file_content_aggregate_sha256": aggregate_sample_records(
            official_records
        ),
        "official_test": {
            "policy": "identifiers-only; image and mask paths were not constructed or opened",
            "image_or_mask_files_opened": 0,
            "split_file": test_split_record,
            "count": len(test_names),
            "semantic_id_sha256": split_semantic_hash(test_names),
        },
    }
    return payload, {"fit": fit_names, "val": val_names, "test": test_names}


def parse_csv_values(text: str, cast) -> tuple:
    values = tuple(cast(item.strip()) for item in text.split(",") if item.strip())
    require(bool(values), "empty comma-separated manifest value")
    return values


def validate_manifest(project_dir: Path, batch_id: str) -> tuple[dict, dict]:
    require(batch_id == EXPECTED_BATCH_ID, f"unexpected batch id: {batch_id}")
    manifest_path = project_dir / "repro_runs" / "clean" / batch_id / "manifest.json"
    manifest_record = stable_file_record(manifest_path, relative_to=project_dir)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise AttestationError(f"cannot decode manifest: {manifest_path}") from error
    require(manifest.get("batch_id") == EXPECTED_BATCH_ID, "manifest batch mismatch")
    require(manifest.get("canonical_source_commit") == EXPECTED_SOURCE_COMMIT, "source commit mismatch")
    require(manifest.get("canonical_protocol") == EXPECTED_PROTOCOL, "protocol mismatch")
    require(manifest.get("stage") == "development_holdout_baseline", "stage mismatch")
    require(
        manifest.get("official_test_policy")
        == "loaded only for disjoint/hash audit; not iterated",
        "official-test policy mismatch",
    )
    args = manifest.get("args", {})
    require(args.get("batch_id") == EXPECTED_BATCH_ID, "manifest args batch mismatch")
    require(parse_csv_values(args.get("datasets", ""), str) == (EXPECTED_DATASET,), "dataset args mismatch")
    require(parse_csv_values(args.get("seeds", ""), int) == EXPECTED_SEEDS, "seed args mismatch")
    require(parse_csv_values(args.get("gpus", ""), str) == EXPECTED_GPUS, "GPU args mismatch")
    for key, expected in EXPECTED_RUN_ARGS.items():
        require(args.get(key) == expected, f"manifest run arg mismatch: {key}")
    require(set(manifest.get("datasets", {})) == {EXPECTED_DATASET}, "manifest dataset set mismatch")
    jobs = manifest.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 3, "manifest must contain exactly three jobs")
    require(tuple(job.get("seed") for job in jobs) == EXPECTED_SEEDS, "manifest job seed/order mismatch")
    require(len({job.get("job_id") for job in jobs}) == 3, "duplicate job id")
    require(len({job.get("run_dir") for job in jobs}) == 3, "duplicate run directory")
    for job in jobs:
        require(job.get("dataset") == EXPECTED_DATASET, "job dataset mismatch")
        require(Path(job["dataset_dir"]).resolve() == (project_dir / "datasets" / EXPECTED_DATASET).resolve(), "job dataset directory mismatch")
        require(job.get("train_file") == "img_idx/train_NUAA-SIRST.txt", "job train split mismatch")
        require(job.get("test_file") == "img_idx/test_NUAA-SIRST.txt", "job test split mismatch")
        require(job.get("job_id") == f"mshnet__nuaa-sirst__seed_{job['seed']}", "job id mismatch")
        expected_run = project_dir / "weight" / "clean" / EXPECTED_BATCH_ID / EXPECTED_DATASET / f"seed_{job['seed']}"
        require(Path(job["run_dir"]).resolve() == expected_run.resolve(), "job run directory mismatch")
    return manifest, manifest_record


def module_name_for_file(project_dir: Path, source_file: Path) -> str:
    relative = source_file.relative_to(project_dir).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def local_module_candidates(project_dir: Path, parts: tuple[str, ...]) -> list[Path]:
    if not parts:
        return []
    base = project_dir.joinpath(*parts)
    return [base.with_suffix(".py"), base / "__init__.py"]


def add_package_initializers(
    project_dir: Path, parts: tuple[str, ...], paths: set[Path], missing: set[Path]
) -> None:
    for length in range(1, len(parts)):
        package_dir = project_dir.joinpath(*parts[:length])
        if not package_dir.is_dir():
            break
        initializer = package_dir / "__init__.py"
        if initializer.is_file():
            paths.add(initializer)
        else:
            missing.add(initializer)


def resolve_local_module(
    project_dir: Path,
    parts: tuple[str, ...],
    paths: set[Path],
    missing: set[Path],
) -> Path | None:
    if not parts or not (project_dir / parts[0]).exists():
        return None
    add_package_initializers(project_dir, parts, paths, missing)
    for candidate in local_module_candidates(project_dir, parts):
        if candidate.is_file():
            paths.add(candidate)
            return candidate
    return None


def imported_module_parts(node: ast.AST, current_module: str) -> list[tuple[str, ...]]:
    modules: list[tuple[str, ...]] = []
    if isinstance(node, ast.Import):
        modules.extend(tuple(alias.name.split(".")) for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            package = current_module.split(".")[:-1]
            remove = node.level - 1
            require(remove <= len(package), f"invalid relative import in {current_module}")
            prefix = package[: len(package) - remove] if remove else package
        else:
            prefix = []
        base = tuple(prefix + (node.module.split(".") if node.module else []))
        if base:
            modules.append(base)
        # A from-import may name either an attribute or a submodule.  Trying
        # the submodule path is conservative and harmless when absent.
        for alias in node.names:
            if alias.name != "*":
                modules.append(base + tuple(alias.name.split(".")))
    return modules


def discover_local_import_closure(project_dir: Path, entrypoint: Path) -> dict:
    project_dir = project_dir.resolve()
    entrypoint = entrypoint.resolve()
    require(entrypoint.is_file(), f"missing entrypoint: {entrypoint}")
    paths: set[Path] = {entrypoint}
    missing: set[Path] = set()
    queue = [entrypoint]
    parsed: set[Path] = set()
    edges: list[dict] = []
    while queue:
        source = queue.pop(0)
        if source in parsed:
            continue
        parsed.add(source)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except (OSError, SyntaxError, UnicodeDecodeError) as error:
            raise AttestationError(f"cannot parse local dependency: {source}") from error
        current_module = module_name_for_file(project_dir, source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for parts in imported_module_parts(node, current_module):
                before = set(paths)
                resolved = resolve_local_module(project_dir, parts, paths, missing)
                if resolved is not None:
                    edges.append(
                        {
                            "from": source.relative_to(project_dir).as_posix(),
                            "to": resolved.relative_to(project_dir).as_posix(),
                            "line": node.lineno,
                        }
                    )
                for added in paths.difference(before):
                    if added not in parsed:
                        queue.append(added)
    # Explicitly expose the namespace-package fact requested by this audit.
    model_initializer = project_dir / "model" / "__init__.py"
    if not model_initializer.is_file():
        missing.add(model_initializer)
    records = [stable_file_record(path, relative_to=project_dir) for path in sorted(paths)]
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update(
            f"{record['path']}\0{record['size_bytes']}\0{record['sha256']}\n".encode("utf-8")
        )
    missing_records = [
        {"path": path.relative_to(project_dir).as_posix(), "exists": False}
        for path in sorted(missing)
        if path.is_relative_to(project_dir)
    ]
    return {
        "method": "recursive AST closure of local imports from main.py; package initializers included",
        "entrypoint": entrypoint.relative_to(project_dir).as_posix(),
        "file_count": len(records),
        "files": records,
        "missing_namespace_initializers": missing_records,
        "import_edges": sorted(edges, key=lambda item: (item["from"], item["line"], item["to"])),
        "aggregate_sha256": aggregate.hexdigest(),
    }


def git_output(project_dir: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_dir), *args],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AttestationError(f"git command failed: {' '.join(args)}") from error


def capture_repository(project_dir: Path, manifest: dict, dependencies: dict) -> dict:
    head = git_output(project_dir, ["rev-parse", "HEAD"])
    require(head == EXPECTED_REPOSITORY_HEAD, "working repository HEAD changed")
    require(manifest["provenance"].get("repository_head") == head, "manifest HEAD mismatch")
    commit_type = git_output(project_dir, ["cat-file", "-t", EXPECTED_SOURCE_COMMIT])
    require(commit_type == "commit", "canonical source commit is unavailable")

    critical = {
        "training_entrypoint_sha256": "main.py",
        "canonical_deterministic_sha256": "model/baselines/mshnet_deterministic.py",
        "canonical_official_sha256": "model/baselines/mshnet_official.py",
    }
    by_path = {record["path"]: record for record in dependencies["files"]}
    critical_checks = []
    for manifest_key, relative_path in critical.items():
        require(relative_path in by_path, f"critical dependency absent: {relative_path}")
        actual = by_path[relative_path]["sha256"]
        expected = manifest["provenance"].get(manifest_key)
        require(actual == expected, f"critical manifest hash mismatch: {relative_path}")
        critical_checks.append(
            {
                "path": relative_path,
                "manifest_key": manifest_key,
                "sha256": actual,
                "matches_manifest": True,
            }
        )
    dependency_paths = [record["path"] for record in dependencies["files"]]
    status = git_output(
        project_dir,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", *dependency_paths],
    )
    return {
        "head": head,
        "matches_locked_head": True,
        "matches_manifest_head": True,
        "canonical_source_commit": EXPECTED_SOURCE_COMMIT,
        "canonical_source_object_type": commit_type,
        "critical_manifest_hash_checks": critical_checks,
        "dependency_worktree_status_porcelain": status.splitlines() if status else [],
    }


def read_null_fields(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise AttestationError(f"cannot read process file: {path}") from error
    return [part.decode("utf-8", errors="surrogateescape") for part in raw.split(b"\0") if part]


def parse_proc_stat(text: str) -> dict:
    right = text.rfind(")")
    require(right >= 0, "malformed /proc stat")
    try:
        pid = int(text[: text.find(" ")])
        comm = text[text.find("(") + 1 : right]
        tail = text[right + 2 :].split()
        state = tail[0]
        ppid = int(tail[1])
        start_ticks = int(tail[19])
    except (ValueError, IndexError) as error:
        raise AttestationError("malformed /proc stat fields") from error
    return {
        "pid": pid,
        "comm": comm,
        "state": state,
        "ppid": ppid,
        "start_ticks_since_boot": start_ticks,
    }


def proc_boot_time(proc_root: Path) -> int:
    for line in (proc_root / "stat").read_text(encoding="utf-8").splitlines():
        if line.startswith("btime "):
            return int(line.split()[1])
    raise AttestationError("/proc/stat has no boot time")


def process_start_utc(start_ticks: int, boot_time: int) -> str:
    ticks_per_second = os.sysconf("SC_CLK_TCK")
    timestamp = boot_time + start_ticks / ticks_per_second
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()


def read_process_snapshot(proc_root: Path, pid: int, boot_time: int) -> dict:
    root = proc_root / str(pid)
    require(root.is_dir(), f"process disappeared: {pid}")
    try:
        parsed_stat = parse_proc_stat((root / "stat").read_text(encoding="utf-8"))
        cmdline = read_null_fields(root / "cmdline")
        env_values = read_null_fields(root / "environ")
        cwd = str((root / "cwd").resolve(strict=True))
        exe = str((root / "exe").resolve(strict=True))
    except OSError as error:
        raise AttestationError(f"cannot snapshot live process {pid}") from error
    require(parsed_stat["pid"] == pid, f"pid mismatch in /proc/{pid}/stat")
    require(parsed_stat["state"] not in {"Z", "X", "x"}, f"process is not live: {pid}")
    require(bool(cmdline), f"empty cmdline for process {pid}")
    environment = {}
    for item in env_values:
        if "=" in item:
            key, value = item.split("=", 1)
            if key in SELECTED_ENV_KEYS:
                environment[key] = value
    return {
        **parsed_stat,
        "cmdline": cmdline,
        "cwd": cwd,
        "exe": exe,
        "selected_environment": {
            key: {"present": key in environment, "value": environment.get(key)}
            for key in SELECTED_ENV_KEYS
        },
        "process_start_utc": process_start_utc(parsed_stat["start_ticks_since_boot"], boot_time),
    }


def iter_process_snapshots(proc_root: Path) -> list[dict]:
    boot_time = proc_boot_time(proc_root)
    snapshots = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or not entry.is_dir():
            continue
        try:
            snapshots.append(read_process_snapshot(proc_root, int(entry.name), boot_time))
        except AttestationError:
            # Processes outside the target set may race with enumeration.
            continue
    return snapshots


def parse_option_pairs(cmdline: list[str], start_index: int = 2) -> dict[str, str]:
    remainder = cmdline[start_index:]
    require(len(remainder) % 2 == 0, f"non key/value training command: {cmdline}")
    values: dict[str, str] = {}
    for index in range(0, len(remainder), 2):
        key, value = remainder[index], remainder[index + 1]
        require(key.startswith("--"), f"unexpected positional training argument: {key}")
        require(key not in values, f"duplicate training argument: {key}")
        values[key] = value
    return values


def expected_worker_options(manifest: dict, job: dict) -> dict[str, str]:
    args = manifest["args"]
    protocol = manifest["canonical_protocol"]
    return {
        "--mode": "train",
        "--model-type": protocol["model_type"],
        "--mshnet-variant": protocol["mshnet_variant"],
        "--evaluation-protocol": protocol["evaluation_protocol"],
        "--deep-supervision": protocol["deep_supervision"],
        "--fusion-regularizer": protocol["fusion_regularizer"],
        "--evaluation-interval": str(protocol["evaluation_interval"]),
        "--skip-final-evaluation": str(protocol["skip_final_evaluation"]).lower(),
        "--dataset-dir": job["dataset_dir"],
        "--train-split-file": job["train_file"],
        "--test-split-file": job["test_file"],
        "--val-fraction": str(args["val_fraction"]),
        "--split-seed": str(args["split_seed"]),
        "--seed": str(job["seed"]),
        "--deterministic": str(protocol["deterministic"]).lower(),
        "--epochs": str(args["epochs"]),
        "--batch-size": str(args["batch_size"]),
        "--num-workers": str(args["num_workers"]),
        "--lr": str(args["lr"]),
        "--warm-epoch": str(args["warm_epoch"]),
        "--run-dir": job["run_dir"],
        "--run-label": job["job_id"],
    }


def validate_selected_environment(snapshot: dict, seed: int, gpu: str) -> None:
    expected = {
        "CUDA_VISIBLE_DEVICES": gpu,
        "PYTHONHASHSEED": str(seed),
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for key, value in expected.items():
        observed = snapshot["selected_environment"][key]
        require(observed["present"], f"pid {snapshot['pid']} lacks {key}")
        require(observed["value"] == value, f"pid {snapshot['pid']} has wrong {key}")


def discover_and_validate_processes(
    project_dir: Path, proc_root: Path, manifest: dict
) -> tuple[dict, dict[int, int]]:
    snapshots = iter_process_snapshots(proc_root)
    entrypoint = str((project_dir / "main.py").resolve())
    jobs_by_run = {str(Path(job["run_dir"]).resolve()): job for job in manifest["jobs"]}
    worker_candidates = []
    for snapshot in snapshots:
        cmdline = snapshot["cmdline"]
        if len(cmdline) < 2:
            continue
        if str(Path(cmdline[1]).resolve()) != entrypoint:
            continue
        try:
            options = parse_option_pairs(cmdline)
        except AttestationError:
            continue
        run_dir = options.get("--run-dir")
        if run_dir and str(Path(run_dir).resolve()) in jobs_by_run:
            worker_candidates.append((snapshot, options))
    require(len(worker_candidates) == 3, "did not find exactly three batch workers")

    scheduler_pids = {snapshot["ppid"] for snapshot, _ in worker_candidates}
    require(len(scheduler_pids) == 1, "workers do not share one scheduler parent")
    scheduler_pid = next(iter(scheduler_pids))
    scheduler_matches = [item for item in snapshots if item["pid"] == scheduler_pid]
    require(len(scheduler_matches) == 1, "scheduler process is not alive")
    scheduler = scheduler_matches[0]
    require(
        any(value.endswith("tools/run_clean_baselines.py") for value in scheduler["cmdline"]),
        "worker parent is not the clean-baseline scheduler",
    )
    require(
        EXPECTED_BATCH_ID in scheduler["cmdline"],
        "scheduler command does not bind the expected batch",
    )
    require(Path(scheduler["cwd"]).resolve() == project_dir.resolve(), "scheduler cwd mismatch")

    worker_payload = []
    initial_identity = {}
    seen_runs = set()
    common_argv0 = None
    common_exe = None
    for snapshot, options in sorted(worker_candidates, key=lambda item: int(item[1]["--seed"])):
        run_key = str(Path(options["--run-dir"]).resolve())
        require(run_key not in seen_runs, "multiple workers for one run directory")
        seen_runs.add(run_key)
        job = jobs_by_run[run_key]
        index = manifest["jobs"].index(job)
        expected_gpu = EXPECTED_GPUS[index]
        require(options == expected_worker_options(manifest, job), f"worker command mismatch for seed {job['seed']}")
        require(Path(snapshot["cwd"]).resolve() == project_dir.resolve(), "worker cwd mismatch")
        require(Path(snapshot["cmdline"][0]).resolve() == Path(snapshot["exe"]).resolve(), "worker argv0/exe mismatch")
        if common_argv0 is None:
            common_argv0 = snapshot["cmdline"][0]
            common_exe = snapshot["exe"]
        require(snapshot["cmdline"][0] == common_argv0, "workers use different argv0 executables")
        require(snapshot["exe"] == common_exe, "workers use different resolved executables")
        validate_selected_environment(snapshot, int(job["seed"]), expected_gpu)
        snapshot["job_id"] = job["job_id"]
        snapshot["seed"] = job["seed"]
        snapshot["expected_physical_gpu"] = expected_gpu
        snapshot["command_matches_manifest"] = True
        snapshot["cwd_git_head"] = git_output(Path(snapshot["cwd"]), ["rev-parse", "HEAD"])
        require(snapshot["cwd_git_head"] == EXPECTED_REPOSITORY_HEAD, "worker cwd git HEAD mismatch")
        worker_payload.append(snapshot)
        initial_identity[snapshot["pid"]] = snapshot["start_ticks_since_boot"]
    require(seen_runs == set(jobs_by_run), "not every manifest job has one live worker")
    return {
        "scheduler": scheduler,
        "workers": worker_payload,
        "worker_count": len(worker_payload),
        "all_workers_alive_at_capture_start": True,
        "all_commands_match_manifest": True,
        "all_selected_environments_match": True,
        "all_cwds_match_locked_git_head": True,
    }, initial_identity


def assert_processes_still_alive(
    proc_root: Path, identities: dict[int, int]
) -> list[dict]:
    boot_time = proc_boot_time(proc_root)
    endings = []
    for pid, start_ticks in sorted(identities.items()):
        snapshot = read_process_snapshot(proc_root, pid, boot_time)
        require(
            snapshot["start_ticks_since_boot"] == start_ticks,
            f"pid identity changed during capture: {pid}",
        )
        endings.append(
            {
                "pid": pid,
                "state": snapshot["state"],
                "start_ticks_since_boot": start_ticks,
                "alive_at_capture_end": True,
            }
        )
    return endings


def capture_job_artifacts(
    project_dir: Path, manifest: dict, splits: dict[str, list[str]]
) -> list[dict]:
    payload = []
    for job in manifest["jobs"]:
        run_dir = Path(job["run_dir"])
        run_config_path = run_dir / "run_config.json"
        train_path = run_dir / "split_train.txt"
        val_path = run_dir / "split_val.txt"
        try:
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AttestationError(f"invalid run config: {run_config_path}") from error
        run_args = run_config.get("args", {})
        method = run_config.get("method_meta", {})
        expected_values = {
            "seed": job["seed"],
            "split_seed": EXPECTED_RUN_ARGS["split_seed"],
            "val_fraction": EXPECTED_RUN_ARGS["val_fraction"],
            "epochs": EXPECTED_RUN_ARGS["epochs"],
            "batch_size": EXPECTED_RUN_ARGS["batch_size"],
            "num_workers": EXPECTED_RUN_ARGS["num_workers"],
            "lr": EXPECTED_RUN_ARGS["lr"],
            "warm_epoch": EXPECTED_RUN_ARGS["warm_epoch"],
            "model_type": "mshnet",
            "mshnet_variant": "deterministic",
            "evaluation_protocol": "internal_holdout",
            "deep_supervision": "legacy_exact",
            "fusion_regularizer": "none",
            "deterministic": True,
            "evaluation_interval": 10,
            "skip_final_evaluation": False,
            "if_checkpoint": False,
            "run_label": job["job_id"],
            "run_dir": job["run_dir"],
            "train_split_sha256": EXPECTED_SPLIT["fit_sha256"],
            "val_split_sha256": EXPECTED_SPLIT["val_sha256"],
            "test_split_sha256": EXPECTED_SPLIT["official_test_sha256"],
        }
        for key, expected in expected_values.items():
            require(run_args.get(key) == expected, f"run config mismatch for {job['job_id']}: {key}")
        require(method.get("method") == "MSHNet-Deterministic", "run method label mismatch")
        actual_fit = [line.strip() for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_val = [line.strip() for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        require(actual_fit == splits["fit"], f"actual fit list mismatch: {job['job_id']}")
        require(actual_val == splits["val"], f"actual val list mismatch: {job['job_id']}")
        payload.append(
            {
                "job_id": job["job_id"],
                "seed": job["seed"],
                "run_config": stable_file_record(run_config_path, relative_to=project_dir),
                "materialized_fit_split": stable_file_record(train_path, relative_to=project_dir),
                "materialized_val_split": stable_file_record(val_path, relative_to=project_dir),
                "semantic_configuration_matches": True,
                "materialized_splits_match": True,
            }
        )
    return payload


def command_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise AttestationError(f"environment command failed: {' '.join(args)}") from error


def parse_nvidia_csv(text: str, fields: tuple[str, ...]) -> list[dict]:
    if not text.strip():
        return []
    rows = []
    for values in csv.reader(text.splitlines(), skipinitialspace=True):
        require(len(values) == len(fields), "unexpected nvidia-smi CSV row")
        rows.append({key: value.strip() for key, value in zip(fields, values)})
    return rows


def capture_runtime_environment(worker_pids: set[int]) -> dict:
    try:
        import torch
    except Exception as error:  # pragma: no cover - exercised in production
        raise AttestationError("cannot import torch in attestation interpreter") from error
    gpu_fields = ("index", "uuid", "name", "driver_version", "memory_total_mib", "pci_bus_id")
    gpu_text = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,pci.bus_id",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_rows = parse_nvidia_csv(gpu_text, gpu_fields)
    require(tuple(row["index"] for row in gpu_rows) == EXPECTED_GPUS, "physical GPU inventory mismatch")
    app_fields = ("pid", "gpu_uuid", "used_memory_mib")
    app_text = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    app_rows = parse_nvidia_csv(app_text, app_fields)
    target_apps = [row for row in app_rows if row["pid"].isdigit() and int(row["pid"]) in worker_pids]
    require({int(row["pid"]) for row in target_apps} == worker_pids, "not every worker owns a GPU compute context")
    require(len(target_apps) == len(worker_pids), "worker has multiple or ambiguous GPU contexts")
    uuid_by_index = {row["index"]: row["uuid"] for row in gpu_rows}
    require(len(uuid_by_index) == 3, "unexpected GPU inventory cardinality")
    packages = {}
    for distribution in ("torch", "torchvision", "numpy", "Pillow", "scikit-image", "tqdm"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "collection_policy": "metadata/NVML only; attester does not call torch.cuda initialization APIs",
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info),
            "executable_argv": sys.executable,
            "executable_resolved": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "packages": packages,
        "torch_build": {
            "version": torch.__version__,
            "git_version": getattr(torch.version, "git_version", None),
            "built_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "config": torch.__config__.show(),
        },
        "nvidia_smi": {
            "binary": command_output(["bash", "-lc", "command -v nvidia-smi"]),
            "gpus": gpu_rows,
            "target_compute_apps": sorted(target_apps, key=lambda row: int(row["pid"])),
            "all_compute_apps": app_rows,
        },
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"refusing to overwrite existing attestation: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_attestation(project_dir: Path, proc_root: Path, batch_id: str) -> dict:
    capture_started = utc_now()
    manifest, manifest_record = validate_manifest(project_dir, batch_id)
    processes, identities = discover_and_validate_processes(project_dir, proc_root, manifest)
    dependencies = discover_local_import_closure(project_dir, project_dir / "main.py")
    repository = capture_repository(project_dir, manifest, dependencies)
    dataset_files, split_names = capture_dataset_files(manifest)
    job_artifacts = capture_job_artifacts(project_dir, manifest, split_names)
    runtime_environment = capture_runtime_environment(set(identities))

    uuid_by_gpu = {
        row["index"]: row["uuid"] for row in runtime_environment["nvidia_smi"]["gpus"]
    }
    app_by_pid = {
        int(row["pid"]): row
        for row in runtime_environment["nvidia_smi"]["target_compute_apps"]
    }
    for worker in processes["workers"]:
        require(
            app_by_pid[worker["pid"]]["gpu_uuid"]
            == uuid_by_gpu[worker["expected_physical_gpu"]],
            f"worker {worker['pid']} GPU UUID does not match CUDA_VISIBLE_DEVICES",
        )
        worker["nvidia_gpu_uuid_matches"] = True

    endings = assert_processes_still_alive(proc_root, identities)
    capture_finished = utc_now()
    tool_record = stable_file_record(Path(__file__).resolve(), relative_to=project_dir)
    return {
        "schema": "trace-stage0-runtime-attestation/v1",
        "status": "PASS",
        "batch_id": batch_id,
        "capture": {
            "started_at_utc": capture_started,
            "finished_at_utc": capture_finished,
            "capture_during_run": True,
            "all_three_workers_alive_at_start_and_end": True,
            "launch_time_attestation": False,
            "claim_scope": (
                "A capture-time observation of live processes, current on-disk source/data, "
                "materialized run configuration, and runtime software/GPU metadata. It does "
                "not prove that capture-time files are byte-identical to files at process launch."
            ),
        },
        "attestation_tool": tool_record,
        "manifest": {"file": manifest_record, "semantic_validation": "PASS"},
        "repository": repository,
        "processes": {**processes, "end_of_capture_recheck": endings},
        "source_dependencies": dependencies,
        "job_artifacts": job_artifacts,
        "dataset_files": dataset_files,
        "runtime_environment": runtime_environment,
        "official_test_pixel_access": {
            "status": "NOT_PERFORMED",
            "opened_files": 0,
            "scope": "official-test split identifier file only",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=EXPECTED_BATCH_ID)
    parser.add_argument(
        "--project-dir", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    output = (
        project_dir
        / "repro_runs"
        / "clean"
        / args.batch_id
        / "runtime_attestation.json"
    )
    payload = build_attestation(project_dir, args.proc_root, args.batch_id)
    atomic_write_json(output, payload)
    print(f"PASS runtime attestation: {output}")
    print(f"sha256={stable_file_record(output)['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
