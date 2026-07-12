#!/usr/bin/env python3
"""Create an auditable SDRR/control branch from a completed shared prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch


METHODS = {
    "sdrr": ("crs_flip_suppression", "SDRR-ScaleDeletionResponsibility"),
    "scale_budget_random": (
        "crs_matched_random",
        "SDRR-ScaleBudgetRandomControl-Unmatched",
    ),
    "same_pixel_random_scale": (
        "crs_same_pixel_random_scale",
        "SDRR-SamePixelRandomScaleControl",
    ),
    "magnitude_nonpivotal": (
        "crs_magnitude_nonpivotal",
        "SDRR-MagnitudeMatchedNonPivotalControl",
    ),
}
COPY_FILES = (
    "checkpoint.pkl",
    "checkpoint_best_iou.pkl",
    "checkpoint_pd_fa_best.pkl",
    "weight.pkl",
    "weight_pd_fa_best.pkl",
    "epoch_metric.log",
    "metric.log",
    "metric_pd_fa_best.log",
    "split_train.txt",
    "split_val.txt",
    "run_config.json",
)
EPOCH_RE = re.compile(r"-\s+(?P<epoch>\d+)\s+-\s+IoU")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _last_logged_epoch(path: Path) -> int:
    epochs: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_RE.search(line.replace("\t", " "))
        if match is not None:
            epochs.append(int(match.group("epoch")))
    if not epochs:
        raise ValueError(f"no epochs parsed from {path}")
    if epochs != list(range(epochs[-1] + 1)):
        raise ValueError(f"non-contiguous shared-prefix metric log: {path}")
    return epochs[-1]


def branch_shared_prefix(
    source: Path,
    destination: Path,
    *,
    variant: str,
    total_epochs: int = 400,
    crs_lambda: float = 0.05,
    start_epoch: int = 250,
    ramp_epochs: int = 50,
    safe_kernel: int = 15,
    detach_scale_evidence: bool = False,
    log_interval: int = 40,
    normalization: str = "event",
) -> dict[str, Any]:
    if variant not in METHODS:
        raise ValueError(f"unknown branch variant: {variant}")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    if not source.is_dir():
        raise ValueError(f"source run does not exist: {source}")
    for required in ("checkpoint.pkl", "epoch_metric.log", "run_config.json"):
        if not (source / required).is_file():
            raise ValueError(f"source run missing {required}: {source}")

    checkpoint = torch.load(
        source / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    shared_epoch = int(checkpoint.get("epoch", -1))
    logged_epoch = _last_logged_epoch(source / "epoch_metric.log")
    if shared_epoch != logged_epoch:
        raise ValueError(
            f"checkpoint/log prefix mismatch: {shared_epoch} != {logged_epoch}"
        )
    if total_epochs <= shared_epoch + 1:
        raise ValueError("total_epochs must extend beyond the shared prefix")

    deep_supervision, method = METHODS[variant]
    if normalization not in ("event", "safe_density", "unique_pixel"):
        raise ValueError(
            "normalization must be event, safe_density, or unique_pixel"
        )
    if variant != "sdrr" and normalization != "event":
        raise ValueError("normalization controls must branch from variant=sdrr")
    if variant == "sdrr" and normalization != "event":
        method = "SDRR-NormalizationControl-" + normalization
    updates: dict[str, Any] = {
        "method": method,
        "deep_supervision": deep_supervision,
        "crs_lambda": float(crs_lambda),
        "crs_start_epoch": int(start_epoch),
        "crs_ramp_epochs": int(ramp_epochs),
        "crs_safe_kernel": int(safe_kernel),
        "crs_detach_scale_evidence": bool(detach_scale_evidence),
        "sdrr_normalization": normalization,
        "rods_log_interval": int(log_interval),
        "run_label": destination.name,
    }

    destination.mkdir(parents=True)
    copied: list[str] = []
    try:
        for filename in COPY_FILES:
            parent = source / filename
            if parent.is_file():
                shutil.copy2(parent, destination / filename)
                copied.append(filename)

        file_records: list[dict[str, Any]] = []
        for filename in (
            "checkpoint.pkl",
            "checkpoint_best_iou.pkl",
            "checkpoint_pd_fa_best.pkl",
        ):
            parent = source / filename
            branched = destination / filename
            if not branched.is_file():
                continue
            payload = torch.load(branched, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not isinstance(
                payload.get("method_meta"), dict
            ):
                raise ValueError(f"checkpoint lacks method_meta: {parent}")
            payload["method_meta"].update(updates)
            torch.save(payload, branched)
            file_records.append(
                {
                    "file": filename,
                    "parent": str(parent),
                    "parent_sha256": sha256(parent),
                    "branched_sha256": sha256(branched),
                }
            )

        config_path = destination / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config.get("args"), dict) or not isinstance(
            config.get("method_meta"), dict
        ):
            raise ValueError("run_config.json lacks args or method_meta")
        config_updates = {
            **updates,
            "checkpoint_dir": str(destination),
            "epochs": int(total_epochs),
            "if_checkpoint": True,
        }
        config["args"].update(config_updates)
        config["method_meta"].update(updates)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        record = {
            "branch_type": "shared_prefix_training_objective_branch",
            "variant": variant,
            "shared_through_epoch": shared_epoch,
            "total_epochs": int(total_epochs),
            "updates": updates,
            "copied_files": copied,
            "files": file_records,
        }
        (destination / "checkpoint_branch.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(METHODS), required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--crs-lambda", type=float, default=0.05)
    parser.add_argument("--start-epoch", type=int, default=250)
    parser.add_argument("--ramp-epochs", type=int, default=50)
    parser.add_argument("--safe-kernel", type=int, default=15)
    parser.add_argument("--detach-scale-evidence", action="store_true")
    parser.add_argument("--log-interval", type=int, default=40)
    parser.add_argument(
        "--normalization",
        choices=("event", "safe_density", "unique_pixel"),
        default="event",
    )
    args = parser.parse_args()
    result = branch_shared_prefix(
        args.source,
        args.destination,
        variant=args.variant,
        total_epochs=args.epochs,
        crs_lambda=args.crs_lambda,
        start_epoch=args.start_epoch,
        ramp_epochs=args.ramp_epochs,
        safe_kernel=args.safe_kernel,
        detach_scale_evidence=args.detach_scale_evidence,
        log_interval=args.log_interval,
        normalization=args.normalization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
