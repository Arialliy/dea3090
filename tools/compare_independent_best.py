#!/usr/bin/env python3
"""Fail-closed comparison of each run's independently selected best checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch


METRIC_RE = re.compile(
    r"(?P<epoch>\d+)\s+-\s+IoU\s+(?P<iou>[0-9.]+)\s+-\s+"
    r"PD\s+(?P<pd>[0-9.]+)\s+-\s+FA\s+(?P<fa>[0-9.]+)"
)
LOG_ROUNDING_TOLERANCE = 5.1e-5
FORMAL_EVALUATION_INTERVAL = 10
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# These settings must be identical when the comparison is intended to isolate
# a physical MSHNet variant.  Variant-specific implementation choices are
# deliberately absent; everything below controls data, optimization, loss, or
# the fixed evaluation schedule.
PAIRED_TRAINING_FIELDS = (
    "model_type",
    "deep_supervision",
    "fusion_regularizer",
    "aux_loss_weight",
    "empty_side_policy",
    "dea_lambda_single",
    "dea_lambda_dec",
    "dea_lambda_empty",
    "dea_ramp_epochs",
    "dea_tau",
    "lr",
    "warm_epoch",
    "batch_size",
    "num_workers",
    "base_size",
    "crop_size",
    "deterministic",
    "train_split_file",
    "val_split_file",
    "test_split_file",
    "val_fraction",
    "split_seed",
    "evaluation_protocol",
    "epochs",
    "evaluation_interval",
    "skip_final_evaluation",
    "init_from_baseline",
    "origin_baseline_checkpoint",
)

# ``get_method_metadata`` persists this subset in run_config.json and in every
# formal checkpoint.  Checking all three copies prevents a checkpoint from a
# parameter-identical but semantically different forward (for example SPT0 or
# BCSF) from being silently relabelled by its directory name.
METHOD_META_ARG_FIELDS = (
    "model_type",
    "mshnet_variant",
    "seed",
    "deterministic",
    "dataset_dir",
    "train_split_file",
    "val_split_file",
    "test_split_file",
    "train_split_sha256",
    "val_split_sha256",
    "test_split_sha256",
    "evaluation_protocol",
    "evaluation_interval",
    "skip_final_evaluation",
    "deep_supervision",
    "fusion_regularizer",
    "aux_loss_weight",
    "init_from_baseline",
)


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_run_config(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise ValueError(f"missing run config: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    args = payload.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"run config has no args mapping: {path}")
    method_meta = payload.get("method_meta")
    if not isinstance(method_meta, dict) or not method_meta:
        raise ValueError(f"run config has no method_meta mapping: {path}")
    return args, method_meta


def _require_fields(mapping: dict[str, Any], fields: tuple[str, ...], *, label: str) -> None:
    missing = [field for field in fields if field not in mapping]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")


def _validate_sha256(value: Any, *, label: str) -> str:
    normalized = str(value).lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256 digest: {value!r}")
    return normalized


def _normalized_split(path: Path) -> tuple[list[str], str]:
    if not path.is_file():
        raise ValueError(f"missing split manifest: {path}")
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"empty split manifest: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate sample IDs in split manifest: {path}")
    digest = hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()
    return names, digest


def _source_split_path(args: dict[str, Any], split: str) -> Path:
    configured = Path(str(args[f"{split}_split_file"]))
    if configured.is_absolute():
        return configured
    return Path(str(args["dataset_dir"])) / configured


def _audit_split_manifests(run_dir: Path, args: dict[str, Any]) -> dict[str, Any]:
    expected_hashes = {
        split: _validate_sha256(
            args[f"{split}_split_sha256"],
            label=f"run config {split}_split_sha256",
        )
        for split in ("train", "test")
    }
    names_by_split: dict[str, list[str]] = {}
    snapshots: dict[str, Any] = {}
    for split in ("train", "test"):
        snapshot_path = run_dir / f"split_{split}.txt"
        names, snapshot_hash = _normalized_split(snapshot_path)
        if snapshot_hash != expected_hashes[split]:
            raise ValueError(
                f"{split} split snapshot/hash mismatch for {run_dir}: "
                f"{snapshot_hash} != {expected_hashes[split]}"
            )
        names_by_split[split] = names

        source_path = _source_split_path(args, split)
        source_verified = False
        if source_path.is_file():
            source_names, source_hash = _normalized_split(source_path)
            if source_hash != expected_hashes[split] or source_names != names:
                raise ValueError(
                    f"configured {split} manifest disagrees with the persisted "
                    f"run snapshot: {source_path}"
                )
            source_verified = True
        snapshots[split] = {
            "snapshot": str(snapshot_path),
            "source": str(source_path),
            "source_verified": source_verified,
            "count": len(names),
            "sha256_normalized": snapshot_hash,
        }

    overlap = sorted(set(names_by_split["train"]) & set(names_by_split["test"]))
    if overlap:
        raise ValueError(
            f"persisted train/test snapshots overlap for {run_dir}: {overlap[:5]}"
        )
    return {"overlap_count": 0, **snapshots}


def _validate_method_metadata(
    *,
    args: dict[str, Any],
    run_metadata: dict[str, Any],
    checkpoint_metadata: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(checkpoint_metadata, dict) or not checkpoint_metadata:
        raise ValueError(f"{label} has no method_meta mapping")
    _require_fields(args, METHOD_META_ARG_FIELDS, label="run config args")
    _require_fields(run_metadata, METHOD_META_ARG_FIELDS, label="run method_meta")
    _require_fields(
        checkpoint_metadata,
        METHOD_META_ARG_FIELDS,
        label=f"{label} method_meta",
    )
    if checkpoint_metadata != run_metadata:
        differing = sorted(
            key
            for key in set(checkpoint_metadata) | set(run_metadata)
            if checkpoint_metadata.get(key) != run_metadata.get(key)
        )
        raise ValueError(
            f"{label} method_meta does not match run_config.json: {differing}"
        )
    mismatches = {
        field: {"args": args[field], "method_meta": run_metadata[field]}
        for field in METHOD_META_ARG_FIELDS
        if args[field] != run_metadata[field]
    }
    if mismatches:
        raise ValueError(f"run args/method_meta mismatch: {mismatches}")
    return {field: run_metadata[field] for field in METHOD_META_ARG_FIELDS}


def expected_evaluation_epochs(args: dict[str, Any]) -> list[int]:
    epochs = int(args["epochs"])
    interval = int(args["evaluation_interval"])
    if epochs < 1 or interval < 1:
        raise ValueError("epochs and evaluation_interval must be positive")
    expected = list(range(interval - 1, epochs, interval))
    if not bool(args.get("skip_final_evaluation", False)) and epochs - 1 not in expected:
        expected.append(epochs - 1)
    return expected


def _read_metric_rows(path: Path) -> list[dict[str, float | int]]:
    if not path.is_file():
        raise ValueError(f"missing metric log: {path}")
    rows: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = METRIC_RE.search(line.replace("\t", " "))
        if match is None:
            continue
        rows.append(
            {
                "epoch": int(match.group("epoch")),
                "iou": float(match.group("iou")),
                "pd": float(match.group("pd")),
                "fa": float(match.group("fa")),
            }
        )
    return rows


def summarize_independent_best(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    invalid = run_dir / "INVALID_RUN.json"
    if invalid.exists():
        raise ValueError(f"run is explicitly invalid: {invalid}")

    args, run_metadata = _read_run_config(run_dir)
    required_args = tuple(
        dict.fromkeys(
            (
                "mshnet_variant",
                "seed",
                "dataset_dir",
                "train_split_sha256",
                "test_split_sha256",
                *PAIRED_TRAINING_FIELDS,
            )
        )
    )
    _require_fields(args, required_args, label="run config args")
    if args["evaluation_protocol"] != "official_train_test":
        raise ValueError(
            "independent-best comparison requires evaluation_protocol="
            "official_train_test"
        )
    if int(args["evaluation_interval"]) != FORMAL_EVALUATION_INTERVAL:
        raise ValueError(
            "formal independent-best comparison requires evaluation_interval="
            f"{FORMAL_EVALUATION_INTERVAL}, got {args['evaluation_interval']}"
        )
    split_audit = _audit_split_manifests(run_dir, args)
    training_config = {field: args[field] for field in PAIRED_TRAINING_FIELDS}
    training_config_sha256 = hashlib.sha256(
        json.dumps(
            training_config,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    expected = expected_evaluation_epochs(args)
    rows = _read_metric_rows(run_dir / "epoch_metric.log")
    if not rows:
        raise ValueError(f"metric log has no parseable rows: {run_dir}")
    for row in rows:
        for key in ("iou", "pd", "fa"):
            if not math.isfinite(float(row[key])):
                raise ValueError(f"non-finite {key} in metric log: {row}")
    logged_epochs = [int(row["epoch"]) for row in rows]
    if logged_epochs != expected:
        raise ValueError(
            f"evaluation schedule mismatch for {run_dir}: "
            f"expected {expected}, got {logged_epochs}"
        )

    checkpoint_path = run_dir / "checkpoint_best_iou.pkl"
    if not checkpoint_path.is_file():
        raise ValueError(f"missing independently selected best checkpoint: {checkpoint_path}")
    checkpoint = _load_torch(checkpoint_path)
    required = ("epoch", "iou", "pd", "fa", "best_iou", "method_meta")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"best checkpoint is missing fields {missing}: {checkpoint_path}")
    method_identity = _validate_method_metadata(
        args=args,
        run_metadata=run_metadata,
        checkpoint_metadata=checkpoint["method_meta"],
        label="best-IoU checkpoint",
    )
    for key in ("iou", "pd", "fa", "best_iou"):
        if not math.isfinite(float(checkpoint[key])):
            raise ValueError(f"best checkpoint has non-finite {key}: {checkpoint[key]}")

    epoch = int(checkpoint["epoch"])
    by_epoch = {int(row["epoch"]): row for row in rows}
    if epoch not in by_epoch:
        raise ValueError(f"best checkpoint epoch {epoch} was not an evaluation epoch")
    row = by_epoch[epoch]
    for key in ("iou", "pd", "fa"):
        if abs(float(checkpoint[key]) - float(row[key])) > LOG_ROUNDING_TOLERANCE:
            raise ValueError(
                f"checkpoint/log {key} mismatch at epoch {epoch}: "
                f"{checkpoint[key]} != {row[key]}"
            )
    if abs(float(checkpoint["iou"]) - float(checkpoint["best_iou"])) > 1e-12:
        raise ValueError("checkpoint iou and best_iou disagree")

    # The final resumable checkpoint contains the exact accumulated best_iou,
    # unlike the four-decimal text log.  It therefore closes the rounding-tie
    # loophole when proving that checkpoint_best_iou.pkl is this run's own
    # independently selected optimum.
    latest_path = run_dir / "checkpoint.pkl"
    if not latest_path.is_file():
        raise ValueError(f"missing final resumable checkpoint: {latest_path}")
    latest = _load_torch(latest_path)
    _require_fields(
        latest,
        ("epoch", "best_iou", "method_meta"),
        label="final resumable checkpoint",
    )
    _validate_method_metadata(
        args=args,
        run_metadata=run_metadata,
        checkpoint_metadata=latest["method_meta"],
        label="final resumable checkpoint",
    )
    if int(latest["epoch"]) != int(args["epochs"]) - 1:
        raise ValueError(
            f"run is incomplete: final checkpoint epoch {latest['epoch']} != "
            f"{int(args['epochs']) - 1}"
        )
    if not math.isfinite(float(latest["best_iou"])):
        raise ValueError("final checkpoint has non-finite best_iou")
    if abs(float(latest["best_iou"]) - float(checkpoint["iou"])) > 1e-12:
        raise ValueError(
            "checkpoint_best_iou.pkl does not match the exact global best_iou "
            "stored by the completed run"
        )

    max_logged_iou = max(float(item["iou"]) for item in rows)
    if float(row["iou"]) != max_logged_iou:
        raise ValueError(
            f"checkpoint epoch {epoch} is not a best-IoU evaluation: "
            f"logged {row['iou']} < {max_logged_iou}"
        )

    return {
        "run_dir": str(run_dir),
        "variant": args["mshnet_variant"],
        "seed": int(args["seed"]),
        "dataset_dir": args["dataset_dir"],
        "evaluation_protocol": args["evaluation_protocol"],
        "train_split_sha256": _validate_sha256(
            args["train_split_sha256"], label="train_split_sha256"
        ),
        "test_split_sha256": _validate_sha256(
            args["test_split_sha256"], label="test_split_sha256"
        ),
        "split_audit": split_audit,
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
        "method_identity": method_identity,
        "epochs": int(args["epochs"]),
        "evaluation_interval": int(args["evaluation_interval"]),
        "evaluated_epochs": expected,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "final_checkpoint": str(latest_path),
        "final_checkpoint_sha256": _sha256(latest_path),
        "best": {
            "epoch": epoch,
            "iou": float(checkpoint["iou"]),
            "pd": float(checkpoint["pd"]),
            "fa": float(checkpoint["fa"]),
        },
    }


def compare_runs(baseline_run: Path, candidate_run: Path) -> dict[str, Any]:
    baseline = summarize_independent_best(baseline_run)
    candidate = summarize_independent_best(candidate_run)
    paired_fields = (
        "seed",
        "dataset_dir",
        "evaluation_protocol",
        "train_split_sha256",
        "test_split_sha256",
        "epochs",
        "evaluation_interval",
        "evaluated_epochs",
        "training_config_sha256",
    )
    mismatches = {
        field: {"baseline": baseline[field], "candidate": candidate[field]}
        for field in paired_fields
        if baseline[field] != candidate[field]
    }
    if mismatches:
        raise ValueError(f"runs do not form a paired protocol: {mismatches}")
    baseline_best = baseline["best"]
    candidate_best = candidate["best"]
    return {
        "selection_rule": "independent_per_run_best_iou_checkpoint",
        "same_epoch_required": False,
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {
            key: float(candidate_best[key]) - float(baseline_best[key])
            for key in ("iou", "pd", "fa")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = compare_runs(args.baseline_run, args.candidate_run)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
