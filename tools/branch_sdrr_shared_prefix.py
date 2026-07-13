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
    "baseline": ("legacy_exact", "Canonical-MSHNet"),
    "oso": ("legacy_exact", "OSO-MSHNet"),
    "dsf": ("legacy_exact", "DSF-MSHNet"),
    "dcdf": ("legacy_exact", "DCDF-MSHNet"),
    "ccfd": ("legacy_exact", "CCFD-MSHNet"),
    "sdrr": ("crs_flip_suppression", "SDRR-ScaleDeletionResponsibility"),
    "rdr": (
        "crs_responsibility_density",
        "RDR-ResponsibilityDensityRisk",
    ),
    "rcr": (
        "crs_responsibility_routing",
        "RCR-ResponsibilityConservingRouting",
    ),
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
    "all_safe_fp": ("crs_all_safe_fp", "SDRR-AllSafeFPControl"),
    "same_pixel_fused": (
        "crs_same_pixel_fused",
        "SDRR-SamePivotalPixelFusedControl",
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
    "split_test.txt",
    "run_config.json",
)
EPOCH_RE = re.compile(r"-\s+(?P<epoch>\d+)\s+-\s+IoU")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_ccfd_state_to_checkpoint(payload: dict[str, Any]) -> None:
    """Migrate a canonical checkpoint to the zero-initialized CCFD model."""

    net = payload.get("net")
    optimizer = payload.get("optimizer")
    if not isinstance(net, dict) or not isinstance(optimizer, dict):
        raise ValueError("CCFD migration requires net and optimizer states")
    prefix = "module." if any(key.startswith("module.") for key in net) else ""
    key = prefix + "conflict_stencil.theta"
    if key in net:
        raise ValueError("checkpoint already contains CCFD stencil state")
    net[key] = torch.zeros(8, dtype=torch.float32)

    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state, dict):
        raise ValueError("CCFD migration requires one auditable optimizer group")
    parameter_ids = groups[0].get("params")
    if not isinstance(parameter_ids, list) or not parameter_ids:
        raise ValueError("CCFD migration found no optimizer parameter ids")
    if set(parameter_ids) != set(state):
        raise ValueError("CCFD migration requires initialized state for every parameter")
    new_id = max(parameter_ids) + 1
    if new_id in state:
        raise ValueError("CCFD optimizer parameter id collision")
    exemplar = state[parameter_ids[-1]]
    step = exemplar.get("step")
    accumulator = exemplar.get("sum")
    if not torch.is_tensor(step) or not torch.is_tensor(accumulator):
        raise ValueError("CCFD migration requires Adagrad step/sum state")
    parameter_ids.append(new_id)
    state[new_id] = {
        "step": torch.zeros_like(step),
        "sum": torch.zeros(8, dtype=accumulator.dtype, device=accumulator.device),
    }


def _add_ccfd_state_to_weight(state_dict: dict[str, Any]) -> None:
    if not state_dict or not all(torch.is_tensor(value) for value in state_dict.values()):
        raise ValueError("CCFD weight migration requires a raw state_dict")
    prefix = "module." if any(key.startswith("module.") for key in state_dict) else ""
    key = prefix + "conflict_stencil.theta"
    if key in state_dict:
        raise ValueError("weight already contains CCFD stencil state")
    state_dict[key] = torch.zeros(8, dtype=torch.float32)


def _last_logged_epoch(path: Path, *, require_contiguous: bool = True) -> int:
    epochs: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = EPOCH_RE.search(line.replace("\t", " "))
        if match is not None:
            epochs.append(int(match.group("epoch")))
    if not epochs:
        raise ValueError(f"no epochs parsed from {path}")
    if require_contiguous and epochs != list(range(epochs[-1] + 1)):
        raise ValueError(f"non-contiguous shared-prefix metric log: {path}")
    if epochs != sorted(set(epochs)):
        raise ValueError(f"metric epochs must be unique and increasing: {path}")
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
    match_shared_grad_norm: bool = True,
) -> dict[str, Any]:
    if variant not in METHODS:
        raise ValueError(f"unknown branch variant: {variant}")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    if not source.is_dir():
        raise ValueError(f"source run does not exist: {source}")
    for required in ("checkpoint.pkl", "run_config.json"):
        if not (source / required).is_file():
            raise ValueError(f"source run missing {required}: {source}")

    source_config = json.loads(
        (source / "run_config.json").read_text(encoding="utf-8")
    )
    source_args = source_config.get("args", {})
    sparse_evaluation = (
        source_args.get("evaluation_protocol") == "official_train_test"
        and int(source_args.get("evaluation_interval", 1)) > 1
    )
    no_evaluation_prefix = (
        source_args.get("evaluation_protocol") == "official_train_test"
        and bool(source_args.get("skip_final_evaluation", False))
    )
    checkpoint = torch.load(
        source / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    shared_epoch = int(checkpoint.get("epoch", -1))
    metric_log = source / "epoch_metric.log"
    if metric_log.is_file():
        logged_epoch = _last_logged_epoch(
            metric_log, require_contiguous=not sparse_evaluation
        )
        if shared_epoch != logged_epoch:
            raise ValueError(
                f"checkpoint/log prefix mismatch: {shared_epoch} != {logged_epoch}"
            )
    elif not no_evaluation_prefix:
        raise ValueError(
            "source run missing epoch_metric.log without an auditable "
            "official no-evaluation-prefix configuration"
        )
    if total_epochs <= shared_epoch + 1:
        raise ValueError("total_epochs must extend beyond the shared prefix")

    deep_supervision, method = METHODS[variant]
    if normalization not in ("event", "safe_density", "unique_pixel"):
        raise ValueError(
            "normalization must be event, safe_density, or unique_pixel"
        )
    if variant not in ("sdrr", "rcr") and normalization != "event":
        raise ValueError("normalization controls must branch from variant=sdrr")
    if variant in ("sdrr", "rcr") and normalization != "event":
        method = "SDRR-NormalizationControl-" + normalization
        if variant == "rcr":
            method = "RCR-ResponsibilityConservingRouting-Density"
    updates: dict[str, Any] = {
        "method": method,
        "deep_supervision": deep_supervision,
        "crs_lambda": (
            0.0
            if variant in ("baseline", "oso", "dsf", "dcdf", "ccfd")
            else float(crs_lambda)
        ),
        "crs_start_epoch": int(start_epoch),
        "crs_ramp_epochs": int(ramp_epochs),
        "crs_safe_kernel": int(safe_kernel),
        "crs_detach_scale_evidence": bool(detach_scale_evidence),
        # RDR has safe-background density normalization by definition.  The
        # legacy field remains populated so old checkpoints/tools can audit
        # its exact numerical equivalence to the winning SDRR density gate.
        "sdrr_normalization": (
            "safe_density" if variant == "rdr" else normalization
        ),
        "sdrr_match_shared_grad_norm": (
            variant not in ("baseline", "oso", "dsf", "dcdf", "ccfd")
            and bool(match_shared_grad_norm)
        ),
        "rods_log_interval": (
            0
            if variant in ("baseline", "oso", "dsf", "dcdf", "ccfd")
            else int(log_interval)
        ),
        "run_label": destination.name,
    }
    if variant == "oso":
        updates["mshnet_variant"] = "oso"
    if variant == "dsf":
        updates["mshnet_variant"] = "dsf"
    if variant == "dcdf":
        updates["mshnet_variant"] = "dcdf"
    if variant == "ccfd":
        updates["mshnet_variant"] = "ccfd"

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
            if variant == "ccfd":
                _add_ccfd_state_to_checkpoint(payload)
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

        if variant == "ccfd":
            for filename in ("weight.pkl", "weight_pd_fa_best.pkl"):
                path = destination / filename
                if not path.is_file():
                    continue
                state_dict = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                _add_ccfd_state_to_weight(state_dict)
                torch.save(state_dict, path)

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
        "--no-match-shared-grad-norm",
        action="store_false",
        dest="match_shared_grad_norm",
        help=(
            "Keep matching at the control's native intervention variable "
            "without an additional full-network gradient-norm rescale."
        ),
    )
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
        match_shared_grad_norm=args.match_shared_grad_norm,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
