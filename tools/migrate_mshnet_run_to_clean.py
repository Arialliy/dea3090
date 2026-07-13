#!/usr/bin/env python3
"""Remove the dormant DEA-lite head from an MSHNet run, with SHA audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adagrad

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.MSHNet import MSHNet as WorkbenchMSHNet
from model.baselines.mshnet_deterministic import MSHNet as CleanMSHNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schemas() -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    workbench_model = WorkbenchMSHNet(3)
    clean_model = CleanMSHNet(3)
    workbench_names = [name for name, _ in workbench_model.named_parameters()]
    clean_names = [name for name, _ in clean_model.named_parameters()]
    extras = [name for name in workbench_names if name not in set(clean_names)]
    if clean_names != [name for name in workbench_names if name not in extras]:
        raise RuntimeError("clean MSHNet parameter order is not a workbench subsequence")
    if not extras or not all(name.startswith("decidability_head.") for name in extras):
        raise RuntimeError(f"unexpected workbench-only parameters: {extras}")
    clean_state_names = list(clean_model.state_dict())
    extra_state_names = [
        name for name in workbench_model.state_dict() if name not in set(clean_state_names)
    ]
    if not extra_state_names or not all(
        name.startswith("decidability_head.") for name in extra_state_names
    ):
        raise RuntimeError(f"unexpected workbench-only state: {extra_state_names}")
    return (
        workbench_names,
        clean_names,
        extras,
        clean_state_names,
        extra_state_names,
    )


def _state_is_zero(state: dict[str, Any]) -> bool:
    for value in state.values():
        if torch.is_tensor(value) and bool((value != 0).any()):
            return False
    return True


def clean_checkpoint_payload(
    payload: dict[str, Any],
    *,
    run_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload.get("net"), dict):
        raise ValueError("checkpoint lacks net state_dict")
    if not isinstance(payload.get("optimizer"), dict):
        raise ValueError("checkpoint lacks optimizer state_dict")
    (
        workbench_names,
        clean_names,
        extras,
        clean_state_names,
        extra_state_names,
    ) = _schemas()
    net = payload["net"]
    missing = set(clean_state_names) - set(net)
    if missing:
        raise ValueError(f"checkpoint missing clean keys: {sorted(missing)[:5]}")
    unexpected = set(net) - set(clean_state_names) - set(extra_state_names)
    if unexpected:
        raise ValueError(f"checkpoint has unexpected model keys: {sorted(unexpected)[:5]}")

    optimizer = copy.deepcopy(payload["optimizer"])
    groups = optimizer.get("param_groups", [])
    if len(groups) != 1:
        raise ValueError("migration expects exactly one optimizer parameter group")
    saved_ids = list(groups[0].get("params", []))
    if len(saved_ids) != len(workbench_names):
        raise ValueError("optimizer parameter order does not match workbench MSHNet")
    extra_positions = [workbench_names.index(name) for name in extras]
    extra_ids = [saved_ids[position] for position in extra_positions]
    for parameter_id in extra_ids:
        state = optimizer.get("state", {}).get(parameter_id, {})
        if not _state_is_zero(state):
            raise ValueError(
                f"refusing to remove non-zero optimizer state for id {parameter_id}"
            )
    keep_ids = [
        parameter_id
        for position, parameter_id in enumerate(saved_ids)
        if position not in set(extra_positions)
    ]
    groups[0]["params"] = keep_ids
    optimizer["state"] = {
        parameter_id: state
        for parameter_id, state in optimizer.get("state", {}).items()
        if parameter_id in set(keep_ids)
    }

    migrated = copy.deepcopy(payload)
    migrated["net"] = {name: net[name] for name in clean_state_names}
    migrated["optimizer"] = optimizer
    metadata = migrated.setdefault("method_meta", {})
    metadata["mshnet_variant"] = "deterministic"
    metadata["run_label"] = run_label
    deep_supervision = metadata.get("deep_supervision", "legacy_exact")
    if deep_supervision == "legacy_exact":
        metadata["method"] = "MSHNet-Deterministic"
    elif deep_supervision == "crs_flip_suppression":
        metadata["method"] = "SDRR-ScaleDeletionResponsibility"

    clean_model = CleanMSHNet(3)
    clean_model.load_state_dict(migrated["net"], strict=True)
    clean_optimizer = Adagrad(clean_model.parameters(), lr=float(groups[0]["lr"]))
    clean_optimizer.load_state_dict(migrated["optimizer"])
    report = {
        "removed_parameter_names": extras,
        "removed_parameter_ids": extra_ids,
        "removed_parameter_elements": sum(
            parameter.numel()
            for name, parameter in WorkbenchMSHNet(3).named_parameters()
            if name in extras
        ),
        "clean_parameter_tensors": len(clean_names),
        "strict_model_and_optimizer_load": True,
    }
    return migrated, report


def migrate_run(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise ValueError(f"source run does not exist: {source}")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    records = []
    try:
        for source_checkpoint in sorted(source.glob("checkpoint*.pkl")):
            target = destination / source_checkpoint.name
            parent_hash = sha256(source_checkpoint)
            payload = torch.load(target, map_location="cpu", weights_only=False)
            migrated, details = clean_checkpoint_payload(
                payload, run_label=destination.name
            )
            torch.save(migrated, target)
            records.append(
                {
                    "file": source_checkpoint.name,
                    "parent_sha256": parent_hash,
                    "migrated_sha256": sha256(target),
                    **details,
                }
            )

        _, _, extras, clean_state_names, extra_state_names = _schemas()
        for source_weight in sorted(source.glob("weight*.pkl")):
            target = destination / source_weight.name
            state = torch.load(target, map_location="cpu", weights_only=False)
            if not isinstance(state, dict):
                raise ValueError(f"raw weight file is not a state_dict: {target}")
            missing = set(clean_state_names) - set(state)
            unexpected = (
                set(state) - set(clean_state_names) - set(extra_state_names)
            )
            if missing or unexpected:
                raise ValueError(
                    f"raw weight schema mismatch: missing={len(missing)} "
                    f"unexpected={len(unexpected)}"
                )
            parent_hash = sha256(source_weight)
            torch.save({name: state[name] for name in clean_state_names}, target)
            records.append(
                {
                    "file": source_weight.name,
                    "parent_sha256": parent_hash,
                    "migrated_sha256": sha256(target),
                    "removed_state_names": extra_state_names,
                }
            )

        config_path = destination / "run_config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.setdefault("args", {})["mshnet_variant"] = "deterministic"
            config["args"]["run_label"] = destination.name
            config.setdefault("method_meta", {})["mshnet_variant"] = "deterministic"
            config["method_meta"]["run_label"] = destination.name
            deep_supervision = config["method_meta"].get(
                "deep_supervision", "legacy_exact"
            )
            if deep_supervision == "legacy_exact":
                config["method_meta"]["method"] = "MSHNet-Deterministic"
            elif deep_supervision == "crs_flip_suppression":
                config["method_meta"]["method"] = (
                    "SDRR-ScaleDeletionResponsibility"
                )
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        report = {
            "migration": "workbench_to_parameter_identical_deterministic_mshnet",
            "source": str(source),
            "destination": str(destination),
            "records": records,
        }
        (destination / "clean_migration.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(migrate_run(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
