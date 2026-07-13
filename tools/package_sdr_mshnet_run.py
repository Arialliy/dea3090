#!/usr/bin/env python3
"""Package a clean deterministic SDRR run as the complete SDR-MSHNet class."""

from __future__ import annotations

import argparse
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

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.sdr_mshnet import SDRMSHNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_identity(state: dict[str, Any]) -> dict[str, Any]:
    baseline = DeterministicMSHNet(3).eval()
    model = SDRMSHNet(3).eval()
    baseline.load_state_dict(state, strict=True)
    model.load_state_dict(state, strict=True)
    if tuple(baseline.state_dict()) != tuple(model.state_dict()):
        raise RuntimeError("SDR-MSHNet state schema differs from the baseline")

    x = torch.linspace(-1.0, 1.0, steps=3 * 32 * 32).reshape(1, 3, 32, 32)
    with torch.no_grad():
        baseline_sides, baseline_pred = baseline(x, True)
        model_sides, model_pred = model(x, True)
        responsibility_state = model(
            x, True, return_responsibility_state=True
        )
    for baseline_side, model_side in zip(baseline_sides, model_sides):
        torch.testing.assert_close(model_side, baseline_side, rtol=0, atol=0)
    torch.testing.assert_close(model_pred, baseline_pred, rtol=0, atol=0)
    torch.testing.assert_close(
        responsibility_state["pred"], baseline_pred, rtol=0, atol=0
    )
    torch.testing.assert_close(
        responsibility_state["reconstructed"],
        baseline_pred,
        rtol=1e-5,
        atol=5e-5,
    )
    return {
        "strict_state_load": True,
        "default_forward_bit_exact": True,
        "responsibility_reconstruction_close": True,
        "parameter_elements": sum(p.numel() for p in model.parameters()),
        "added_parameter_elements": 0,
    }


def _update_metadata(metadata: dict[str, Any], run_label: str) -> None:
    metadata.update(
        {
            "method": "SDR-MSHNet",
            "mshnet_variant": "sdr",
            "fusion_regularizer": "sdrr",
            "deep_supervision": "crs_flip_suppression",
            "training_objective": "Scale-Deletion Responsibility Refinement",
            "run_label": run_label,
        }
    )


def package_run(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_dir():
        raise ValueError(f"source run does not exist: {source}")
    if destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    for filename in ("checkpoint.pkl", "checkpoint_best_iou.pkl", "run_config.json"):
        if not (source / filename).is_file():
            raise ValueError(f"source run missing {filename}")

    shutil.copytree(source, destination)
    records: list[dict[str, Any]] = []
    identity: dict[str, Any] | None = None
    try:
        for target in sorted(destination.glob("checkpoint*.pkl")):
            parent = source / target.name
            payload = torch.load(target, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or not isinstance(payload.get("net"), dict):
                raise ValueError(f"invalid checkpoint payload: {target}")
            identity = _strict_identity(payload["net"])
            if isinstance(payload.get("optimizer"), dict):
                model = SDRMSHNet(3)
                groups = payload["optimizer"].get("param_groups", [])
                if len(groups) != 1:
                    raise ValueError("expected one optimizer parameter group")
                optimizer = Adagrad(model.parameters(), lr=float(groups[0]["lr"]))
                optimizer.load_state_dict(payload["optimizer"])
            _update_metadata(payload.setdefault("method_meta", {}), destination.name)
            parent_hash = sha256(parent)
            torch.save(payload, target)
            records.append(
                {
                    "file": target.name,
                    "parent_sha256": parent_hash,
                    "packaged_sha256": sha256(target),
                    "strict_optimizer_load": isinstance(payload.get("optimizer"), dict),
                }
            )

        for target in sorted(destination.glob("weight*.pkl")):
            state = torch.load(target, map_location="cpu", weights_only=False)
            if not isinstance(state, dict):
                raise ValueError(f"invalid weight payload: {target}")
            identity = _strict_identity(state)
            records.append(
                {
                    "file": target.name,
                    "parent_sha256": sha256(source / target.name),
                    "packaged_sha256": sha256(target),
                    "unchanged_raw_weights": True,
                }
            )

        config_path = destination / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.setdefault("args", {}).update(
            {
                "mshnet_variant": "sdr",
                "fusion_regularizer": "sdrr",
                "deep_supervision": "crs_flip_suppression",
                "run_label": destination.name,
                "checkpoint_dir": str(destination),
            }
        )
        _update_metadata(config.setdefault("method_meta", {}), destination.name)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        report = {
            "package": "complete_parameter_identical_sdr_mshnet",
            "source": str(source),
            "destination": str(destination),
            "identity": identity,
            "records": records,
            "metrics_copied_without_recomputation": True,
            "performance_claim_scope": "internal_holdout_only",
        }
        (destination / "sdr_mshnet_package.json").write_text(
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
    print(json.dumps(package_run(args.source, args.destination), indent=2))


if __name__ == "__main__":
    main()
