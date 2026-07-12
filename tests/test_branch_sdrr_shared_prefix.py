from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.branch_sdrr_shared_prefix import branch_shared_prefix, sha256


def _make_prefix(path: Path, checkpoint_epoch: int = 1) -> None:
    path.mkdir()
    meta = {"method": "MSHNet", "deep_supervision": "legacy_exact"}
    payload = {"epoch": checkpoint_epoch, "method_meta": meta}
    for filename in ("checkpoint.pkl", "checkpoint_best_iou.pkl"):
        torch.save(payload, path / filename)
    (path / "epoch_metric.log").write_text(
        "2026-07-12-00-00-00 - 0000 - IoU 0.1 - PD 0.2 - FA 3.0\n"
        "2026-07-12-00-00-01 - 0001 - IoU 0.2 - PD 0.3 - FA 2.0\n"
    )
    (path / "run_config.json").write_text(
        json.dumps({"args": {}, "method_meta": dict(meta)}) + "\n"
    )


def test_branch_records_parent_hash_and_changes_only_method_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "control"
    _make_prefix(source)
    parent_hash = sha256(source / "checkpoint.pkl")

    record = branch_shared_prefix(
        source,
        destination,
        variant="same_pixel_random_scale",
        total_epochs=4,
    )

    assert record["shared_through_epoch"] == 1
    assert record["files"][0]["parent_sha256"] == parent_hash
    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    assert payload["epoch"] == 1
    assert (
        payload["method_meta"]["deep_supervision"]
        == "crs_same_pixel_random_scale"
    )
    assert (
        payload["method_meta"]["method"]
        == "SDRR-SamePixelRandomScaleControl"
    )
    config = json.loads((destination / "run_config.json").read_text())
    assert config["args"]["epochs"] == 4
    assert config["args"]["checkpoint_dir"] == str(destination)


def test_branch_fails_closed_on_checkpoint_log_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _make_prefix(source, checkpoint_epoch=0)

    with pytest.raises(ValueError, match="checkpoint/log prefix mismatch"):
        branch_shared_prefix(
            source,
            tmp_path / "destination",
            variant="sdrr",
            total_epochs=4,
        )


def test_branch_refuses_to_overwrite_destination(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_prefix(source)
    destination.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        branch_shared_prefix(
            source,
            destination,
            variant="sdrr",
            total_epochs=4,
        )


def test_normalization_control_is_persisted_as_checkpoint_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "density"
    _make_prefix(source)

    branch_shared_prefix(
        source,
        destination,
        variant="sdrr",
        normalization="safe_density",
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    assert payload["method_meta"]["sdrr_normalization"] == "safe_density"
    assert (
        payload["method_meta"]["method"]
        == "SDRR-NormalizationControl-safe_density"
    )
