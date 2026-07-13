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


def test_control_can_disable_full_network_gradient_matching(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "contribution_match_only"
    _make_prefix(source)

    branch_shared_prefix(
        source,
        destination,
        variant="same_pixel_random_scale",
        match_shared_grad_norm=False,
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    assert payload["method_meta"]["sdrr_match_shared_grad_norm"] is False
    config = json.loads((destination / "run_config.json").read_text())
    assert config["args"]["sdrr_match_shared_grad_norm"] is False


def test_rcr_branch_persists_single_routing_objective(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "rcr"
    _make_prefix(source)

    branch_shared_prefix(
        source,
        destination,
        variant="rcr",
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    assert payload["method_meta"]["deep_supervision"] == (
        "crs_responsibility_routing"
    )
    assert payload["method_meta"]["method"] == (
        "RCR-ResponsibilityConservingRouting"
    )


def test_rdr_branch_persists_final_density_risk_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "rdr"
    _make_prefix(source)

    branch_shared_prefix(
        source,
        destination,
        variant="rdr",
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    metadata = payload["method_meta"]
    assert metadata["deep_supervision"] == "crs_responsibility_density"
    assert metadata["method"] == "RDR-ResponsibilityDensityRisk"
    assert metadata["sdrr_normalization"] == "safe_density"


def test_rcr_density_branch_is_named_explicitly(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "rcr_density"
    _make_prefix(source)

    branch_shared_prefix(
        source,
        destination,
        variant="rcr",
        normalization="safe_density",
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    assert payload["method_meta"]["sdrr_normalization"] == "safe_density"
    assert payload["method_meta"]["method"] == (
        "RCR-ResponsibilityConservingRouting-Density"
    )


def test_baseline_can_branch_from_the_identical_registered_prefix(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "baseline"
    _make_prefix(source)

    record = branch_shared_prefix(
        source,
        destination,
        variant="baseline",
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    metadata = payload["method_meta"]
    assert record["variant"] == "baseline"
    assert metadata["method"] == "Canonical-MSHNet"
    assert metadata["deep_supervision"] == "legacy_exact"
    assert metadata["crs_lambda"] == 0.0
    assert metadata["rods_log_interval"] == 0


def test_oso_branch_changes_only_registered_fusion_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "oso"
    _make_prefix(source)

    record = branch_shared_prefix(
        source,
        destination,
        variant="oso",
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    metadata = payload["method_meta"]
    assert record["variant"] == "oso"
    assert metadata["method"] == "OSO-MSHNet"
    assert metadata["mshnet_variant"] == "oso"
    assert metadata["deep_supervision"] == "legacy_exact"
    assert metadata["crs_lambda"] == 0.0
    assert metadata["rods_log_interval"] == 0


def test_dsf_branch_changes_only_registered_fusion_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dsf"
    _make_prefix(source)

    branch_shared_prefix(
        source,
        destination,
        variant="dsf",
        total_epochs=4,
    )

    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    metadata = payload["method_meta"]
    assert metadata["method"] == "DSF-MSHNet"
    assert metadata["mshnet_variant"] == "dsf"
    assert metadata["deep_supervision"] == "legacy_exact"
    assert metadata["crs_lambda"] == 0.0


def test_dcdf_branch_changes_only_registered_fusion_semantics(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "dcdf"
    _make_prefix(source)
    branch_shared_prefix(
        source,
        destination,
        variant="dcdf",
        total_epochs=4,
    )
    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    metadata = payload["method_meta"]
    assert metadata["method"] == "DCDF-MSHNet"
    assert metadata["mshnet_variant"] == "dcdf"
    assert metadata["crs_lambda"] == 0.0


def test_ccfd_branch_migrates_zero_stencil_and_adagrad_state(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "ccfd"
    _make_prefix(source)
    for filename in ("checkpoint.pkl", "checkpoint_best_iou.pkl"):
        path = source / filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["net"] = {"final.weight": torch.ones(1)}
        payload["optimizer"] = {
            "param_groups": [{"params": [0]}],
            "state": {
                0: {
                    "step": torch.tensor(7.0),
                    "sum": torch.ones(1),
                }
            },
        }
        torch.save(payload, path)

    branch_shared_prefix(
        source,
        destination,
        variant="ccfd",
        total_epochs=4,
    )
    payload = torch.load(
        destination / "checkpoint.pkl", map_location="cpu", weights_only=False
    )
    assert payload["method_meta"]["method"] == "CCFD-MSHNet"
    assert payload["method_meta"]["mshnet_variant"] == "ccfd"
    assert torch.equal(
        payload["net"]["conflict_stencil.theta"], torch.zeros(8)
    )
    assert payload["optimizer"]["param_groups"][0]["params"] == [0, 1]
    assert payload["optimizer"]["state"][1]["step"].item() == 0.0
    assert torch.equal(
        payload["optimizer"]["state"][1]["sum"], torch.zeros(8)
    )


def test_official_sparse_evaluation_prefix_can_branch_at_last_logged_epoch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_prefix(source, checkpoint_epoch=250)
    (source / "epoch_metric.log").write_text(
        "2026-07-12-00-00-00 - 0250 - IoU 0.2 - PD 0.3 - FA 2.0\n"
    )
    config = json.loads((source / "run_config.json").read_text())
    config["args"].update(
        {
            "evaluation_protocol": "official_train_test",
            "evaluation_interval": 251,
        }
    )
    (source / "run_config.json").write_text(json.dumps(config) + "\n")

    record = branch_shared_prefix(
        source,
        tmp_path / "candidate",
        variant="sdrr",
        total_epochs=400,
    )

    assert record["shared_through_epoch"] == 250


def test_official_no_evaluation_prefix_can_branch_without_metric_log(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _make_prefix(source, checkpoint_epoch=250)
    (source / "epoch_metric.log").unlink()
    config = json.loads((source / "run_config.json").read_text())
    config["args"].update(
        {
            "evaluation_protocol": "official_train_test",
            "evaluation_interval": 400,
            "skip_final_evaluation": True,
        }
    )
    (source / "run_config.json").write_text(json.dumps(config) + "\n")

    destination = tmp_path / "candidate"
    record = branch_shared_prefix(
        source,
        destination,
        variant="sdrr",
        total_epochs=400,
    )

    assert record["shared_through_epoch"] == 250
    assert not (destination / "epoch_metric.log").exists()
