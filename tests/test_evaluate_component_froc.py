from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.evaluate_component_froc import (
    MODEL_VARIANTS,
    extract_state_dict,
    parse_args,
    threshold_grid_metadata,
    validate_checkpoint_identity,
)


CHECKPOINT_HASH = "a" * 64
SPLIT_HASH = "b" * 64


def test_component_froc_tool_exposes_only_audited_variants() -> None:
    assert set(MODEL_VARIANTS) == {"deterministic", "ccfd", "spt0", "bcsf"}


def test_extract_state_dict_accepts_checkpoint_and_removes_data_parallel_prefix() -> None:
    tensor = torch.ones(1)
    state = extract_state_dict({"net": {"module.final.weight": tensor}})
    assert set(state) == {"final.weight"}
    assert state["final.weight"] is tensor


def test_extract_state_dict_fails_closed_on_non_tensor_payload() -> None:
    with pytest.raises(ValueError, match="non-tensor"):
        extract_state_dict({"net": {"bad": "value"}})


def test_component_froc_defaults_to_the_formal_181_point_logit_grid() -> None:
    args = parse_args(
        [
            "--checkpoint",
            str(Path("checkpoint_best_iou.pkl")),
            "--variant",
            "deterministic",
            "--dataset-dir",
            str(Path("dataset")),
            "--test-split-file",
            "img_idx/test_fixture.txt",
        ]
    )
    assert args.num_thresholds == 181
    assert args.threshold_space == "logit"
    assert args.min_logit == -20.0
    assert args.max_logit == 160.0


def test_checkpoint_identity_rejects_parameter_identical_variant_relabelling() -> None:
    payload = {
        "method_meta": {
            "mshnet_variant": "deterministic",
            "test_split_sha256": SPLIT_HASH,
            "evaluation_protocol": "official_train_test",
        }
    }
    with pytest.raises(ValueError, match="checkpoint variant mismatch"):
        validate_checkpoint_identity(
            payload,
            requested_variant="bcsf",
            checkpoint_sha256=CHECKPOINT_HASH,
            test_split_sha256=SPLIT_HASH,
        )


def test_modern_checkpoint_identity_validates_variant_and_split_metadata() -> None:
    payload = {
        "method_meta": {
            "mshnet_variant": "deterministic",
            "test_split_sha256": SPLIT_HASH,
            "evaluation_protocol": "official_train_test",
        }
    }
    result = validate_checkpoint_identity(
        payload,
        requested_variant="deterministic",
        checkpoint_sha256=CHECKPOINT_HASH,
        test_split_sha256=SPLIT_HASH,
    )
    assert result["mode"] == "method_meta"
    assert result["metadata_variant"] == "deterministic"
    assert result["test_split_sha256_verified"] is True
    assert result["warnings"] == []


def test_legacy_checkpoint_requires_explicit_artifact_and_split_bindings() -> None:
    payload = {"net": {"weight": torch.ones(1)}}
    with pytest.raises(ValueError, match="expected-checkpoint-sha256"):
        validate_checkpoint_identity(
            payload,
            requested_variant="deterministic",
            checkpoint_sha256=CHECKPOINT_HASH,
            test_split_sha256=SPLIT_HASH,
        )

    result = validate_checkpoint_identity(
        payload,
        requested_variant="deterministic",
        checkpoint_sha256=CHECKPOINT_HASH,
        test_split_sha256=SPLIT_HASH,
        expected_checkpoint_sha256=CHECKPOINT_HASH,
        expected_test_split_sha256=SPLIT_HASH,
    )
    assert result["mode"] == "legacy_explicit_binding"
    assert result["checkpoint_sha256_verified"] is True
    assert len(result["warnings"]) == 2


def test_expected_checkpoint_hash_mismatch_fails_closed() -> None:
    payload = {
        "method_meta": {
            "mshnet_variant": "deterministic",
            "test_split_sha256": SPLIT_HASH,
        }
    }
    with pytest.raises(ValueError, match="does not match"):
        validate_checkpoint_identity(
            payload,
            requested_variant="deterministic",
            checkpoint_sha256=CHECKPOINT_HASH,
            test_split_sha256=SPLIT_HASH,
            expected_checkpoint_sha256="c" * 64,
        )


def test_threshold_grid_metadata_records_exact_uniform_grid_identity() -> None:
    grid = threshold_grid_metadata([-20.0, -19.0, -18.0], "logit")
    assert grid["space"] == "logit"
    assert grid["construction"] == "linear_inclusive"
    assert grid["count"] == 3
    assert grid["minimum"] == -20.0
    assert grid["maximum"] == -18.0
    assert grid["uniform_step"] == 1.0
    assert len(grid["float64_le_sha256"]) == 64
