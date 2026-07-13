from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "tools" / "audit_trace_fixed_partition.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_trace_fixed_partition", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def observation(bits: str, sample_id: str):
    return audit.observation_from_binary(
        np.array([[int(item) for item in bits]], dtype=np.uint8),
        {"sample_id": sample_id},
    )


def test_exhaustive_one_by_three_grid_matches_every_set_partition() -> None:
    result = audit.exhaustive_small_grid_validation()

    assert result == {
        "grid": [1, 3],
        "connectivity": 8,
        "possible_binary_masks": 8,
        "mask_corpora_exhaustively_checked": 256,
        "set_partitions_per_corpus": 5,
        "theorem_bruteforce_mismatch_count": 0,
        "mismatches": [],
        "passed": True,
    }


def test_cross_sample_equality_chain_is_a_replayable_first_conflict() -> None:
    # S0 requires coordinate 0 != coordinate 2.  S1 makes 0 == 1, then S2
    # makes 1 == 2.  The third observation is therefore the first infeasible
    # prefix, with an explicit two-hyperedge equality path.
    result = audit.audit_fixed_partition(
        [
            observation("101", "S0"),
            observation("110", "S1"),
            observation("011", "S2"),
        ]
    )

    assert result["arbitrary_fixed_partition_feasible"] is False
    assert result["final_self_loop_constraint_count_with_multiplicity"] == 1
    certificate = result["first_conflict_certificate"]
    assert certificate["trigger"]["phase"] == "component_equality_hyperedge"
    assert certificate["trigger"]["trigger_observation_index"] == 2
    assert certificate["inequality_source_observation_index"] == 0
    assert certificate["component_a"]["metadata"]["sample_id"] == "S0"
    assert certificate["component_b"]["metadata"]["sample_id"] == "S0"
    assert len(
        certificate["component_a"][
            "support_flat_indices_uint32_le_sha256"
        ]
    ) == 64
    path = certificate["equality_path"]
    assert path["start_coordinate"] == [0, 0]
    assert path["end_coordinate"] == [0, 2]
    assert [
        item["metadata"]["sample_id"] for item in path["component_chain"]
    ] == ["S1", "S2"]
    assert path["uses_cross_observation_equalities"] is True
    assert certificate["verification_conditions"] == {
        "components_are_distinct_in_same_mask": True,
        "path_endpoints_match_inequality_anchors": True,
        "every_equality_transition_uses_one_observed_component": True,
    }


def test_union_find_pass_does_not_imply_independent_cell_set_product() -> None:
    result = audit.audit_fixed_partition([observation("101", "two_targets")])

    assert result["arbitrary_fixed_partition_feasible"] is True
    graph = result["quotient_spatial_8_adjacency"]
    assert graph["quotient_vertex_count"] == 3
    assert graph["quotient_spatial_edge_count"] == 2
    assert graph["quotient_spatial_connected_component_count"] == 1
    assert graph["unique_nonself_inequality_edge_count"] == 1
    assert graph["spatial_and_inequality_edge_overlap_count"] == 0
    assert graph[
        "merge_every_spatial_edge_conflicts_with_observed_inequalities"
    ] is True
    assert "hard-core" in graph["required_extension"]
    assert "not a valid exact set construction" in graph[
        "independent_product_warning"
    ]


def test_sufficiency_witness_is_found_by_bruteforce_and_loop_rejects() -> None:
    feasible = [observation("101", "different")]
    infeasible = [
        observation("101", "different"),
        observation("111", "equal"),
    ]

    assert audit.brute_force_partition_exists(3, feasible) is True
    assert audit.audit_fixed_partition(feasible)[
        "arbitrary_fixed_partition_feasible"
    ] is True
    assert audit.brute_force_partition_exists(3, infeasible) is False
    assert audit.audit_fixed_partition(infeasible)[
        "arbitrary_fixed_partition_feasible"
    ] is False
    direct_bridge = audit.audit_fixed_partition(infeasible)[
        "first_conflict_certificate"
    ]["equality_path"]
    assert direct_bridge["distinct_observation_count"] == 1
    assert direct_bridge["spans_multiple_equality_observations"] is False
    assert direct_bridge["uses_cross_observation_equalities"] is True


def _make_train_only_dataset(root: Path) -> Path:
    dataset = root / "TOY-TRAIN-ONLY"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    names = [f"sample_{index}" for index in range(5)]
    (dataset / "trainval.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )
    for index, name in enumerate(names):
        image = np.full((4, 4, 3), 20 + index, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[index % 4, index % 4] = 255
        Image.fromarray(image, mode="RGB").save(
            dataset / "images" / f"{name}.png"
        )
        Image.fromarray(mask, mode="L").save(
            dataset / "masks" / f"{name}.png"
        )
    # Intentionally create no official-test manifest or files.
    return dataset


def test_dataset_audit_runs_with_training_files_only(tmp_path: Path) -> None:
    dataset = _make_train_only_dataset(tmp_path)
    result = audit.audit_dataset(
        dataset,
        [17],
        split_seed=11,
        val_fraction=0.2,
        size=4,
        batch_size=2,
    )

    assert result["dataset"] == "TOY-TRAIN-ONLY"
    assert result["official_train_eval_resize"]["audit"][
        "observation_count"
    ] == 5
    augmented = result["actual_augmented_internal_fit_one_epoch"]
    assert augmented["per_seed_provenance"][0][
        "fit_count_before_drop_last"
    ] == 4
    assert augmented["per_seed_provenance"][0]["observed_count"] == 4
    assert augmented["aggregate_audit"]["observation_count"] == 4
    assert result["cumulative_official_train_plus_observed_augmentation"][
        "audit"
    ]["observation_count"] == 9


def test_build_report_states_finite_stream_and_test_nonuse(
    tmp_path: Path,
) -> None:
    dataset = _make_train_only_dataset(tmp_path)
    report = audit.build_report(
        [dataset],
        [19],
        split_seed=11,
        val_fraction=0.2,
        size=4,
        batch_size=2,
    )

    assert report["theorem"]["executable_exhaustive_check"]["passed"] is True
    guard = report["official_test_nonuse_guardrail"]
    assert guard["official_test_manifest_read"] is False
    assert guard["official_test_images_read"] is False
    assert guard["official_test_masks_read"] is False
    assert "No test path is resolved" in guard["statement"]
    assert "consume no RNG" in guard["rng_equivalence_note"]
    assert "not all 400 epochs" in report["finite_stream_scope"]
    assert "hard-core" in report["set_model_scope"]
    assert report["headline_decision"]["augmentation_expansion_required"] == (
        not report["headline_decision"][
            "every_dataset_seed_one_epoch_stream_infeasible"
        ]
    )
