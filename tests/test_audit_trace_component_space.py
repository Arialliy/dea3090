from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_trace_component_space import (
    build_report,
    deterministic_internal_split,
    geometry_from_binary,
    is_single_row_run_component,
    pil_mask_to_binary,
    summarize_geometries,
)


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _make_fixture(root: Path) -> Path:
    dataset = root / "fixture"
    (dataset / "img_idx").mkdir(parents=True)
    train_names = ["empty", "diagonal", "u_shape", "two"]
    test_names = ["test_diagonal"]
    (dataset / "img_idx" / "train_fixture.txt").write_text(
        "\n".join(train_names) + "\n", encoding="utf-8"
    )
    (dataset / "img_idx" / "test_fixture.txt").write_text(
        "\n".join(test_names) + "\n", encoding="utf-8"
    )

    masks: dict[str, np.ndarray] = {}
    masks["empty"] = np.zeros((8, 8), dtype=np.uint8)
    diagonal = np.zeros((8, 8), dtype=np.uint8)
    diagonal[2, 2] = 255
    diagonal[3, 3] = 255
    masks["diagonal"] = diagonal
    masks["test_diagonal"] = diagonal
    u_shape = np.zeros((8, 8), dtype=np.uint8)
    u_shape[1:6, 1] = 255
    u_shape[1:6, 5] = 255
    u_shape[5, 1:6] = 255
    masks["u_shape"] = u_shape
    two = np.zeros((8, 8), dtype=np.uint8)
    two[1:3, 1:3] = 255
    two[5:7, 5:7] = 255
    masks["two"] = two

    for name, mask in masks.items():
        _write_png(dataset / "masks" / f"{name}.png", mask)
        image = np.repeat(mask[:, :, None], 3, axis=2)
        _write_png(dataset / "images" / f"{name}.png", image)
    return dataset


def test_geometry_uses_strict_threshold_and_eight_connectivity() -> None:
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[0, 0] = 0.5  # Strict ``> 0.5`` is background.
    mask[1, 1] = 1.0
    mask[2, 2] = 1.0  # Diagonal contact is one 8-connected component.

    geometry = geometry_from_binary(mask > 0.5)

    assert geometry.components == 1
    assert geometry.foreground_pixels == 2
    assert geometry.non_single_row_run_components == 0


def test_pil_binarization_distinguishes_uint8_127_and_128() -> None:
    mask = Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8))

    binary = pil_mask_to_binary(mask, threshold=0.5)

    np.testing.assert_array_equal(binary, np.array([[0, 0, 1, 1]], dtype=np.uint8))


def test_single_row_run_rejects_a_u_shape_but_accepts_diagonal_path() -> None:
    u_shape = np.zeros((6, 7), dtype=bool)
    u_shape[1:5, 1] = True
    u_shape[1:5, 5] = True
    u_shape[4, 1:6] = True
    diagonal = np.eye(5, dtype=bool)

    assert not is_single_row_run_component(u_shape)
    assert is_single_row_run_component(diagonal)


def test_summary_separates_empty_single_multi_and_non_ssr() -> None:
    empty = geometry_from_binary(np.zeros((6, 6), dtype=bool))
    single = geometry_from_binary(np.eye(6, dtype=bool))
    multi_mask = np.zeros((6, 6), dtype=bool)
    multi_mask[0, 0] = True
    multi_mask[5, 5] = True
    multi = geometry_from_binary(multi_mask)
    non_ssr_mask = np.zeros((6, 7), dtype=bool)
    non_ssr_mask[1:5, 1] = True
    non_ssr_mask[1:5, 5] = True
    non_ssr_mask[4, 1:6] = True
    non_ssr = geometry_from_binary(non_ssr_mask)

    summary = summarize_geometries([empty, single, multi, non_ssr])

    assert summary["empty_samples"] == 1
    assert summary["single_component_samples"] == 2
    assert summary["multi_component_samples"] == 1
    assert summary["component_count"] == 4
    assert summary["max_components_per_sample"] == 2
    assert summary["non_single_row_run_components"] == 1
    assert summary["samples_with_non_single_row_run_component"] == 1
    assert summary[
        "samples_outside_empty_or_single_single_row_run_component"
    ] == 2


def test_internal_split_is_seeded_hash_ranking_and_preserves_source_order() -> None:
    names = ["z", "a", "m", "b", "q"]
    fit_a, val_a = deterministic_internal_split(names, 20260711, 0.4)
    fit_b, val_b = deterministic_internal_split(names, 20260711, 0.4)

    assert (fit_a, val_a) == (fit_b, val_b)
    assert len(fit_a) == 3
    assert len(val_a) == 2
    assert fit_a == [name for name in names if name in set(fit_a)]
    assert val_a == [name for name in names if name in set(val_a)]
    assert set(fit_a).isdisjoint(val_a)
    assert set(fit_a).union(val_a) == set(names)


def test_fixture_audit_runs_actual_data_loader_and_excludes_test_from_selection(
    tmp_path: Path,
) -> None:
    dataset = _make_fixture(tmp_path)

    report = build_report(
        [dataset],
        seeds=[11, 12],
        split_seed=7,
        val_fraction=0.25,
        base_size=8,
        crop_size=8,
        batch_size=2,
        num_workers=0,
    )
    result = report["datasets"][0]

    raw_train = result["official_splits"]["train"]["raw"]
    assert raw_train["sample_count"] == 4
    assert raw_train["empty_samples"] == 1
    assert raw_train["single_component_samples"] == 2
    assert raw_train["multi_component_samples"] == 1
    assert raw_train["non_single_row_run_components"] == 1
    holdout = result["internal_holdout"]
    assert holdout["fit_count"] == 3
    assert holdout["validation_count"] == 1
    stream = result["actual_augmented_internal_fit_epoch"]
    assert stream["loader_contract"] == {
        "implementation": "utils.data.IRSTD_Dataset",
        "shuffle": True,
        "batch_size": 2,
        "drop_last": True,
        "num_workers": 0,
        "pin_memory": True,
        "persistent_workers": False,
        "generator_seed": "the corresponding model seed",
        "worker_init_fn": "main.seed_worker-equivalent",
        "base_size": 8,
        "crop_size": 8,
    }
    assert holdout["validation_raw"]["sample_count"] == 1
    assert holdout["validation_nearest_resize_8"]["sample_count"] == 1
    assert stream["aggregate"]["sample_count"] == 4
    assert [item["seed"] for item in stream["per_seed"]] == [11, 12]
    guardrail = report["test_usage_guardrail"]
    assert guardrail["purpose"] == "task-definition validation only"
    assert guardrail["used_for_component_family_selection"] is False
    assert guardrail["used_for_hyperparameter_selection"] is False
    assert guardrail["used_for_model_selection"] is False
