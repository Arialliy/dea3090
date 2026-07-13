from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from tools.audit_trace_slicing_families import (
    Box,
    analyze_binary_mask,
    audit_dataset,
    brute_force_uniform_grid_count,
    build_report,
    count_component_holes,
    count_uniform_grid_configurations,
    is_guillotine_separable,
    uint8_pil_to_binary,
)


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _make_dataset(root: Path) -> Path:
    dataset = root / "fixture"
    (dataset / "img_idx").mkdir(parents=True)
    names = ["ring_and_point", "diagonal", "empty"]
    (dataset / "img_idx" / "train_fixture.txt").write_text(
        "\n".join(names) + "\n", encoding="utf-8"
    )
    # Deliberately invalid official-test content: the audit must never read it.
    (dataset / "img_idx" / "test_fixture.txt").write_bytes(b"\xff\xfe")

    ring = np.zeros((8, 8), dtype=np.uint8)
    ring[1, 1:6] = 255
    ring[5, 1:6] = 255
    ring[1:6, 1] = 255
    ring[1:6, 5] = 255
    ring[3, 7] = 255
    diagonal = np.zeros((8, 8), dtype=np.uint8)
    diagonal[2, 2] = 255
    diagonal[3, 3] = 255
    empty = np.zeros((8, 8), dtype=np.uint8)
    for name, mask in {
        "ring_and_point": ring,
        "diagonal": diagonal,
        "empty": empty,
    }.items():
        _write_mask(dataset / "masks" / f"{name}.png", mask)
    return dataset


def _reference_guillotine(
    boxes: tuple[Box, ...], height: int, width: int
) -> bool:
    """Independent small-grid reference: try every coordinate cut."""

    if len(boxes) <= 1:
        return True
    for axis_size, axis in ((height, "row"), (width, "column")):
        for boundary in range(1, axis_size):
            first = []
            second = []
            valid = True
            for box in boxes:
                low, high = (
                    (box.top, box.bottom)
                    if axis == "row"
                    else (box.left, box.right)
                )
                if high < boundary:
                    first.append(box)
                elif low >= boundary:
                    second.append(box)
                else:
                    valid = False
                    break
            if (
                valid
                and first
                and second
                and _reference_guillotine(tuple(first), height, width)
                and _reference_guillotine(tuple(second), height, width)
            ):
                return True
    return False


def test_threshold_eight_connectivity_runs_concurrency_and_hole() -> None:
    values = Image.fromarray(
        np.array([[0, 127, 128, 255]], dtype=np.uint8)
    )
    binary_values = uint8_pil_to_binary(values, (1, 4))
    np.testing.assert_array_equal(
        binary_values, np.array([[False, False, True, True]])
    )

    mask = np.zeros((8, 8), dtype=bool)
    mask[1, 1:6] = True
    mask[5, 1:6] = True
    mask[1:6, 1] = True
    mask[1:6, 5] = True
    mask[3, 7] = True
    geometry = analyze_binary_mask("sample", mask)

    assert len(geometry.components) == 2
    assert geometry.components[0].area == 16
    assert geometry.components[0].max_row_runs == 2
    assert geometry.components[0].holes == 1
    assert geometry.max_concurrent_components_per_row == 2
    assert geometry.max_concurrent_runs_per_row == 3
    assert count_component_holes(mask[:, :6]) == 1


def test_uniform_grid_exact_counter_matches_small_grid_brute_force() -> None:
    fixtures = [
        [],
        [[Box(1, 1, 1, 1)]],
        [[Box(0, 0, 0, 0), Box(2, 2, 2, 2)]],
        [[Box(0, 0, 1, 0)], [Box(2, 1, 2, 1)]],
        # Overlap on both bounding-box axes is an exact zero certificate.
        [[Box(0, 0, 2, 2), Box(1, 1, 3, 3)]],
    ]
    for samples in fixtures:
        fast = count_uniform_grid_configurations(samples, 4, 4)
        brute = brute_force_uniform_grid_count(samples, 4, 4)
        assert fast["valid_parameter_tuples"] == brute
    assert count_uniform_grid_configurations(fixtures[-1], 4, 4)[
        "zero_certificate"
    ] is not None


def test_guillotine_known_windmill_and_exhaustive_small_fixture_subsets() -> None:
    # Four disjoint rectangles in a pinwheel: every full first cut is blocked.
    windmill = (
        Box(0, 1, 1, 5),
        Box(1, 7, 5, 8),
        Box(7, 3, 8, 8),
        Box(3, 0, 8, 1),
    )
    assert not is_guillotine_separable(windmill, 9, 9)
    assert _reference_guillotine(windmill, 9, 9) is False
    assert is_guillotine_separable(
        (Box(0, 0, 1, 1), Box(3, 0, 4, 1), Box(3, 3, 4, 4)), 5, 5
    )

    library = (
        Box(0, 0, 0, 1),
        Box(0, 3, 2, 3),
        Box(2, 0, 3, 0),
        Box(3, 2, 3, 3),
        Box(1, 1, 1, 2),
        Box(2, 2, 2, 2),
    )
    # Exhaust all 64 subsets; the implementation prunes candidate coordinates,
    # while the reference enumerates every row/column boundary at every node.
    for subset_size in range(len(library) + 1):
        for subset in combinations(library, subset_size):
            assert is_guillotine_separable(subset, 4, 4) == (
                _reference_guillotine(subset, 4, 4)
            )


def test_official_train_only_fixture_audit_and_hashes(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    result = audit_dataset(dataset, height=8, width=8)
    geometry = result["geometry"]

    assert result["input_provenance"]["official_train_identifier_count"] == 3
    assert result["input_provenance"]["train_masks_opened"] == 3
    assert result["input_provenance"]["train_images_opened"] == 0
    assert len(result["input_provenance"]["train_mask_corpus_sha256"]) == 64
    assert geometry["component_count"] == 3
    assert geometry["largest_component_only"]["discarded_components"] == 1
    assert geometry["largest_component_only"]["discarded_pixels"] == 1
    assert geometry["row_geometry"]["component_max_row_runs"] == 2
    assert geometry["row_geometry"]["sample_max_concurrent_components"] == 2
    assert geometry["row_geometry"]["sample_max_concurrent_runs"] == 3
    assert geometry["holes"]["hole_count"] == 1
    assert geometry["recursive_guillotine_family"][
        "nonseparable_multi_component_samples"
    ] == 0

    report = build_report([dataset], height=8, width=8)
    guardrail = report["official_test_guardrail"]
    assert guardrail["official_test_split_read"] is False
    assert guardrail["official_test_identifiers_read"] is False
    assert guardrail["official_test_images_opened"] == 0
    assert guardrail["official_test_masks_opened"] == 0
    assert report["family_scope"]["not_claimed"] == (
        "all possible deterministic slicing families"
    )


def test_train_mask_corpus_hash_changes_only_with_train_input(tmp_path: Path) -> None:
    dataset = _make_dataset(tmp_path)
    first = audit_dataset(dataset, height=8, width=8)
    # Changing the invalid official-test manifest cannot influence this tool.
    (dataset / "img_idx" / "test_fixture.txt").write_text(
        "anything\n", encoding="utf-8"
    )
    second = audit_dataset(dataset, height=8, width=8)
    assert first == second

    mask_path = dataset / "masks" / "empty.png"
    changed = np.zeros((8, 8), dtype=np.uint8)
    changed[0, 0] = 255
    _write_mask(mask_path, changed)
    third = audit_dataset(dataset, height=8, width=8)
    assert first["input_provenance"]["train_mask_corpus_sha256"] != third[
        "input_provenance"
    ]["train_mask_corpus_sha256"]
