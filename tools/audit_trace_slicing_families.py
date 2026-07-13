#!/usr/bin/env python3
"""Read-only Stage-0 audit of two deterministic slicing families for TRACE.

Only masks named by each dataset's *official training* manifest are opened.
Official-test manifests, identifiers, images, and masks are deliberately outside
the input surface of this program.  Masks are resized to 256 x 256 with nearest
neighbour interpolation and binarized with the repository convention
``ToTensor(mask)[0] > 0.5`` (for the benchmark's uint8 PNG masks this is exactly
``value >= 128``).  Foreground components use 8-connectivity.

The audit covers exactly two slicing families; it does not claim to rule out
every possible slicing construction:

1. uniform axis-aligned grids, including every integer tile height/width and
   every phase; and
2. recursive axis-aligned guillotine partitions.

The generated JSON is evidence about task geometry, not a model-selection or
hyperparameter-selection artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import glob
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from skimage import measure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "repro_runs"
    / "clean"
    / "trace_stage0_slicing_families_v1.json"
)
PROTOCOL = {
    "split": "official_train_only",
    "evaluation_height": 256,
    "evaluation_width": 256,
    "resize": "PIL nearest-neighbour",
    "mask_channel": 0,
    "mask_threshold": "> 0.5 after uint8/255 conversion (uint8 >= 128)",
    "foreground_connectivity": 8,
    "hole_background_connectivity": 4,
}


@dataclass(frozen=True, order=True)
class Box:
    """Inclusive integer bounding box of one 8-connected component."""

    top: int
    left: int
    bottom: int
    right: int

    def __post_init__(self) -> None:
        if min(self.top, self.left) < 0:
            raise ValueError("box coordinates must be non-negative")
        if self.bottom < self.top or self.right < self.left:
            raise ValueError("box must have non-empty inclusive extents")

    def as_list(self) -> list[int]:
        return [self.top, self.left, self.bottom, self.right]


@dataclass(frozen=True)
class ComponentGeometry:
    label: int
    area: int
    box: Box
    max_row_runs: int
    holes: int


@dataclass(frozen=True)
class SampleGeometry:
    sample_id: str
    foreground_pixels: int
    components: tuple[ComponentGeometry, ...]
    max_concurrent_components_per_row: int
    max_concurrent_runs_per_row: int

    @property
    def boxes(self) -> tuple[Box, ...]:
        return tuple(component.box for component in self.components)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names_sha256(names: Sequence[str]) -> str:
    return sha256_bytes(("\n".join(names) + "\n").encode("utf-8"))


def read_unique_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"empty official-training split: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate identifiers in split: {path}")
    return names


def resolve_official_train_split(dataset_dir: Path) -> Path:
    """Resolve only the repository's official-training manifest candidates."""

    dataset_dir = dataset_dir.resolve()
    candidates = [
        dataset_dir / "trainval.txt",
        dataset_dir / "img_idx" / f"train_{dataset_dir.name}.txt",
    ]
    candidates.extend(
        Path(item)
        for item in sorted(
            glob.glob(str(dataset_dir / "img_idx" / "train_*.txt"))
        )
    )
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "could not resolve official-training split; tried: "
        + ", ".join(str(item) for item in candidates)
    )


def uint8_pil_to_binary(mask: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Apply nearest resize and strict ``ToTensor()[0] > 0.5`` binarization."""

    resampling = getattr(Image, "Resampling", Image).NEAREST
    resized = mask.resize((size[1], size[0]), resampling)
    array = np.asarray(resized)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"mask must yield one 2-D channel, got {array.shape}")
    if array.dtype == np.uint8:
        return array > 127.5
    # torchvision ToTensor leaves non-byte numeric PIL arrays unscaled.
    return array.astype(np.float64, copy=False) > 0.5


def count_row_runs(row: np.ndarray) -> int:
    row = np.asarray(row, dtype=bool)
    if row.ndim != 1:
        raise ValueError("row must be one-dimensional")
    if row.size == 0:
        return 0
    return int(row[0]) + int(np.count_nonzero(row[1:] & ~row[:-1]))


def count_component_holes(component: np.ndarray) -> int:
    """Count bounded 4-connected background regions inside one component."""

    component = np.asarray(component, dtype=bool)
    if component.ndim != 2 or not component.any():
        raise ValueError("component must be a non-empty 2-D mask")
    rows, columns = np.nonzero(component)
    crop = component[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ]
    background_labels = measure.label(
        (~crop).astype(np.uint8), connectivity=1, background=0
    )
    if int(background_labels.max()) == 0:
        return 0
    boundary = np.concatenate(
        (
            background_labels[0, :],
            background_labels[-1, :],
            background_labels[:, 0],
            background_labels[:, -1],
        )
    )
    exterior = set(int(value) for value in np.unique(boundary) if value > 0)
    all_background = set(
        int(value) for value in np.unique(background_labels) if value > 0
    )
    return len(all_background - exterior)


def analyze_binary_mask(sample_id: str, binary: np.ndarray) -> SampleGeometry:
    """Extract exact 8-CC, row-run, concurrency, and hole geometry."""

    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("binary mask must be two-dimensional")
    labels = measure.label(
        binary.astype(np.uint8), connectivity=2, background=0
    )
    components: list[ComponentGeometry] = []
    row_component_counts = np.zeros(binary.shape[0], dtype=np.int64)
    for component_id in range(1, int(labels.max()) + 1):
        component = labels == component_id
        rows, columns = np.nonzero(component)
        row_component_counts += component.any(axis=1)
        components.append(
            ComponentGeometry(
                label=component_id,
                area=int(component.sum()),
                box=Box(
                    top=int(rows.min()),
                    left=int(columns.min()),
                    bottom=int(rows.max()),
                    right=int(columns.max()),
                ),
                max_row_runs=max(
                    count_row_runs(component[row])
                    for row in range(int(rows.min()), int(rows.max()) + 1)
                ),
                holes=count_component_holes(component),
            )
        )
    row_runs = np.array(
        [count_row_runs(binary[row]) for row in range(binary.shape[0])],
        dtype=np.int64,
    )
    return SampleGeometry(
        sample_id=sample_id,
        foreground_pixels=int(binary.sum()),
        components=tuple(components),
        max_concurrent_components_per_row=int(
            row_component_counts.max(initial=0)
        ),
        max_concurrent_runs_per_row=int(row_runs.max(initial=0)),
    )


def boxes_overlap_on_axis(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def overlapping_box_zero_certificate(
    samples: Sequence[Sequence[Box]],
) -> tuple[int, int, int] | None:
    """Find a pair that no non-splitting rectangular tiling can separate."""

    for sample_index, boxes in enumerate(samples):
        for first in range(len(boxes)):
            for second in range(first + 1, len(boxes)):
                a, b = boxes[first], boxes[second]
                if boxes_overlap_on_axis(a.top, a.bottom, b.top, b.bottom) and (
                    boxes_overlap_on_axis(a.left, a.right, b.left, b.right)
                ):
                    return sample_index, first, second
    return None


def _component_pairs(
    samples: Sequence[Sequence[Box]],
) -> list[tuple[int, int, int]]:
    return [
        (sample_index, first, second)
        for sample_index, boxes in enumerate(samples)
        for first in range(len(boxes))
        for second in range(first + 1, len(boxes))
    ]


def _axis_signature_counts(
    samples: Sequence[Sequence[Box]],
    axis_size: int,
    axis: str,
    pairs: Sequence[tuple[int, int, int]],
) -> Counter[int]:
    """Group every non-splitting 1-D grid by its same-bin pair signature."""

    if axis not in {"row", "column"}:
        raise ValueError("axis must be 'row' or 'column'")
    signatures: Counter[int] = Counter()
    for tile_extent in range(1, axis_size + 1):
        for phase in range(tile_extent):
            sample_bins: list[list[int]] = []
            valid = True
            for boxes in samples:
                bins: list[int] = []
                for box in boxes:
                    low, high = (
                        (box.top, box.bottom)
                        if axis == "row"
                        else (box.left, box.right)
                    )
                    low_bin = (low - phase) // tile_extent
                    if (high - phase) // tile_extent != low_bin:
                        valid = False
                        break
                    bins.append(low_bin)
                if not valid:
                    break
                sample_bins.append(bins)
            if not valid:
                continue
            same_bin_signature = 0
            for bit, (sample_index, first, second) in enumerate(pairs):
                if (
                    sample_bins[sample_index][first]
                    == sample_bins[sample_index][second]
                ):
                    same_bin_signature |= 1 << bit
            signatures[same_bin_signature] += 1
    return signatures


def count_uniform_grid_configurations(
    samples: Sequence[Sequence[Box]],
    height: int,
    width: int,
) -> dict:
    """Exactly count valid uniform-grid parameter tuples.

    A tuple ``(tile_height, row_phase, tile_width, column_phase)`` has
    ``1 <= tile_height <= height``, ``0 <= row_phase < tile_height`` and the
    analogous column constraints.  A pixel at ``(r, c)`` is assigned tile
    ``((r-row_phase)//tile_height, (c-column_phase)//tile_width)``.  Validity
    requires every component to occupy exactly one tile and every tile of
    every sample to contain at most one component.
    """

    if height <= 0 or width <= 0:
        raise ValueError("grid dimensions must be positive")
    for boxes in samples:
        for box in boxes:
            if box.bottom >= height or box.right >= width:
                raise ValueError("component box lies outside the grid")
    axis_row_candidates = height * (height + 1) // 2
    axis_column_candidates = width * (width + 1) // 2
    total = axis_row_candidates * axis_column_candidates
    certificate = overlapping_box_zero_certificate(samples)
    if certificate is not None:
        sample_index, first, second = certificate
        return {
            "candidate_parameter_tuples": total,
            "valid_parameter_tuples": 0,
            "counting_algorithm": "exact overlapping-bounding-box zero certificate",
            "zero_certificate": {
                "sample_index": sample_index,
                "component_indices_zero_based": [first, second],
                "boxes": [
                    samples[sample_index][first].as_list(),
                    samples[sample_index][second].as_list(),
                ],
                "proof": (
                    "The inclusive bounding intervals overlap on both axes. "
                    "Any rectangular tiles containing both components without "
                    "splitting them must therefore be the same tile."
                ),
            },
            "valid_non_splitting_row_axis_parameters": None,
            "valid_non_splitting_column_axis_parameters": None,
            "distinct_row_pair_signatures": None,
            "distinct_column_pair_signatures": None,
        }

    pairs = _component_pairs(samples)
    rows = _axis_signature_counts(samples, height, "row", pairs)
    columns = _axis_signature_counts(samples, width, "column", pairs)
    valid = sum(
        row_count * column_count
        for row_signature, row_count in rows.items()
        for column_signature, column_count in columns.items()
        if row_signature & column_signature == 0
    )
    return {
        "candidate_parameter_tuples": total,
        "valid_parameter_tuples": int(valid),
        "counting_algorithm": "exact grouped same-tile pair signatures",
        "zero_certificate": None,
        "valid_non_splitting_row_axis_parameters": int(sum(rows.values())),
        "valid_non_splitting_column_axis_parameters": int(
            sum(columns.values())
        ),
        "distinct_row_pair_signatures": len(rows),
        "distinct_column_pair_signatures": len(columns),
    }


def brute_force_uniform_grid_count(
    samples: Sequence[Sequence[Box]], height: int, width: int
) -> int:
    """Reference enumerator for small test grids; intentionally unoptimized."""

    valid_count = 0
    for tile_height in range(1, height + 1):
        for row_phase in range(tile_height):
            for tile_width in range(1, width + 1):
                for column_phase in range(tile_width):
                    valid = True
                    for boxes in samples:
                        occupied: set[tuple[int, int]] = set()
                        for box in boxes:
                            top_tile = (box.top - row_phase) // tile_height
                            bottom_tile = (
                                box.bottom - row_phase
                            ) // tile_height
                            left_tile = (
                                box.left - column_phase
                            ) // tile_width
                            right_tile = (
                                box.right - column_phase
                            ) // tile_width
                            if top_tile != bottom_tile or left_tile != right_tile:
                                valid = False
                                break
                            tile = (top_tile, left_tile)
                            if tile in occupied:
                                valid = False
                                break
                            occupied.add(tile)
                        if not valid:
                            break
                    valid_count += int(valid)
    return valid_count


def guillotine_partition_tree(
    boxes: Sequence[Box], height: int, width: int
) -> dict | None:
    """Return one exhaustive recursive-guillotine witness, or ``None``.

    Each internal cut is an integer boundary between adjacent rows or columns,
    extends across the current rectangular cell, intersects no component box,
    and leaves at least one component on each side.  Leaves contain zero or one
    component.  Candidate-boundary pruning is exact: every distinct component
    bipartition has a representative immediately after a component's bottom or
    right edge.
    """

    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    for box in boxes:
        if box.bottom >= height or box.right >= width:
            raise ValueError("component box lies outside the image")
    all_indices = tuple(range(len(boxes)))

    @lru_cache(maxsize=None)
    def solve(indices: tuple[int, ...]) -> dict | None:
        if len(indices) <= 1:
            return {"leaf_component_indices": list(indices)}
        for axis in ("horizontal", "vertical"):
            if axis == "horizontal":
                boundaries = sorted(
                    {
                        boxes[index].bottom + 1
                        for index in indices
                        if boxes[index].bottom + 1 < height
                    }
                )
            else:
                boundaries = sorted(
                    {
                        boxes[index].right + 1
                        for index in indices
                        if boxes[index].right + 1 < width
                    }
                )
            for boundary in boundaries:
                first: list[int] = []
                second: list[int] = []
                crossed = False
                for index in indices:
                    box = boxes[index]
                    low, high = (
                        (box.top, box.bottom)
                        if axis == "horizontal"
                        else (box.left, box.right)
                    )
                    if high < boundary:
                        first.append(index)
                    elif low >= boundary:
                        second.append(index)
                    else:
                        crossed = True
                        break
                if crossed or not first or not second:
                    continue
                first_tuple, second_tuple = tuple(first), tuple(second)
                first_tree = solve(first_tuple)
                if first_tree is None:
                    continue
                second_tree = solve(second_tuple)
                if second_tree is None:
                    continue
                return {
                    "axis": axis,
                    "boundary": boundary,
                    "first": first_tree,
                    "second": second_tree,
                }
        return None

    return solve(all_indices)


def is_guillotine_separable(
    boxes: Sequence[Box], height: int, width: int
) -> bool:
    return guillotine_partition_tree(boxes, height, width) is not None


def _histogram(values: Iterable[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def summarize_dataset_geometry(
    samples: Sequence[SampleGeometry], height: int, width: int
) -> dict:
    total_components = sum(len(sample.components) for sample in samples)
    nonempty_samples = sum(bool(sample.components) for sample in samples)
    largest_retained_pixels = sum(
        max((component.area for component in sample.components), default=0)
        for sample in samples
    )
    total_pixels = sum(sample.foreground_pixels for sample in samples)
    component_row_run_maximum = max(
        (
            component.max_row_runs
            for sample in samples
            for component in sample.components
        ),
        default=0,
    )
    sample_component_concurrency_maximum = max(
        (sample.max_concurrent_components_per_row for sample in samples),
        default=0,
    )
    sample_run_concurrency_maximum = max(
        (sample.max_concurrent_runs_per_row for sample in samples), default=0
    )
    total_holes = sum(
        component.holes
        for sample in samples
        for component in sample.components
    )
    components_with_holes = sum(
        component.holes > 0
        for sample in samples
        for component in sample.components
    )
    samples_with_holes = sum(
        any(component.holes > 0 for component in sample.components)
        for sample in samples
    )
    guillotine_flags = [
        is_guillotine_separable(sample.boxes, height, width)
        for sample in samples
    ]
    multi_indices = [
        index for index, sample in enumerate(samples) if len(sample.components) > 1
    ]
    guillotine_nonseparable_ids = [
        samples[index].sample_id
        for index in multi_indices
        if not guillotine_flags[index]
    ]
    uniform = count_uniform_grid_configurations(
        [sample.boxes for sample in samples], height, width
    )
    certificate = uniform.get("zero_certificate")
    if certificate is not None:
        sample = samples[int(certificate["sample_index"])]
        certificate["sample_id"] = sample.sample_id
        certificate["component_labels"] = [
            sample.components[index].label
            for index in certificate["component_indices_zero_based"]
        ]
    return {
        "samples": len(samples),
        "empty_samples": int(sum(len(sample.components) == 0 for sample in samples)),
        "single_component_samples": int(
            sum(len(sample.components) == 1 for sample in samples)
        ),
        "multi_component_samples": int(
            sum(len(sample.components) > 1 for sample in samples)
        ),
        "component_count": int(total_components),
        "component_count_per_sample_histogram": _histogram(
            len(sample.components) for sample in samples
        ),
        "largest_component_only": {
            "definition": (
                "Retain exactly one maximum-area 8-connected component in "
                "each non-empty sample; ties are resolved by the smallest "
                "row-major component label. Counts and pixels below are "
                "unavoidable for every such one-component retention rule."
            ),
            "samples_losing_components": int(
                sum(len(sample.components) > 1 for sample in samples)
            ),
            "retained_components": int(nonempty_samples),
            "discarded_components": int(total_components - nonempty_samples),
            "foreground_pixels": int(total_pixels),
            "retained_pixels": int(largest_retained_pixels),
            "discarded_pixels": int(total_pixels - largest_retained_pixels),
        },
        "row_geometry": {
            "component_row_run_definition": (
                "Number of maximal contiguous foreground intervals belonging "
                "to one component on one row."
            ),
            "component_max_row_runs": int(component_row_run_maximum),
            "component_max_row_runs_argmax_sample_ids": sorted(
                {
                    sample.sample_id
                    for sample in samples
                    if any(
                        component.max_row_runs == component_row_run_maximum
                        for component in sample.components
                    )
                }
            ),
            "sample_concurrent_component_definition": (
                "For one row, count distinct 8-connected components with at "
                "least one pixel on that row; take the per-sample then corpus maximum."
            ),
            "sample_max_concurrent_components": int(
                sample_component_concurrency_maximum
            ),
            "sample_max_concurrent_components_argmax_ids": sorted(
                sample.sample_id
                for sample in samples
                if sample.max_concurrent_components_per_row
                == sample_component_concurrency_maximum
            ),
            "sample_concurrent_run_definition": (
                "For one row, count all maximal foreground runs across all "
                "components; take the per-sample then corpus maximum."
            ),
            "sample_max_concurrent_runs": int(sample_run_concurrency_maximum),
            "sample_max_concurrent_runs_argmax_ids": sorted(
                sample.sample_id
                for sample in samples
                if sample.max_concurrent_runs_per_row
                == sample_run_concurrency_maximum
            ),
        },
        "holes": {
            "definition": (
                "A hole is a bounded 4-connected background component inside "
                "the tight bounding box of one 8-connected foreground component."
            ),
            "hole_count": int(total_holes),
            "components_with_holes": int(components_with_holes),
            "samples_with_holes": int(samples_with_holes),
            "max_holes_per_component": int(
                max(
                    (
                        component.holes
                        for sample in samples
                        for component in sample.components
                    ),
                    default=0,
                )
            ),
            "sample_ids": sorted(
                sample.sample_id
                for sample in samples
                if any(component.holes > 0 for component in sample.components)
            ),
        },
        "uniform_grid_family": {
            "scope_warning": (
                "This result covers only the precisely defined uniform-grid "
                "family, not every deterministic slicing construction."
            ),
            "definition": (
                "Enumerate tile height h=1..H and row phase 0..h-1, tile "
                "width w=1..W and column phase 0..w-1. Tile indices are "
                "floor((row-phase)/h), floor((column-phase)/w). A configuration "
                "is valid iff no 8-CC crosses a tile boundary and no tile in "
                "any training sample contains more than one 8-CC."
            ),
            **uniform,
        },
        "recursive_guillotine_family": {
            "scope_warning": (
                "This result covers only recursive full axis-aligned cuts, not "
                "every deterministic slicing construction."
            ),
            "definition": (
                "Recursively cut the current rectangular cell at an integer "
                "row or column boundary. Each cut spans the full current cell, "
                "intersects no component bounding box, and sends at least one "
                "whole component to each child. A sample is separable iff this "
                "recursion reaches leaves containing at most one component."
            ),
            "all_samples_separable": int(sum(guillotine_flags)),
            "multi_component_samples": len(multi_indices),
            "separable_multi_component_samples": int(
                sum(guillotine_flags[index] for index in multi_indices)
            ),
            "nonseparable_multi_component_samples": len(
                guillotine_nonseparable_ids
            ),
            "nonseparable_sample_ids": guillotine_nonseparable_ids,
        },
    }


def audit_dataset(
    dataset_dir: Path, height: int = 256, width: int = 256
) -> dict:
    dataset_dir = dataset_dir.resolve()
    train_split = resolve_official_train_split(dataset_dir)
    names = read_unique_names(train_split)
    mask_dir = dataset_dir / "masks"
    corpus_digest = hashlib.sha256()
    samples: list[SampleGeometry] = []
    for name in names:
        mask_path = mask_dir / f"{name}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        payload = mask_path.read_bytes()
        mask_digest = sha256_bytes(payload)
        corpus_digest.update(
            f"{name}\0{mask_digest}\0{len(payload)}\n".encode("utf-8")
        )
        with Image.open(io.BytesIO(payload)) as mask:
            binary = uint8_pil_to_binary(mask, (height, width))
        samples.append(analyze_binary_mask(name, binary))
    return {
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "input_provenance": {
            "official_train_split": str(train_split),
            "official_train_split_file_sha256": sha256_file(train_split),
            "official_train_identifier_sequence_sha256": names_sha256(names),
            "official_train_identifier_count": len(names),
            "train_mask_corpus_sha256": corpus_digest.hexdigest(),
            "train_mask_corpus_hash_definition": (
                "SHA256 over source-order UTF-8 records "
                "'sample_id\\0mask_file_sha256\\0mask_size_bytes\\n'."
            ),
            "train_masks_opened": len(names),
            "train_images_opened": 0,
        },
        "geometry": summarize_dataset_geometry(samples, height, width),
    }


def build_report(
    dataset_dirs: Sequence[Path], height: int = 256, width: int = 256
) -> dict:
    return {
        "schema_version": 1,
        "audit": "TRACE Stage-0 official-train slicing-family audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            **PROTOCOL,
            "evaluation_height": height,
            "evaluation_width": width,
        },
        "family_scope": {
            "covered": [
                "all integer-phase uniform axis-aligned grids",
                "axis-aligned recursive guillotine partitions",
            ],
            "not_claimed": "all possible deterministic slicing families",
        },
        "official_test_guardrail": {
            "policy": (
                "Official-test manifests, identifiers, images, and masks are "
                "not inputs to this audit and must not affect any statistic."
            ),
            "official_test_split_read": False,
            "official_test_identifiers_read": False,
            "official_test_images_opened": 0,
            "official_test_masks_opened": 0,
            "used_for_family_selection": False,
            "used_for_hyperparameter_selection": False,
            "used_for_model_selection": False,
        },
        "datasets": [
            audit_dataset(path, height=height, width=width)
            for path in dataset_dirs
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help=(
            "Comma-separated dataset names below PROJECT_ROOT/datasets or "
            "explicit dataset directories."
        ),
    )
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_dataset_arguments(text: str) -> list[Path]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one dataset is required")
    paths = []
    for value in values:
        candidate = Path(value)
        if not candidate.is_absolute() and len(candidate.parts) == 1:
            candidate = PROJECT_ROOT / "datasets" / candidate
        elif not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        if not candidate.is_dir():
            raise FileNotFoundError(candidate)
        paths.append(candidate.resolve())
    return paths


def main() -> None:
    args = parse_args()
    report = build_report(
        resolve_dataset_arguments(args.datasets),
        height=args.height,
        width=args.width,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
