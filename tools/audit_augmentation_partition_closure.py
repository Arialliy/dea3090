#!/usr/bin/env python3
"""Certify a fixed-coordinate partition contradiction under train augmentation.

This audit is deliberately narrower and stronger than a finite-seed stream
audit.  It uses only official-training masks and a small, explicit subset of
the random augmentation *support*.  Crop positions are handled symbolically,
so the conclusion does not depend on whether a particular seed happened to
draw a witness.

Let ``q`` assign every output coordinate to one fixed, non-overlapping slice.
An exact single-component slicing protocol needs both:

1. ``q`` is constant on every observed connected component (no target split);
2. distinct components in one observed mask receive distinct values of ``q``
   (at most one real component per slice).

For each frozen equality witness below, legal scale/crop/flip transforms make
every horizontal and vertical edge of the 256 x 256 output grid occur as an
adjacent foreground pair.  Condition 1 therefore makes ``q`` constant on the
whole grid.  A second legal transform contains at least two components, which
contradicts condition 2.

The theorem covers arbitrary shapes and arbitrary numbers of cells in a fixed
disjoint coordinate partition.  It does not cover overlapping crop families,
sample-adaptive assignment, or a protocol that is allowed to duplicate/drop
targets; those are different task definitions and need separate audits.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from skimage import measure


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CROP_SIZE = 256
BASE_SIZE = 256
LONG_SIZE_SUPPORT = (int(BASE_SIZE * 0.5), int(BASE_SIZE * 2.0))

# These IDs were selected only from each official training manifest.  The
# audit fails closed if a witness is absent from that manifest.
WITNESSES = {
    "NUAA-SIRST": {
        "equality_names": ("Misc_11", "Misc_119"),
        "equality_flips": (False, True),
        "inequality_name": "Misc_119",
    },
    "NUDT-SIRST": {
        "equality_names": ("000848",),
        "equality_flips": (False,),
        "inequality_name": "000848",
    },
    "IRSTD-1K": {
        "equality_names": ("XDU730",),
        "equality_flips": (False,),
        "inequality_name": "XDU907",
    },
}

SPLIT_SEED = 20260711
VAL_FRACTION = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_official_train_names(dataset_dir: Path) -> tuple[Path, list[str]]:
    """Resolve only the official train manifest; never fall back to test."""

    dataset_name = dataset_dir.name
    candidates = (
        dataset_dir / "trainval.txt",
        dataset_dir / "img_idx" / f"train_{dataset_name}.txt",
    )
    manifest = next((path for path in candidates if path.is_file()), None)
    if manifest is None:
        raise FileNotFoundError(
            "official train manifest not found; tried: "
            + ", ".join(str(path) for path in candidates)
        )
    names = [
        line.strip()
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"empty train manifest: {manifest}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate names in train manifest: {manifest}")
    return manifest.resolve(), names


def deterministic_internal_split(
    names: Sequence[str],
    *,
    split_seed: int = SPLIT_SEED,
    val_fraction: float = VAL_FRACTION,
) -> tuple[list[str], list[str]]:
    """Reproduce the repository's fixed fit/validation split exactly."""

    ranked = sorted(
        names,
        key=lambda name: hashlib.sha256(
            (f"{split_seed}\0{name}").encode("utf-8")
        ).digest(),
    )
    number_validation = max(
        1,
        min(len(names) - 1, int(round(len(names) * val_fraction))),
    )
    validation_set = set(ranked[:number_validation])
    fit_names = [name for name in names if name not in validation_set]
    validation_names = [name for name in names if name in validation_set]
    return fit_names, validation_names


def names_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def pil_mask_to_binary(mask: Image.Image) -> np.ndarray:
    """Match ``transforms.ToTensor()(mask)[0] > 0.5`` for 8-bit masks."""

    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2 or array.dtype != np.uint8:
        raise ValueError(
            f"expected a two-dimensional uint8 mask, got {array.shape}/{array.dtype}"
        )
    return array.astype(np.float32) / 255.0 > 0.5


def repository_resize_shape(size: tuple[int, int], long_size: int) -> tuple[int, int]:
    """Mirror ``IRSTD_Dataset._sync_transform`` integer resize semantics."""

    width, height = size
    if height > width:
        out_height = int(long_size)
        out_width = int(width * long_size / height + 0.5)
    else:
        out_width = int(long_size)
        out_height = int(height * long_size / width + 0.5)
    return out_width, out_height


def resize_and_pad_mask(
    mask: Image.Image,
    *,
    long_size: int,
    flip: bool,
    crop_size: int = CROP_SIZE,
) -> Image.Image:
    """Apply the mask-relevant prefix of the repository train transform."""

    if not LONG_SIZE_SUPPORT[0] <= long_size <= LONG_SIZE_SUPPORT[1]:
        raise ValueError(
            f"long_size={long_size} is outside repository support {LONG_SIZE_SUPPORT}"
        )
    transformed = mask.transpose(Image.FLIP_LEFT_RIGHT) if flip else mask.copy()
    out_width, out_height = repository_resize_shape(transformed.size, long_size)
    transformed = transformed.resize((out_width, out_height), Image.NEAREST)
    pad_width = max(crop_size - out_width, 0)
    pad_height = max(crop_size - out_height, 0)
    if pad_width or pad_height:
        transformed = ImageOps.expand(
            transformed,
            border=(0, 0, pad_width, pad_height),
            fill=0,
        )
    return transformed


def transform_mask(
    mask: Image.Image,
    *,
    long_size: int,
    flip: bool,
    crop_x: int,
    crop_y: int,
    crop_size: int = CROP_SIZE,
) -> np.ndarray:
    """Return one exact binary mask in the augmentation support."""

    transformed = resize_and_pad_mask(
        mask, long_size=long_size, flip=flip, crop_size=crop_size
    )
    max_crop_x = transformed.width - crop_size
    max_crop_y = transformed.height - crop_size
    if not 0 <= crop_x <= max_crop_x or not 0 <= crop_y <= max_crop_y:
        raise ValueError(
            f"invalid crop ({crop_x},{crop_y}); support is "
            f"x=0..{max_crop_x}, y=0..{max_crop_y}"
        )
    transformed = transformed.crop(
        (crop_x, crop_y, crop_x + crop_size, crop_y + crop_size)
    )
    return pil_mask_to_binary(transformed)


def _edge_rectangles(
    binary: np.ndarray,
    *,
    flip: bool,
    crop_size: int = CROP_SIZE,
) -> dict[str, list[dict]]:
    """Symbolically map foreground neighbor pairs over every legal crop.

    A horizontal foreground pair starting at resized coordinate ``(x,y)`` can
    appear at every output horizontal edge ``(x-cx,y-cy)`` for legal crop
    offsets.  Its reachable output positions are therefore one axis-aligned
    rectangle.  The vertical case is identical with the edge axes exchanged.
    """

    height, width = binary.shape
    if height < crop_size or width < crop_size:
        raise ValueError("binary mask must already be padded to crop size")
    crop_x_max = width - crop_size
    crop_y_max = height - crop_size
    rectangles: dict[str, list[dict]] = {"horizontal": [], "vertical": []}

    ys, xs = np.nonzero(binary[:, :-1] & binary[:, 1:])
    for y_value, x_value in zip(ys.tolist(), xs.tolist()):
        x0 = max(0, x_value - crop_x_max)
        x1 = min(crop_size - 2, x_value)
        y0 = max(0, y_value - crop_y_max)
        y1 = min(crop_size - 1, y_value)
        if x0 <= x1 and y0 <= y1:
            rectangles["horizontal"].append(
                {
                    "flip": bool(flip),
                    "resized_pair": [
                        [x_value, y_value],
                        [x_value + 1, y_value],
                    ],
                    "reachable_edge_rectangle_xyxy": [x0, y0, x1, y1],
                }
            )

    ys, xs = np.nonzero(binary[:-1, :] & binary[1:, :])
    for y_value, x_value in zip(ys.tolist(), xs.tolist()):
        x0 = max(0, x_value - crop_x_max)
        x1 = min(crop_size - 1, x_value)
        y0 = max(0, y_value - crop_y_max)
        y1 = min(crop_size - 2, y_value)
        if x0 <= x1 and y0 <= y1:
            rectangles["vertical"].append(
                {
                    "flip": bool(flip),
                    "resized_pair": [
                        [x_value, y_value],
                        [x_value, y_value + 1],
                    ],
                    "reachable_edge_rectangle_xyxy": [x0, y0, x1, y1],
                }
            )
    return rectangles


def _coverage_shape(orientation: str, crop_size: int) -> tuple[int, int]:
    if orientation == "horizontal":
        return crop_size, crop_size - 1
    if orientation == "vertical":
        return crop_size - 1, crop_size
    raise ValueError(f"unknown orientation: {orientation}")


def coverage_from_rectangles(
    rectangles: Iterable[dict], *, orientation: str, crop_size: int = CROP_SIZE
) -> np.ndarray:
    """Materialize the exact union of symbolic reachable-edge rectangles."""

    shape = _coverage_shape(orientation, crop_size)
    difference = np.zeros((shape[0] + 1, shape[1] + 1), dtype=np.int64)
    for rectangle in rectangles:
        x0, y0, x1, y1 = rectangle["reachable_edge_rectangle_xyxy"]
        if not (0 <= y0 <= y1 < shape[0] and 0 <= x0 <= x1 < shape[1]):
            raise ValueError(f"invalid {orientation} edge rectangle: {rectangle}")
        difference[y0, x0] += 1
        difference[y1 + 1, x0] -= 1
        difference[y0, x1 + 1] -= 1
        difference[y1 + 1, x1 + 1] += 1
    return difference.cumsum(axis=0).cumsum(axis=1)[: shape[0], : shape[1]] > 0


def greedy_rectangle_certificate(
    rectangles: Sequence[dict], *, orientation: str, crop_size: int = CROP_SIZE
) -> list[dict]:
    """Return a deterministic small cover whose union is independently checkable."""

    coverage = np.zeros(_coverage_shape(orientation, crop_size), dtype=bool)
    available = list(range(len(rectangles)))
    selected: list[dict] = []
    while not bool(coverage.all()):
        best_index = -1
        best_gain = -1
        for index in available:
            x0, y0, x1, y1 = rectangles[index][
                "reachable_edge_rectangle_xyxy"
            ]
            gain = int((~coverage[y0 : y1 + 1, x0 : x1 + 1]).sum())
            if gain > best_gain:
                best_gain = gain
                best_index = index
        if best_index < 0 or best_gain <= 0:
            raise AssertionError(
                f"reachable {orientation} edge union does not cover the grid"
            )
        item = dict(rectangles[best_index])
        item["new_edges_covered"] = best_gain
        selected.append(item)
        x0, y0, x1, y1 = item["reachable_edge_rectangle_xyxy"]
        coverage[y0 : y1 + 1, x0 : x1 + 1] = True
        available.remove(best_index)
    return selected


def component_summary(binary: np.ndarray) -> dict:
    labels = measure.label(binary.astype(np.uint8), connectivity=2, background=0)
    components = []
    for component_id in range(1, int(labels.max()) + 1):
        ys, xs = np.nonzero(labels == component_id)
        components.append(
            {
                "component_id": component_id,
                "pixels": int(xs.size),
                "bbox_xyxy": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ],
            }
        )
    return {
        "connectivity": 8,
        "component_count": int(labels.max()),
        "foreground_pixels": int(binary.sum()),
        "components": components,
    }


def _load_train_witness(
    dataset_dir: Path, name: str, eligible_fit_names: Sequence[str]
) -> tuple[Path, Path, Image.Image]:
    if name not in set(eligible_fit_names):
        raise AssertionError(
            f"frozen witness {name!r} is not in the deterministic training fit set"
        )
    path = (dataset_dir / "masks" / f"{name}.png").resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    image_path = (dataset_dir / "images" / f"{name}.png").resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(path) as source:
        mask = source.copy()
    with Image.open(image_path) as source_image:
        image_size = source_image.size
    # The repository derives resize geometry from ``img.size`` and applies
    # that shape to the mask.  Using ``mask.size`` below is equivalent only
    # after this explicit per-witness equality check.
    if image_size != mask.size:
        raise AssertionError(
            f"image/mask geometry mismatch for {name}: {image_size} != {mask.size}"
        )
    # Every frozen witness is binary uint8.  Fail if corpus semantics drift.
    unique_values = np.unique(np.asarray(mask)).tolist()
    if unique_values != [0, 255]:
        raise AssertionError(
            f"witness {path} is no longer binary uint8: {unique_values}"
        )
    return path, image_path, mask


def audit_dataset(dataset_dir: Path) -> dict:
    dataset_name = dataset_dir.name
    if dataset_name not in WITNESSES:
        raise ValueError(f"no frozen witness configuration for {dataset_name}")
    configuration = WITNESSES[dataset_name]
    train_manifest, train_names = read_official_train_names(dataset_dir)
    fit_names, validation_names = deterministic_internal_split(train_names)

    all_rectangles: dict[str, list[dict]] = {
        "horizontal": [],
        "vertical": [],
    }
    equality_sources = []
    for equality_name in configuration["equality_names"]:
        equality_path, equality_image_path, equality_mask = _load_train_witness(
            dataset_dir, equality_name, fit_names
        )
        source = {
            "name": equality_name,
            "mask_path": str(equality_path),
            "mask_sha256": sha256_file(equality_path),
            "image_path": str(equality_image_path),
            "image_sha256": sha256_file(equality_image_path),
            "repository_resize_geometry_source": "image.size",
            "image_mask_size_equal": True,
            "original_image_size_wh": list(equality_mask.size),
            "original_size_wh": list(equality_mask.size),
            "transforms": [],
        }
        for flip in configuration["equality_flips"]:
            resized = resize_and_pad_mask(
                equality_mask,
                long_size=LONG_SIZE_SUPPORT[1],
                flip=bool(flip),
            )
            binary = pil_mask_to_binary(resized)
            rectangles = _edge_rectangles(binary, flip=bool(flip))
            for orientation in all_rectangles:
                for rectangle in rectangles[orientation]:
                    rectangle["witness_name"] = equality_name
                all_rectangles[orientation].extend(rectangles[orientation])
            source["transforms"].append(
                {
                    "flip": bool(flip),
                    "resized_shape_hw": [resized.height, resized.width],
                    "crop_x_inclusive": [0, resized.width - CROP_SIZE],
                    "crop_y_inclusive": [0, resized.height - CROP_SIZE],
                }
            )
        equality_sources.append(source)

    equality = {}
    full_edge_coverage = True
    for orientation in ("horizontal", "vertical"):
        coverage = coverage_from_rectangles(
            all_rectangles[orientation], orientation=orientation
        )
        expected = int(np.prod(_coverage_shape(orientation, CROP_SIZE)))
        covered = int(coverage.sum())
        complete = covered == expected
        full_edge_coverage = full_edge_coverage and complete
        equality[orientation] = {
            "foreground_neighbor_pairs": len(all_rectangles[orientation]),
            "reachable_output_edges": covered,
            "all_output_edges": expected,
            "complete": complete,
            "coverage_sha256": hashlib.sha256(
                np.ascontiguousarray(coverage.astype(np.uint8)).tobytes()
            ).hexdigest(),
            "greedy_rectangle_certificate": (
                greedy_rectangle_certificate(
                    all_rectangles[orientation], orientation=orientation
                )
                if complete
                else []
            ),
        }
    if not full_edge_coverage:
        raise AssertionError(
            f"{dataset_name}: equality witness does not cover the 4-neighbor grid"
        )

    inequality_path, inequality_image_path, inequality_mask = _load_train_witness(
        dataset_dir, configuration["inequality_name"], fit_names
    )
    inequality_binary = transform_mask(
        inequality_mask,
        long_size=BASE_SIZE,
        flip=False,
        crop_x=0,
        crop_y=0,
    )
    inequality = component_summary(inequality_binary)
    if inequality["component_count"] < 2:
        raise AssertionError(
            f"{dataset_name}: inequality witness no longer has multiple components"
        )

    return {
        "dataset": dataset_name,
        "official_train_manifest": str(train_manifest),
        "official_train_manifest_sha256": sha256_file(train_manifest),
        "official_train_sample_count": len(train_names),
        "internal_split": {
            "split_seed": SPLIT_SEED,
            "val_fraction": VAL_FRACTION,
            "fit_sample_count": len(fit_names),
            "validation_sample_count": len(validation_names),
            "fit_names_sha256": names_sha256(fit_names),
            "validation_names_sha256": names_sha256(validation_names),
            "all_witnesses_are_fit_members": True,
        },
        "equality_witness": {
            "names": list(configuration["equality_names"]),
            "sources": equality_sources,
            "long_size": LONG_SIZE_SUPPORT[1],
            "flip_support_used": list(configuration["equality_flips"]),
            "each_edge_event_has_positive_probability_given_witness_access": True,
            "per_edge_event_probability_lower_bound": "1 / (385 * 2 * 257 * 257)",
            "edge_coverage": equality,
            "four_neighbor_grid_connected": True,
        },
        "inequality_witness": {
            "name": configuration["inequality_name"],
            "mask_path": str(inequality_path),
            "mask_sha256": sha256_file(inequality_path),
            "image_path": str(inequality_image_path),
            "image_sha256": sha256_file(inequality_image_path),
            "repository_resize_geometry_source": "image.size",
            "image_mask_size_equal": True,
            "original_image_size_wh": list(inequality_mask.size),
            "original_size_wh": list(inequality_mask.size),
            "long_size": BASE_SIZE,
            "flip": False,
            "crop_xy": [0, 0],
            "event_has_positive_probability_given_witness_access": True,
            "event_probability": "1 / (385 * 2)",
            **inequality,
        },
        "fixed_disjoint_coordinate_partition_exists": False,
        "contradiction": (
            "no-split closure makes the slice label constant on the connected "
            "256x256 coordinate grid, while the multi-component witness requires "
            "two distinct slice labels"
        ),
    }


def _git_output(arguments: Sequence[str]) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def build_audit(datasets_root: Path, datasets: Sequence[str]) -> dict:
    results = [audit_dataset(datasets_root / name) for name in datasets]
    return {
        "schema_version": 1,
        "audit": "trace_train_augmentation_fixed_partition_closure",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head": _git_output(("rev-parse", "HEAD")),
        "repository_dirty": bool(_git_output(("status", "--porcelain"))),
        "source_files": {
            "augmentation_implementation": str(
                (PROJECT_ROOT / "utils" / "data.py").resolve()
            ),
            "augmentation_implementation_sha256": sha256_file(
                PROJECT_ROOT / "utils" / "data.py"
            ),
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "augmentation_support_used": {
            "base_size": BASE_SIZE,
            "crop_size": CROP_SIZE,
            "repository_long_size_integer_support_inclusive": list(
                LONG_SIZE_SUPPORT
            ),
            "repository_long_size_support_count": (
                LONG_SIZE_SUPPORT[1] - LONG_SIZE_SUPPORT[0] + 1
            ),
            "proof_long_sizes": [256, 512],
            "internal_split_seed": SPLIT_SEED,
            "internal_validation_fraction": VAL_FRACTION,
            "horizontal_flip_probability_each_branch": 0.5,
            "crop_offsets": "all integer offsets are uniform over their legal ranges",
            "mask_resize": "PIL nearest-neighbor",
            "mask_threshold": "ToTensor channel 0 > 0.5 (uint8 >= 128)",
            "gaussian_blur": "irrelevant because it is applied only to the image",
        },
        "formal_claim": {
            "domain": "fixed disjoint coordinate partitions of the 256x256 training crop",
            "requirements": [
                "each observed 8-connected target component lies wholly in one partition cell",
                "distinct observed components lie in distinct partition cells",
            ],
            "proof": [
                "symbolic crop reachability covers every horizontal and vertical grid edge with an adjacent foreground pair",
                "the no-split requirement equates the partition labels at both endpoints of every such edge",
                "the 4-neighbor output grid is connected, so every coordinate has one partition label",
                "a positive-support transform with at least two components requires two partition labels, a contradiction",
            ],
        },
        "datasets": results,
        "all_datasets_certified": all(
            not result["fixed_disjoint_coordinate_partition_exists"]
            for result in results
        ),
        "evidence_scope": {
            "uses_official_train_only": True,
            "reads_test_manifests_or_masks": False,
            "finite_seed_stream_is_not_used_as_proof": True,
            "crop_support_is_symbolically_exhaustive_for_the_frozen_witness_transforms": True,
            "sufficient_support_subset_only": (
                "The proof needs only long_size 512 for equality closure and "
                "long_size 256 for the multi-component contradiction; it does "
                "not claim to enumerate all other augmentation settings."
            ),
            "not_covered": [
                "overlapping crop families",
                "sample-adaptive slicing or assignment",
                "protocols that duplicate, merge, drop, or ignore targets",
            ],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=PROJECT_ROOT / "datasets",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(WITNESSES),
        help="comma-separated frozen dataset names",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "repro_runs"
            / "clean"
            / "trace_stage0_augmentation_partition_closure_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets:
        raise ValueError("at least one dataset is required")
    unknown = sorted(set(datasets) - set(WITNESSES))
    if unknown:
        raise ValueError(f"unknown frozen datasets: {unknown}")
    audit = build_audit(args.datasets_root.resolve(), datasets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
