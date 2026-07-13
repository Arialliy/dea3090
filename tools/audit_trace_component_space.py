#!/usr/bin/env python3
"""Audit whether TRACE's proposed component state space covers the data.

This is a Stage-0 task-definition audit, not a model-selection script.  The
official test manifests are inspected only to establish the benchmark's label
semantics.  In particular, test geometry must not be used to choose the
component family, a solver restriction, or any model hyperparameter.

The augmented-stream audit deliberately instantiates ``IRSTD_Dataset`` and a
``DataLoader`` with the same single-process RNG contract as ``main.py``.  It
therefore measures the labels that optimization actually sees rather than an
approximation of the augmentation policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from skimage import measure
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.data import IRSTD_Dataset


DEFAULT_DATASETS = ("NUAA-SIRST", "NUDT-SIRST", "IRSTD-1K")
DEFAULT_SEEDS = (20260711, 20260712, 20260713)
FORMAL_PROTOCOL = {
    "mask_threshold": 0.5,
    "connectivity": 8,
    "evaluation_resize": [256, 256],
    "evaluation_resize_mode": "nearest",
    "split_seed": 20260711,
    "val_fraction": 0.2,
    "batch_size": 4,
    "drop_last": True,
    "num_workers": 0,
    "augmentation_epochs_per_seed": 1,
}


@dataclass(frozen=True)
class MaskGeometry:
    """Per-sample geometry after repository-compatible binarization."""

    components: int
    foreground_pixels: int
    non_single_row_run_components: int


def parse_csv(text: str, cast) -> list:
    values = [cast(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected at least one comma-separated value"
        )
    return values


def split_sha256(names: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(names) + "\n").encode("utf-8")
    ).hexdigest()


def read_names(path: Path) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"empty split file: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate sample names in split file: {path}")
    return names


def resolve_split_file(dataset_dir: Path, split: str) -> Path:
    """Mirror ``IRSTD_Dataset._resolve_split_file`` for audit provenance."""

    if split not in ("train", "test"):
        raise ValueError(f"unknown split: {split}")
    default_name = "trainval.txt" if split == "train" else "test.txt"
    dataset_name = dataset_dir.name
    candidates = [
        dataset_dir / default_name,
        dataset_dir / "img_idx" / f"{split}_{dataset_name}.txt",
    ]
    candidates.extend(
        Path(path)
        for path in sorted(
            glob.glob(str(dataset_dir / "img_idx" / f"{split}_*.txt"))
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "could not resolve %s split; tried: %s"
        % (split, ", ".join(str(path) for path in candidates))
    )


def deterministic_internal_split(
    source_names: Sequence[str], split_seed: int, val_fraction: float
) -> tuple[list[str], list[str]]:
    """Reproduce ``IRSTD_Dataset._split_train_validation`` exactly."""

    if len(source_names) < 2:
        raise ValueError("at least two training samples are required")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be strictly between zero and one")
    ranked = sorted(
        source_names,
        key=lambda name: hashlib.sha256(
            (f"{split_seed}\0{name}").encode("utf-8")
        ).digest(),
    )
    num_val = max(
        1,
        min(
            len(source_names) - 1,
            int(round(len(source_names) * val_fraction)),
        ),
    )
    val_set = set(ranked[:num_val])
    fit_names = [name for name in source_names if name not in val_set]
    val_names = [name for name in source_names if name in val_set]
    return fit_names, val_names


def pil_mask_to_binary(mask: Image.Image, threshold: float = 0.5) -> np.ndarray:
    """Apply the repository's ``ToTensor()[0] > 0.5`` mask convention."""

    tensor = transforms.ToTensor()(mask)
    if tensor.ndim != 3 or tensor.shape[0] < 1:
        raise ValueError(f"unexpected mask tensor shape: {tuple(tensor.shape)}")
    return (tensor[0].numpy() > threshold).astype(np.uint8)


def is_single_row_run_component(component: np.ndarray) -> bool:
    """Return whether every occupied row contains exactly one contiguous run."""

    component = np.asarray(component, dtype=bool)
    if component.ndim != 2:
        raise ValueError("component must be a 2-D array")
    if not component.any():
        return False
    for row in component:
        columns = np.flatnonzero(row)
        if columns.size and int(columns[-1] - columns[0] + 1) != columns.size:
            return False
    return True


def geometry_from_binary(binary: np.ndarray) -> MaskGeometry:
    """Measure 8-connected components and row-run membership."""

    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("binary mask must be a 2-D array")
    labels = measure.label(
        binary.astype(np.uint8), connectivity=2, background=0
    )
    component_count = int(labels.max())
    non_single_row_run = 0
    for component_id in range(1, component_count + 1):
        if not is_single_row_run_component(labels == component_id):
            non_single_row_run += 1
    return MaskGeometry(
        components=component_count,
        foreground_pixels=int(binary.sum()),
        non_single_row_run_components=non_single_row_run,
    )


def summarize_geometries(geometries: Iterable[MaskGeometry]) -> dict:
    """Aggregate empty/single/multi and component-family coverage counts."""

    values = list(geometries)
    histogram = Counter(item.components for item in values)
    empty = int(histogram.get(0, 0))
    single = int(histogram.get(1, 0))
    multi = int(sum(count for k, count in histogram.items() if k > 1))
    non_ssr_components = int(
        sum(item.non_single_row_run_components for item in values)
    )
    samples_with_non_ssr = int(
        sum(item.non_single_row_run_components > 0 for item in values)
    )
    outside_empty_or_single_ssr = int(
        sum(
            item.components > 1 or item.non_single_row_run_components > 0
            for item in values
        )
    )
    return {
        "sample_count": len(values),
        "empty_samples": empty,
        "single_component_samples": single,
        "multi_component_samples": multi,
        "component_count": int(sum(item.components for item in values)),
        "max_components_per_sample": int(max(histogram, default=0)),
        "component_histogram": {
            str(k): int(histogram[k]) for k in sorted(histogram)
        },
        "foreground_pixels": int(
            sum(item.foreground_pixels for item in values)
        ),
        "non_single_row_run_components": non_ssr_components,
        "samples_with_non_single_row_run_component": samples_with_non_ssr,
        "samples_outside_set_of_single_row_run_components": samples_with_non_ssr,
        "samples_outside_empty_or_single_single_row_run_component": (
            outside_empty_or_single_ssr
        ),
    }


class MaskAuditCache:
    """Avoid decoding the same mask repeatedly across derived train subsets."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._cache: dict[tuple[str, int | None], MaskGeometry] = {}

    def geometry(self, path: Path, resize: int | None) -> MaskGeometry:
        key = (str(path.resolve()), resize)
        if key not in self._cache:
            with Image.open(path) as source:
                mask = source.copy()
            if resize is not None:
                mask = mask.resize((resize, resize), Image.NEAREST)
            self._cache[key] = geometry_from_binary(
                pil_mask_to_binary(mask, self.threshold)
            )
        return self._cache[key]


def audit_named_masks(
    dataset_dir: Path,
    names: Sequence[str],
    resize: int | None,
    cache: MaskAuditCache,
) -> dict:
    missing = [
        str(dataset_dir / "masks" / f"{name}.png")
        for name in names
        if not (dataset_dir / "masks" / f"{name}.png").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "missing masks (first ten): " + ", ".join(missing[:10])
        )
    return summarize_geometries(
        cache.geometry(dataset_dir / "masks" / f"{name}.png", resize)
        for name in names
    )


def _seed_everything(seed: int) -> None:
    """Match the CPU-relevant part of ``main.seed_everything``."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataset_args(
    dataset_dir: Path,
    train_file: Path,
    test_file: Path,
    seed: int,
    split_seed: int,
    val_fraction: float,
    base_size: int,
    crop_size: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_dir=str(dataset_dir.resolve()),
        evaluation_protocol="internal_holdout",
        train_split_file=str(train_file.resolve()),
        val_split_file="",
        test_split_file=str(test_file.resolve()),
        val_fraction=val_fraction,
        split_seed=split_seed,
        seed=seed,
        base_size=base_size,
        crop_size=crop_size,
        return_instance_map=False,
    )


def audit_augmented_epoch(
    dataset_dir: Path,
    train_file: Path,
    test_file: Path,
    seed: int,
    split_seed: int,
    val_fraction: float,
    base_size: int,
    crop_size: int,
    batch_size: int = 4,
    num_workers: int = 0,
) -> tuple[dict, list[str]]:
    """Audit one actual seeded ``utils/data.py`` internal-fit epoch."""

    if num_workers != 0:
        raise ValueError("formal TRACE Stage-0 audit fixes num_workers=0")
    _seed_everything(seed)
    args = make_dataset_args(
        dataset_dir,
        train_file,
        test_file,
        seed,
        split_seed,
        val_fraction,
        base_size,
        crop_size,
    )
    dataset = IRSTD_Dataset(args, mode="train")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    geometries: list[MaskGeometry] = []
    batch_count = 0
    for _, masks in loader:
        batch_count += 1
        for mask in masks:
            geometries.append(
                geometry_from_binary((mask[0].numpy() > 0.5).astype(np.uint8))
            )
    summary = summarize_geometries(geometries)
    summary.update(
        {
            "seed": seed,
            "batch_count": batch_count,
            "dataset_fit_count_before_drop_last": len(dataset),
            "dropped_by_drop_last": len(dataset) - len(geometries),
            "fit_split_sha256": dataset.split_sha256,
        }
    )
    return summary, list(dataset.names)


def merge_augmented_summaries(per_seed: Sequence[dict]) -> dict:
    """Merge per-seed counts without losing the formal sample denominator."""

    count_fields = (
        "sample_count",
        "empty_samples",
        "single_component_samples",
        "multi_component_samples",
        "component_count",
        "foreground_pixels",
        "non_single_row_run_components",
        "samples_with_non_single_row_run_component",
        "samples_outside_set_of_single_row_run_components",
        "samples_outside_empty_or_single_single_row_run_component",
        "batch_count",
        "dropped_by_drop_last",
    )
    merged = {
        field: int(sum(int(item[field]) for item in per_seed))
        for field in count_fields
    }
    merged["max_components_per_sample"] = int(
        max((int(item["max_components_per_sample"]) for item in per_seed), default=0)
    )
    histogram: Counter[int] = Counter()
    for item in per_seed:
        histogram.update(
            {
                int(k): int(value)
                for k, value in item["component_histogram"].items()
            }
        )
    merged["component_histogram"] = {
        str(k): int(histogram[k]) for k in sorted(histogram)
    }
    merged["seed_count"] = len(per_seed)
    return merged


def audit_dataset(
    dataset_dir: Path,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    split_seed: int = 20260711,
    val_fraction: float = 0.2,
    base_size: int = 256,
    crop_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict:
    dataset_dir = dataset_dir.resolve()
    train_file = resolve_split_file(dataset_dir, "train")
    test_file = resolve_split_file(dataset_dir, "test")
    official_train = read_names(train_file)
    official_test = read_names(test_file)
    overlap = sorted(set(official_train).intersection(official_test))
    if overlap:
        raise ValueError(
            f"official train/test overlap (first ten): {overlap[:10]}"
        )

    expected_fit, expected_val = deterministic_internal_split(
        official_train, split_seed, val_fraction
    )
    cache = MaskAuditCache(threshold=0.5)

    official = {}
    for split_name, names, manifest in (
        ("train", official_train, train_file),
        ("test", official_test, test_file),
    ):
        official[split_name] = {
            "manifest": str(manifest),
            "manifest_file_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "ordered_ids_sha256": split_sha256(names),
            "raw": audit_named_masks(dataset_dir, names, None, cache),
            f"nearest_resize_{base_size}": audit_named_masks(
                dataset_dir, names, base_size, cache
            ),
        }

    per_seed = []
    observed_fit_names: list[str] | None = None
    for seed in seeds:
        seed_summary, fit_names = audit_augmented_epoch(
            dataset_dir,
            train_file,
            test_file,
            int(seed),
            split_seed,
            val_fraction,
            base_size,
            crop_size,
            batch_size,
            num_workers,
        )
        if fit_names != expected_fit:
            raise AssertionError(
                "utils/data.py internal-fit differs from independently reproduced split"
            )
        if observed_fit_names is not None and fit_names != observed_fit_names:
            raise AssertionError("internal-fit membership changed across model seeds")
        observed_fit_names = fit_names
        per_seed.append(seed_summary)

    return {
        "dataset": dataset_dir.name,
        "dataset_dir": str(dataset_dir),
        "official_splits": official,
        "official_overlap_count": 0,
        "internal_holdout": {
            "source": "official train only",
            "split_seed": split_seed,
            "val_fraction": val_fraction,
            "fit_count": len(expected_fit),
            "validation_count": len(expected_val),
            "fit_ordered_ids_sha256": split_sha256(expected_fit),
            "validation_ordered_ids_sha256": split_sha256(expected_val),
            "fit_raw": audit_named_masks(
                dataset_dir, expected_fit, None, cache
            ),
            f"fit_nearest_resize_{base_size}": audit_named_masks(
                dataset_dir, expected_fit, base_size, cache
            ),
            "validation_raw": audit_named_masks(
                dataset_dir, expected_val, None, cache
            ),
            f"validation_nearest_resize_{base_size}": audit_named_masks(
                dataset_dir, expected_val, base_size, cache
            ),
        },
        "actual_augmented_internal_fit_epoch": {
            "loader_contract": {
                "implementation": "utils.data.IRSTD_Dataset",
                "shuffle": True,
                "batch_size": batch_size,
                "drop_last": True,
                "num_workers": num_workers,
                "pin_memory": True,
                "persistent_workers": False,
                "generator_seed": "the corresponding model seed",
                "worker_init_fn": "main.seed_worker-equivalent",
                "base_size": base_size,
                "crop_size": crop_size,
            },
            "seeds": [int(seed) for seed in seeds],
            "per_seed": per_seed,
            "aggregate": merge_augmented_summaries(per_seed),
        },
    }


def repository_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_report(
    dataset_dirs: Sequence[Path],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    split_seed: int = 20260711,
    val_fraction: float = 0.2,
    base_size: int = 256,
    crop_size: int = 256,
    batch_size: int = 4,
    num_workers: int = 0,
) -> dict:
    datasets = [
        audit_dataset(
            dataset_dir,
            seeds,
            split_seed=split_seed,
            val_fraction=val_fraction,
            base_size=base_size,
            crop_size=crop_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        for dataset_dir in dataset_dirs
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": str(Path(__file__).resolve()),
        "repository_head": repository_head(),
        "protocol": {
            **FORMAL_PROTOCOL,
            "evaluation_resize": [base_size, base_size],
            "split_seed": split_seed,
            "val_fraction": val_fraction,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "seeds": [int(seed) for seed in seeds],
        },
        "component_family_definition": {
            "name": "single-row-run (SSR) component",
            "criterion": (
                "each 8-connected component has exactly one contiguous run "
                "in every occupied row"
            ),
        },
        "test_usage_guardrail": {
            "purpose": "task-definition validation only",
            "used_for_component_family_selection": False,
            "used_for_hyperparameter_selection": False,
            "used_for_model_selection": False,
            "permitted_component_family_selection_inputs": [
                "official_train labels",
                "internal-fit augmented labels",
            ],
            "statement": (
                "Official test labels are audited only to verify the frozen "
                "benchmark task definition; they must not be used to choose C, "
                "a solver restriction, or a model/hyperparameter."
            ),
        },
        "datasets": datasets,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_dataset_dirs(tokens: Sequence[str], dataset_root: Path) -> list[Path]:
    paths = []
    for token in tokens:
        candidate = Path(token).expanduser()
        if not candidate.is_absolute() and not candidate.exists():
            candidate = dataset_root / token
        if not candidate.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {candidate}")
        paths.append(candidate.resolve())
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit official and actual augmented mask component geometry for "
            "TRACE Stage 0."
        )
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset directory names or paths.",
    )
    parser.add_argument(
        "--dataset-root", type=Path, default=PROJECT_ROOT / "datasets"
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated augmentation/model seeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "repro_runs"
            / "clean"
            / "trace_stage0_component_space_audit_v1.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_tokens = parse_csv(args.datasets, str)
    seeds = parse_csv(args.seeds, int)
    dataset_dirs = resolve_dataset_dirs(dataset_tokens, args.dataset_root)
    report = build_report(dataset_dirs, seeds)
    output = args.output.expanduser().resolve()
    write_json(output, report)
    print(json.dumps({
        "output": str(output),
        "datasets": [item["dataset"] for item in report["datasets"]],
        "test_usage": report["test_usage_guardrail"]["statement"],
    }, indent=2))


if __name__ == "__main__":
    main()
