#!/usr/bin/env python3
"""Audit manifest coverage and image/mask geometry without mutating data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image


EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff")


def _resolve(directory: Path, image_id: str) -> Path | None:
    return next(
        (directory / f"{image_id}{extension}" for extension in EXTENSIONS
         if (directory / f"{image_id}{extension}").is_file()),
        None,
    )


def audit_dataset(dataset: Path, aspect_tolerance: float = 0.01) -> dict[str, Any]:
    manifest_dir = dataset / "img_idx"
    manifests = sorted(
        path
        for path in manifest_dir.glob("*.txt")
        if path.name.startswith(("train_", "test_"))
    )
    ids_by_manifest: dict[str, list[str]] = {}
    all_ids: list[str] = []
    for manifest in manifests:
        ids = [
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ids_by_manifest[manifest.name] = ids
        all_ids.extend(ids)
    unique_ids = list(dict.fromkeys(all_ids))
    missing: list[dict[str, Any]] = []
    rescalable: list[dict[str, Any]] = []
    invalid_geometry: list[dict[str, Any]] = []
    exact_pairs = 0
    for image_id in unique_ids:
        image_path = _resolve(dataset / "images", image_id)
        mask_path = _resolve(dataset / "masks", image_id)
        if image_path is None or mask_path is None:
            missing.append(
                {
                    "id": image_id,
                    "image_present": image_path is not None,
                    "mask_present": mask_path is not None,
                }
            )
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            image_size = image.size
            mask_size = mask.size
        if image_size == mask_size:
            exact_pairs += 1
            continue
        image_ratio = image_size[0] / image_size[1]
        mask_ratio = mask_size[0] / mask_size[1]
        relative_error = abs(image_ratio - mask_ratio) / max(abs(image_ratio), 1e-12)
        record = {
            "id": image_id,
            "image_size": list(image_size),
            "mask_size": list(mask_size),
            "relative_aspect_error": relative_error,
        }
        if relative_error <= aspect_tolerance:
            rescalable.append(record)
        else:
            invalid_geometry.append(record)
    duplicate_counts = {
        name: len(ids) - len(set(ids))
        for name, ids in ids_by_manifest.items()
    }
    return {
        "dataset": dataset.name,
        "manifests": {name: len(ids) for name, ids in ids_by_manifest.items()},
        "manifest_duplicate_counts": duplicate_counts,
        "unique_manifest_ids": len(unique_ids),
        "exact_geometry_pairs": exact_pairs,
        "same_aspect_rescalable_pairs": rescalable,
        "missing_pairs": missing,
        "invalid_geometry_pairs": invalid_geometry,
        "pass": not missing and not invalid_geometry,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--aspect-tolerance", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    names = args.datasets or sorted(path.name for path in args.root.iterdir() if path.is_dir())
    result = {
        "root": str(args.root.resolve()),
        "aspect_tolerance": args.aspect_tolerance,
        "datasets": [
            audit_dataset(args.root / name, args.aspect_tolerance) for name in names
        ],
    }
    result["pass"] = all(item["pass"] for item in result["datasets"])
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
