#!/usr/bin/env python3
"""Paired image/component bootstrap for two frozen MSHNet checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines.mshnet_deterministic import MSHNet
from utils.data import IRSTD_Dataset
from utils.metric import match_connected_components


METRICS = ("IoU", "PD", "FA")


def _load_clean_model(path: Path, device: torch.device) -> MSHNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["net"] if isinstance(payload, dict) and "net" in payload else payload
    model = MSHNet(3)
    keys = set(model.state_dict())
    missing = keys - set(state)
    if missing:
        raise ValueError(f"checkpoint missing MSHNet keys: {sorted(missing)[:5]}")
    model.load_state_dict({key: state[key] for key in keys}, strict=True)
    return model.to(device).eval()


def _per_image_statistics(
    model: MSHNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    rows = {
        "intersection": [],
        "union": [],
        "matches": [],
        "targets": [],
        "false_alarm_area": [],
        "image_area": [],
    }
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            _, logits = model(images, True)
            predictions = (logits > 0).cpu().numpy()
            target_array = (targets > 0.5).numpy()
            for index in range(images.shape[0]):
                prediction = predictions[index, 0]
                target = target_array[index, 0]
                rows["intersection"].append(int((prediction & target).sum()))
                rows["union"].append(int((prediction | target).sum()))
                match = match_connected_components(prediction, target)
                rows["matches"].append(len(match.matches))
                rows["targets"].append(len(match.target_regions))
                rows["false_alarm_area"].append(
                    sum(
                        match.prediction_regions[item].area
                        for item in match.unmatched_prediction_indices
                    )
                )
                rows["image_area"].append(int(prediction.size))
    return {key: np.asarray(values, dtype=np.float64) for key, values in rows.items()}


def _metrics(stats: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    intersection = stats["intersection"][indices].sum(axis=-1)
    union = stats["union"][indices].sum(axis=-1)
    matches = stats["matches"][indices].sum(axis=-1)
    targets = stats["targets"][indices].sum(axis=-1)
    false_alarm_area = stats["false_alarm_area"][indices].sum(axis=-1)
    image_area = stats["image_area"][indices].sum(axis=-1)
    return np.stack(
        [
            np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0),
            np.divide(matches, targets, out=np.zeros_like(matches), where=targets > 0),
            np.divide(
                false_alarm_area * 1e6,
                image_area,
                out=np.zeros_like(false_alarm_area),
                where=image_area > 0,
            ),
        ],
        axis=-1,
    )


def paired_bootstrap(
    baseline: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    image_count = len(baseline["intersection"])
    if image_count != len(candidate["intersection"]):
        raise ValueError("paired statistics have different image counts")
    if samples < 1:
        raise ValueError("samples must be positive")
    all_indices = np.arange(image_count)[None, :]
    baseline_point = _metrics(baseline, all_indices)[0]
    candidate_point = _metrics(candidate, all_indices)[0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, image_count, size=(samples, image_count))
    delta = _metrics(candidate, indices) - _metrics(baseline, indices)
    report: dict[str, Any] = {
        "images": image_count,
        "samples": samples,
        "baseline": dict(zip(METRICS, baseline_point.tolist())),
        "candidate": dict(zip(METRICS, candidate_point.tolist())),
        "delta": dict(zip(METRICS, (candidate_point - baseline_point).tolist())),
        "bootstrap": {},
    }
    for metric_index, metric in enumerate(METRICS):
        values = delta[:, metric_index]
        report["bootstrap"][metric] = {
            "mean_delta": float(values.mean()),
            "percentile_95_ci": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "probability_delta_gt_zero": float((values > 0).mean()),
            "probability_delta_lt_zero": float((values < 0).mean()),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset-dir", default="datasets/NUAA-SIRST")
    parser.add_argument("--train-split-file", default="")
    parser.add_argument("--val-split-file", default="")
    parser.add_argument("--test-split-file", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260711)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    dataset = IRSTD_Dataset(args, "val")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    baseline_model = _load_clean_model(args.baseline, device)
    candidate_model = _load_clean_model(args.candidate, device)
    baseline_stats = _per_image_statistics(baseline_model, loader, device)
    candidate_stats = _per_image_statistics(candidate_model, loader, device)
    report = paired_bootstrap(
        baseline_stats,
        candidate_stats,
        samples=args.samples,
        seed=args.seed,
    )
    report.update(
        {
            "baseline_checkpoint": str(args.baseline),
            "candidate_checkpoint": str(args.candidate),
            "split_sha256": dataset.split_sha256,
        }
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
