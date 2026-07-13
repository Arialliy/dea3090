#!/usr/bin/env python3
"""Trace matched, false, and missed components through canonical MSHNet stages."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines.mshnet_deterministic import MSHNet
from utils.data import IRSTD_Dataset
from utils.metric import match_connected_components


STAGES = (
    "stem",
    "encoder0",
    "encoder1",
    "encoder2",
    "encoder3",
    "middle",
    "decoder3",
    "decoder2",
    "decoder1",
    "decoder0",
)


def probability_auc(positive: list[float], negative: list[float]) -> float | None:
    """Probability that a random positive score exceeds a random negative."""

    if not positive or not negative:
        return None
    pos = np.asarray(positive, dtype=np.float64)[:, None]
    neg = np.asarray(negative, dtype=np.float64)[None, :]
    return float(((pos > neg) + 0.5 * (pos == neg)).mean())


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "q25": None, "q75": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload if isinstance(payload, dict) else {}
    state = payload.get("net") if isinstance(payload, dict) and "net" in payload else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    if state and all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state, meta


def forward_trace(model: MSHNet, image: Tensor) -> tuple[dict[str, Tensor], Tensor]:
    stem = model.conv_init(image)
    e0 = model.encoder_0(stem)
    e1 = model.encoder_1(model.pool(e0))
    e2 = model.encoder_2(model.pool(e1))
    e3 = model.encoder_3(model.pool(e2))
    middle = model.middle_layer(model.pool(e3))
    d3 = model.decoder_3(torch.cat([e3, model.up(middle)], dim=1))
    d2 = model.decoder_2(torch.cat([e2, model.up(d3)], dim=1))
    d1 = model.decoder_1(torch.cat([e1, model.up(d2)], dim=1))
    d0 = model.decoder_0(torch.cat([e0, model.up(d1)], dim=1))
    side = [
        model.output_0(d0),
        model.up(model.output_1(d1)),
        model.up_4(model.output_2(d2)),
        model.up_8(model.output_3(d3)),
    ]
    prediction = model.final(torch.cat(side, dim=1))
    return {
        "stem": stem,
        "encoder0": e0,
        "encoder1": e1,
        "encoder2": e2,
        "encoder3": e3,
        "middle": middle,
        "decoder3": d3,
        "decoder2": d2,
        "decoder1": d1,
        "decoder0": d0,
    }, prediction


def normalized_energy_maps(
    features: dict[str, Tensor], output_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for name in STAGES:
        energy = features[name].square().mean(dim=1, keepdim=True).sqrt()
        if energy.shape[-2:] != output_size:
            energy = F.interpolate(
                energy, size=output_size, mode="bilinear", align_corners=True
            )
        mean = energy.mean(dim=(-2, -1), keepdim=True)
        std = energy.std(dim=(-2, -1), keepdim=True, unbiased=False)
        maps[name] = ((energy - mean) / (std + 1e-6))[0, 0].cpu().numpy()
    return maps


def region_scores(
    maps: dict[str, np.ndarray], coordinates: np.ndarray
) -> dict[str, float]:
    y, x = coordinates[:, 0], coordinates[:, 1]
    return {name: float(maps[name][y, x].mean()) for name in STAGES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--test-split-file", required=True)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-centroid-distance", type=float, default=3.0)
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
    dataset_args = SimpleNamespace(
        dataset_dir=str(args.dataset_dir.resolve()),
        evaluation_protocol="official_train_test",
        test_split_file=args.test_split_file,
        train_split_file="",
        val_split_file="",
        val_fraction=0.2,
        split_seed=0,
        seed=0,
        crop_size=args.crop_size,
        base_size=args.base_size,
        return_instance_map=False,
    )
    dataset = IRSTD_Dataset(dataset_args, mode="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = MSHNet(3).to(device).eval()
    state, checkpoint_meta = load_state(args.checkpoint)
    model.load_state_dict(state, strict=True)

    groups = ("matched_prediction", "false_prediction", "matched_target", "missed_target")
    values = {group: {stage: [] for stage in STAGES} for group in groups}
    survival_patterns = {group: Counter() for group in groups}
    counts = Counter()
    with torch.no_grad():
        for image, target in tqdm(loader, desc="stage-component-trace"):
            image = image.to(device)
            features, prediction = forward_trace(model, image)
            probability = torch.sigmoid(prediction)[0, 0].cpu().numpy()
            target_array = (target[0, 0].numpy() > 0.5).astype(np.int64)
            predicted_array = (probability > 0.5).astype(np.int64)
            match = match_connected_components(
                predicted_array,
                target_array,
                max_centroid_distance=args.max_centroid_distance,
            )
            maps = normalized_energy_maps(features, predicted_array.shape)
            matched_prediction = {item[1] for item in match.matches}
            matched_target = {item[0] for item in match.matches}

            for index, region in enumerate(match.prediction_regions):
                group = "matched_prediction" if index in matched_prediction else "false_prediction"
                scores = region_scores(maps, region.coords)
                counts[group] += 1
                pattern = "".join("1" if scores[stage] > 0.0 else "0" for stage in STAGES)
                survival_patterns[group][pattern] += 1
                for stage, score in scores.items():
                    values[group][stage].append(score)
            for index, region in enumerate(match.target_regions):
                group = "matched_target" if index in matched_target else "missed_target"
                scores = region_scores(maps, region.coords)
                counts[group] += 1
                pattern = "".join("1" if scores[stage] > 0.0 else "0" for stage in STAGES)
                survival_patterns[group][pattern] += 1
                for stage, score in scores.items():
                    values[group][stage].append(score)

    stage_report = {}
    for stage in STAGES:
        matched = values["matched_prediction"][stage]
        false = values["false_prediction"][stage]
        detected = values["matched_target"][stage]
        missed = values["missed_target"][stage]
        stage_report[stage] = {
            "matched_prediction": distribution(matched),
            "false_prediction": distribution(false),
            "matched_vs_false_auc": probability_auc(matched, false),
            "matched_target": distribution(detected),
            "missed_target": distribution(missed),
            "matched_vs_missed_target_auc": probability_auc(detected, missed),
        }
    report = {
        "scope": "correlational stage trace on final prediction components; not a causal ablation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_iou": checkpoint_meta.get("iou"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "test_split": str(Path(dataset.list_dir).resolve()),
        "test_split_sha256_normalized": dataset.split_sha256,
        "images": len(dataset),
        "stage_order": list(STAGES),
        "counts": dict(counts),
        "stages": stage_report,
        "top_survival_patterns": {
            group: [
                {"pattern": pattern, "count": count}
                for pattern, count in survival_patterns[group].most_common(10)
            ]
            for group in groups
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
