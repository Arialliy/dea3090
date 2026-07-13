#!/usr/bin/env python3
"""Audit whether CSL's stage-0 deletion residual separates component types."""

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
from model.counterfactual_sufficient_lift import CounterfactualSufficientPool2d
from tools.audit_mshnet_stage_component_trace import distribution, probability_auc
from utils.data import IRSTD_Dataset
from utils.metric import match_connected_components


STATISTICS = (
    "factual_energy",
    "deleted_energy",
    "exclusive_energy",
    "survival_ratio",
    "exclusive_ratio",
    "mean_channel_exclusive_fraction",
    "rank3_survival_ratio",
    "rank4_survival_ratio",
    "rank2_marginal_ratio",
    "rank3_marginal_ratio",
    "rank4_marginal_ratio",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload if isinstance(payload, dict) else {}
    state = payload.get("net") if isinstance(payload, dict) and "net" in payload else payload
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    variant = metadata.get("method_meta", {}).get("mshnet_variant")
    if variant != "deterministic":
        raise ValueError(
            "CSL stage audit requires a deterministic baseline checkpoint; "
            f"metadata reports {variant!r}"
        )
    return state, metadata


def csl_statistic_maps(e0: Tensor, output_size: tuple[int, int]) -> dict[str, np.ndarray]:
    _, state = CounterfactualSufficientPool2d()(e0, return_state=True)
    factual = state["factual_maximum"]
    deleted = state["deleted_maximum"]
    exclusive = state["exclusive_residual"]
    eps = torch.finfo(e0.dtype).eps
    factual_energy = factual.square().mean(dim=1, keepdim=True).sqrt()
    deleted_energy = deleted.square().mean(dim=1, keepdim=True).sqrt()
    exclusive_energy = exclusive.square().mean(dim=1, keepdim=True).sqrt()
    batch, channels, height, width = e0.shape
    cells = F.unfold(e0, kernel_size=2, stride=2).view(
        batch, channels, 4, height // 2, width // 2
    )
    ranks = torch.sort(cells, dim=2, descending=True).values

    def energy(value: Tensor) -> Tensor:
        return value.square().mean(dim=1, keepdim=True).sqrt()

    rank3_energy = energy(ranks[:, :, 2])
    rank4_energy = energy(ranks[:, :, 3])
    rank2_marginal = energy(ranks[:, :, 1] - ranks[:, :, 2])
    rank3_marginal = energy(ranks[:, :, 2] - ranks[:, :, 3])
    rank4_marginal = rank4_energy
    maps = {
        "factual_energy": factual_energy,
        "deleted_energy": deleted_energy,
        "exclusive_energy": exclusive_energy,
        "survival_ratio": deleted_energy / (factual_energy + eps),
        "exclusive_ratio": exclusive_energy / (factual_energy + eps),
        "mean_channel_exclusive_fraction": (
            exclusive / (factual.abs() + eps)
        ).mean(dim=1, keepdim=True),
        "rank3_survival_ratio": rank3_energy / (factual_energy + eps),
        "rank4_survival_ratio": rank4_energy / (factual_energy + eps),
        "rank2_marginal_ratio": rank2_marginal / (factual_energy + eps),
        "rank3_marginal_ratio": rank3_marginal / (factual_energy + eps),
        "rank4_marginal_ratio": rank4_marginal / (factual_energy + eps),
    }
    return {
        name: F.interpolate(value, size=output_size, mode="nearest")[0, 0]
        .detach()
        .cpu()
        .numpy()
        for name, value in maps.items()
    }


def region_scores(
    maps: dict[str, np.ndarray], coordinates: np.ndarray
) -> dict[str, float]:
    rows, columns = coordinates[:, 0], coordinates[:, 1]
    return {
        name: float(value[rows, columns].mean()) for name, value in maps.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split-file", required=True)
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
        train_split_file="",
        val_split_file="",
        test_split_file=args.split_file,
        val_fraction=0.2,
        split_seed=0,
        seed=0,
        crop_size=args.crop_size,
        base_size=args.base_size,
        return_instance_map=False,
    )
    # Test mode is intentional even for a train manifest: the audit must not
    # introduce random augmentation into its component attribution statistics.
    dataset = IRSTD_Dataset(dataset_args, mode="test")
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    model = MSHNet(3).to(device).eval()
    state, checkpoint_metadata = load_state(args.checkpoint)
    model.load_state_dict(state, strict=True)

    groups = (
        "matched_prediction",
        "false_prediction",
        "matched_target",
        "missed_target",
    )
    values = {group: {name: [] for name in STATISTICS} for group in groups}
    counts: Counter[str] = Counter()
    with torch.no_grad():
        for image, target in tqdm(loader, desc="CSL-stage0-audit"):
            image = image.to(device)
            e0 = model.encoder_0(model.conv_init(image))
            _, prediction = model(image, True)
            predicted = (prediction[0, 0].cpu().numpy() > 0.0).astype(np.int64)
            target_array = (target[0, 0].numpy() > 0.5).astype(np.int64)
            match = match_connected_components(
                predicted,
                target_array,
                max_centroid_distance=args.max_centroid_distance,
            )
            maps = csl_statistic_maps(e0, predicted.shape)
            matched_predictions = {item[1] for item in match.matches}
            matched_targets = {item[0] for item in match.matches}

            for index, region in enumerate(match.prediction_regions):
                group = (
                    "matched_prediction"
                    if index in matched_predictions
                    else "false_prediction"
                )
                scores = region_scores(maps, region.coords)
                counts[group] += 1
                for name, score in scores.items():
                    values[group][name].append(score)
            for index, region in enumerate(match.target_regions):
                group = "matched_target" if index in matched_targets else "missed_target"
                scores = region_scores(maps, region.coords)
                counts[group] += 1
                for name, score in scores.items():
                    values[group][name].append(score)

    statistics = {}
    for name in STATISTICS:
        matched = values["matched_prediction"][name]
        false = values["false_prediction"][name]
        detected = values["matched_target"][name]
        missed = values["missed_target"][name]
        statistics[name] = {
            "matched_prediction": distribution(matched),
            "false_prediction": distribution(false),
            "matched_vs_false_auc": probability_auc(matched, false),
            "matched_target": distribution(detected),
            "missed_target": distribution(missed),
            "matched_vs_missed_target_auc": probability_auc(detected, missed),
        }
    report = {
        "scope": (
            "correlational frozen-checkpoint audit of the exact CSL coordinate; "
            "not a performance result or model-selection pass"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_iou": checkpoint_metadata.get("iou"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "split": str(Path(dataset.list_dir).resolve()),
        "split_sha256_normalized": dataset.split_sha256,
        "images": len(dataset),
        "counts": dict(counts),
        "statistics": statistics,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
