#!/usr/bin/env python3
"""Audit SPT's operator occupancy and its isolated causal effect."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
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
from model.support_persistence_transport import SupportPersistenceMSHNet
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, match_connected_components


FIELDS = (
    "persistence",
    "exclusive_fraction",
    "removal_rate",
    "removed_fraction",
)
GROUPS = ("matched_prediction", "false_prediction", "matched_target", "missed_target")


def probability_auc(positive: list[float], negative: list[float]) -> float | None:
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state(path: Path) -> tuple[dict[str, Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    meta = payload if isinstance(payload, dict) else {}
    state = payload.get("net") if isinstance(payload, dict) and "net" in payload else payload
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state, meta


def forward_spt0_with_state(
    model: SupportPersistenceMSHNet, image: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    e0 = model.encoder_0(model.conv_init(image))
    e1_input, state = model.support_persistence(e0, return_state=True)
    e1 = model.encoder_1(e1_input)
    e2 = model.encoder_2(model.pool(e1))
    e3 = model.encoder_3(model.pool(e2))
    middle = model.middle_layer(model.pool(e3))
    d3 = model.decoder_3(torch.cat([e3, model.up(middle)], dim=1))
    d2 = model.decoder_2(torch.cat([e2, model.up(d3)], dim=1))
    d1 = model.decoder_1(torch.cat([e1, model.up(d2)], dim=1))
    d0 = model.decoder_0(torch.cat([e0, model.up(d1)], dim=1))
    logits = model.final(
        torch.cat(
            [
                model.output_0(d0),
                model.up(model.output_1(d1)),
                model.up_4(model.output_2(d2)),
                model.up_8(model.output_3(d3)),
            ],
            dim=1,
        )
    )
    return logits, state


def state_maps(state: dict[str, Tensor], output_size: tuple[int, int]) -> dict[str, np.ndarray]:
    maximum = state["maximum"]
    exclusive = state["single_site_ownership"]
    persistence = state["channel_persistence"]
    gate = state["survival_gate"]
    denominator = maximum.abs() + 1e-6
    maps = {
        "persistence": persistence.mean(dim=1, keepdim=True),
        "exclusive_fraction": (exclusive / denominator).mean(dim=1, keepdim=True),
        "removal_rate": (1.0 - gate).mean(dim=1, keepdim=True),
        "removed_fraction": ((1.0 - gate) * exclusive / denominator).mean(
            dim=1, keepdim=True
        ),
    }
    return {
        name: F.interpolate(value, size=output_size, mode="bilinear", align_corners=True)[
            0, 0
        ].cpu().numpy()
        for name, value in maps.items()
    }


def region_values(maps: dict[str, np.ndarray], coordinates: np.ndarray) -> dict[str, float]:
    y, x = coordinates[:, 0], coordinates[:, 1]
    return {name: float(value[y, x].mean()) for name, value in maps.items()}


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
        test_split_file=args.test_split_file,
        val_fraction=0.2,
        split_seed=0,
        seed=0,
        crop_size=args.crop_size,
        base_size=args.base_size,
        return_instance_map=False,
    )
    dataset = IRSTD_Dataset(dataset_args, mode="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    state_dict, checkpoint_meta = load_state(args.checkpoint)
    spt = SupportPersistenceMSHNet(3, active_stages=(0,)).to(device).eval()
    maxpool = MSHNet(3).to(device).eval()
    spt.load_state_dict(state_dict, strict=True)
    maxpool.load_state_dict(state_dict, strict=True)

    variants = ("spt0", "same_weights_maxpool")
    intersections = dict.fromkeys(variants, 0)
    unions = dict.fromkeys(variants, 0)
    pd_fa = {name: PD_FA(nclass=1, bins=10, size=args.crop_size) for name in variants}
    matched_targets = dict.fromkeys(variants, 0)
    false_components = dict.fromkeys(variants, 0)
    total_targets = 0
    changed_pixels = 0
    absolute_logit_change = 0.0
    total_pixels = 0
    values: dict[str, dict[str, list[float]]] = {
        group: defaultdict(list) for group in GROUPS
    }

    with torch.no_grad():
        for image, label in tqdm(loader, desc="SPT-mechanism"):
            image = image.to(device)
            truth_device = label.to(device) > 0.5
            spt_logits, operator_state = forward_spt0_with_state(spt, image)
            _, max_logits = maxpool(image, True)
            logits_by_name = {"spt0": spt_logits, "same_weights_maxpool": max_logits}
            target = (label[0, 0].numpy() > 0.5).astype(np.int64)
            spt_prediction = (spt_logits[0, 0].cpu().numpy() > 0.0).astype(np.int64)
            match = match_connected_components(
                spt_prediction,
                target,
                max_centroid_distance=args.max_centroid_distance,
            )
            total_targets += len(match.target_regions)
            matched_prediction = {item[1] for item in match.matches}
            matched_target = {item[0] for item in match.matches}
            maps = state_maps(operator_state, spt_prediction.shape)

            for index, region in enumerate(match.prediction_regions):
                group = "matched_prediction" if index in matched_prediction else "false_prediction"
                for name, value in region_values(maps, region.coords).items():
                    values[group][name].append(value)
            for index, region in enumerate(match.target_regions):
                group = "matched_target" if index in matched_target else "missed_target"
                for name, value in region_values(maps, region.coords).items():
                    values[group][name].append(value)

            for name, logits in logits_by_name.items():
                prediction = logits > 0.0
                intersections[name] += int((prediction & truth_device).sum())
                unions[name] += int((prediction | truth_device).sum())
                pd_fa[name].update(logits.cpu(), label)
                local_match = match_connected_components(
                    prediction[0, 0].cpu().numpy().astype(np.int64),
                    target,
                    max_centroid_distance=args.max_centroid_distance,
                )
                matched_targets[name] += len(local_match.matches)
                false_components[name] += len(local_match.unmatched_prediction_indices)

            changed_pixels += int(((spt_logits > 0) != (max_logits > 0)).sum())
            absolute_logit_change += float((spt_logits - max_logits).abs().sum())
            total_pixels += int(spt_logits.numel())

    causal_rows = []
    for name in variants:
        false_alarm, detection_probability = pd_fa[name].get()
        causal_rows.append(
            {
                "name": name,
                "iou": float(intersections[name] / max(1, unions[name])),
                "pd": float(detection_probability[0]),
                "fa_per_million": float(false_alarm[0] * 1e6),
                "matched_targets": int(matched_targets[name]),
                "target_count": int(total_targets),
                "false_components": int(false_components[name]),
                "fppi_at_logit_zero": float(false_components[name] / len(dataset)),
            }
        )
    strata = {}
    for field in FIELDS:
        strata[field] = {
            group: distribution(values[group][field]) for group in GROUPS
        }
        strata[field]["matched_prediction_vs_false_auc"] = probability_auc(
            values["matched_prediction"][field], values["false_prediction"][field]
        )
        strata[field]["false_prediction_vs_matched_auc"] = probability_auc(
            values["false_prediction"][field], values["matched_prediction"][field]
        )
        strata[field]["matched_target_vs_missed_auc"] = probability_auc(
            values["matched_target"][field], values["missed_target"][field]
        )

    report = {
        "scope": "trained-weight causal operator audit plus component-stratified occupancy",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_iou": checkpoint_meta.get("iou"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "test_split": str(Path(dataset.list_dir).resolve()),
        "test_split_sha256_normalized": dataset.split_sha256,
        "images": len(dataset),
        "parameter_delta_vs_mshnet": 0,
        "changed_binary_pixels": changed_pixels,
        "mean_absolute_logit_change": absolute_logit_change / max(1, total_pixels),
        "causal_rows": causal_rows,
        "operator_strata": strata,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
