#!/usr/bin/env python3
"""Frozen MSHNet audit of leave-one-peak interventions at one pool boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines.mshnet_deterministic import MSHNet
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, match_connected_components
from utils.order_statistic_pool import (
    channel_consensus_pool2d,
    counterfactual_self_support_pool2d,
    leave_one_channel_influence_pool2d,
    leave_one_peak_pool2d,
    support_persistence_pool2d,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--test-split-file", required=True)
    parser.add_argument("--stage", type=int, choices=range(4), required=True)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=(0.0, 0.05, 0.1, 0.2, 0.4, 1.0)
    )
    parser.add_argument("--include-channel-consensus", action="store_true")
    parser.add_argument("--include-support-persistence", action="store_true")
    parser.add_argument("--include-leave-one-channel-influence", action="store_true")
    parser.add_argument("--include-counterfactual-self-support", action="store_true")
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


def forward_with_pool_counterfactual(
    model: MSHNet,
    image: Tensor,
    *,
    stage: int,
    alpha: float = 0.0,
    operator: str = "leave_one_peak",
) -> Tensor:
    """Run canonical MSHNet while changing exactly one pool boundary."""

    if stage not in range(4):
        raise ValueError("stage must be one of 0, 1, 2, 3")

    def pool(x: Tensor, index: int) -> Tensor:
        if index == stage:
            if operator == "leave_one_peak":
                return leave_one_peak_pool2d(x, alpha)
            if operator == "channel_consensus":
                return channel_consensus_pool2d(x)
            if operator == "support_persistence":
                return support_persistence_pool2d(x)
            if operator == "leave_one_channel_influence":
                return leave_one_channel_influence_pool2d(x)
            if operator == "counterfactual_self_support":
                return counterfactual_self_support_pool2d(x)
            raise ValueError("unsupported pooling counterfactual")
        return model.pool(x)

    e0 = model.encoder_0(model.conv_init(image))
    e1 = model.encoder_1(pool(e0, 0))
    e2 = model.encoder_2(pool(e1, 1))
    e3 = model.encoder_3(pool(e2, 2))
    middle = model.middle_layer(pool(e3, 3))
    d3 = model.decoder_3(torch.cat([e3, model.up(middle)], dim=1))
    d2 = model.decoder_2(torch.cat([e2, model.up(d3)], dim=1))
    d1 = model.decoder_1(torch.cat([e1, model.up(d2)], dim=1))
    d0 = model.decoder_0(torch.cat([e0, model.up(d1)], dim=1))
    side = torch.cat(
        [
            model.output_0(d0),
            model.up(model.output_1(d1)),
            model.up_4(model.output_2(d2)),
            model.up_8(model.output_3(d3)),
        ],
        dim=1,
    )
    return model.final(side)


def main() -> None:
    args = parse_args()
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas must not contain duplicates")
    if any(not 0.0 <= alpha <= 1.0 for alpha in args.alphas):
        raise ValueError("--alphas must lie in [0, 1]")
    if 0.0 not in args.alphas:
        raise ValueError("--alphas must contain 0 for the identity audit")
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
    model = MSHNet(3).to(device).eval()
    state, checkpoint_meta = load_state(args.checkpoint)
    model.load_state_dict(state, strict=True)

    candidates: list[tuple[str, str, float | None]] = [
        (f"leave_one_peak_alpha_{float(alpha):g}", "leave_one_peak", float(alpha))
        for alpha in args.alphas
    ]
    if args.include_channel_consensus:
        candidates.append(("channel_consensus", "channel_consensus", None))
    if args.include_support_persistence:
        candidates.append(("support_persistence", "support_persistence", None))
    if args.include_leave_one_channel_influence:
        candidates.append(
            (
                "leave_one_channel_influence",
                "leave_one_channel_influence",
                None,
            )
        )
    if args.include_counterfactual_self_support:
        candidates.append(
            (
                "counterfactual_self_support",
                "counterfactual_self_support",
                None,
            )
        )
    names = [name for name, _, _ in candidates]
    intersection = {name: 0 for name in names}
    union = {name: 0 for name in names}
    matched_targets = {name: 0 for name in names}
    false_components = {name: 0 for name in names}
    false_component_area = {name: 0 for name in names}
    target_count = 0
    pd_fa = {alpha: PD_FA(nclass=1, bins=10, size=args.crop_size) for alpha in names}
    max_identity_error = 0.0

    with torch.no_grad():
        for image, label in tqdm(loader, desc=f"pool-{args.stage}-counterfactual"):
            image_device = image.to(device)
            label_device = label.to(device)
            _, canonical = model(image_device, True)
            target = (label[0, 0].numpy() > 0.5).astype(np.int64)
            counted_target = False
            for name, operator, alpha in candidates:
                logits = forward_with_pool_counterfactual(
                    model,
                    image_device,
                    stage=args.stage,
                    alpha=0.0 if alpha is None else alpha,
                    operator=operator,
                )
                if operator == "leave_one_peak" and alpha == 0.0:
                    max_identity_error = max(
                        max_identity_error, float((logits - canonical).abs().max())
                    )
                prediction = logits > 0.0
                truth = label_device > 0.5
                intersection[name] += int((prediction & truth).sum())
                union[name] += int((prediction | truth).sum())
                pd_fa[name].update(logits.cpu(), label)
                match = match_connected_components(
                    prediction[0, 0].cpu().numpy().astype(np.int64),
                    target,
                    max_centroid_distance=args.max_centroid_distance,
                )
                if not counted_target:
                    target_count += len(match.target_regions)
                    counted_target = True
                matched_targets[name] += len(match.matches)
                false_components[name] += len(match.unmatched_prediction_indices)
                false_component_area[name] += sum(
                    match.prediction_regions[index].area
                    for index in match.unmatched_prediction_indices
                )

    rows = []
    for name, operator, alpha in candidates:
        false_alarm, detection_probability = pd_fa[name].get()
        rows.append(
            {
                "name": name,
                "operator": operator,
                "alpha": alpha,
                "iou": float(intersection[name] / max(1, union[name])),
                "pd": float(detection_probability[0]),
                "fa_per_million": float(false_alarm[0] * 1e6),
                "matched_targets": int(matched_targets[name]),
                "target_count": int(target_count),
                "false_components": int(false_components[name]),
                "fppi_at_logit_zero": float(false_components[name] / len(dataset)),
                "false_component_area": int(false_component_area[name]),
            }
        )
    report = {
        "scope": "frozen-checkpoint causal screening control; not a trained model",
        "intervention": "convex max/leave-one-peak 2x2 pooling at exactly one boundary",
        "stage": args.stage,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_iou": checkpoint_meta.get("iou"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "test_split": str(Path(dataset.list_dir).resolve()),
        "test_split_sha256_normalized": dataset.split_sha256,
        "images": len(dataset),
        "max_alpha_zero_identity_abs_error": max_identity_error,
        "rows": rows,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
