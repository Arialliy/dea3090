#!/usr/bin/env python3
"""Audit decision-flip scale responsibility on a frozen MSHNet split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.MSHNet import MSHNet
from model.counterfactual_responsibility import build_safe_background
from model.mshnet_evidence_view import forward_mshnet_evidence
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, match_connected_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", default="datasets/NUAA-SIRST")
    parser.add_argument("--mode", choices=("val", "test"), default="val")
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
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--safe-kernel", type=int, default=15)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_state_dict(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "net" in checkpoint:
        return checkpoint["net"]
    return checkpoint


def divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    dataset = IRSTD_Dataset(args, args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = MSHNet(args.input_channels).to(device).eval()
    model.load_state_dict(load_state_dict(args.checkpoint))

    component_metric = PD_FA(nclass=1, bins=10, size=args.crop_size)
    thresholds = tuple(float(value) for value in component_metric.thresholds)
    intersections = np.zeros(len(thresholds), dtype=np.int64)
    unions = np.zeros(len(thresholds), dtype=np.int64)

    total_pixels = 0
    safe_pixels = 0
    target_pixels = 0
    predicted_pixels = 0
    safe_positive_pixels = 0
    responsible_pixel_union = 0
    responsible_events = torch.zeros(4, dtype=torch.float64)
    responsible_contribution_sum = torch.zeros(4, dtype=torch.float64)
    responsible_full_margin_sum = torch.zeros(4, dtype=torch.float64)
    responsible_deleted_margin_sum = torch.zeros(4, dtype=torch.float64)
    active_images = 0
    reconstruction_error = 0.0
    predicted_components = 0
    false_alarm_components = 0
    matched_components = 0
    false_alarm_components_with_responsibility = 0
    matched_components_with_responsibility = 0
    responsible_pixels_in_false_alarm_components = 0
    responsible_pixels_in_matched_components = 0
    responsible_events_in_false_alarm_components = 0
    responsible_events_in_matched_components = 0
    event_component_rows = []
    image_offset = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            evidence = forward_mshnet_evidence(model, images, detach=True)
            z_full = evidence["z_base"]
            contributions = evidence["contributions"]
            component_metric.update(z_full, targets)
            probabilities = torch.sigmoid(z_full)
            target_binary = targets > 0.5
            for index, threshold in enumerate(thresholds):
                prediction_at_threshold = probabilities > threshold
                intersections[index] += int(
                    (prediction_at_threshold & target_binary).sum()
                )
                unions[index] += int(
                    (prediction_at_threshold | target_binary).sum()
                )
            safe = build_safe_background(targets, args.safe_kernel).bool()
            positive = z_full > 0.0
            safe_positive = positive & safe
            responsible = (
                positive.expand_as(contributions)
                & ((z_full - contributions) <= 0.0)
                & safe.expand_as(contributions)
            )
            responsible_union = responsible.any(dim=1, keepdim=True)

            prediction_np = positive[:, 0].detach().cpu().numpy()
            target_np = target_binary[:, 0].detach().cpu().numpy()
            responsible_np = responsible.detach().cpu().numpy()
            responsible_union_np = responsible_union[:, 0].detach().cpu().numpy()
            for batch_index in range(images.shape[0]):
                component_match = match_connected_components(
                    prediction_np[batch_index], target_np[batch_index]
                )
                matched_indices = {
                    prediction_index
                    for _, prediction_index, _ in component_match.matches
                }
                false_indices = set(component_match.unmatched_prediction_indices)
                predicted_components += len(component_match.prediction_regions)
                false_alarm_components += len(false_indices)
                matched_components += len(matched_indices)
                for component_index, region in enumerate(
                    component_match.prediction_regions
                ):
                    component_mask = (
                        component_match.prediction_label_map == region.label
                    )
                    event_pixels = int(
                        responsible_union_np[batch_index][component_mask].sum()
                    )
                    event_count = int(
                        responsible_np[batch_index, :, component_mask].sum()
                    )
                    if event_pixels == 0:
                        continue
                    is_false_alarm = component_index in false_indices
                    if is_false_alarm:
                        false_alarm_components_with_responsibility += 1
                        responsible_pixels_in_false_alarm_components += event_pixels
                        responsible_events_in_false_alarm_components += event_count
                    else:
                        matched_components_with_responsibility += 1
                        responsible_pixels_in_matched_components += event_pixels
                        responsible_events_in_matched_components += event_count
                    event_component_rows.append(
                        {
                            "image_index": image_offset + batch_index,
                            "image_name": dataset.names[image_offset + batch_index],
                            "component_index": component_index,
                            "is_false_alarm": is_false_alarm,
                            "area": int(region.area),
                            "centroid_yx": [float(value) for value in region.centroid],
                            "responsible_pixels": event_pixels,
                            "responsible_events": event_count,
                        }
                    )
            image_offset += images.shape[0]

            total_pixels += targets.numel()
            safe_pixels += int(safe.sum())
            target_pixels += int((targets > 0.5).sum())
            predicted_pixels += int(positive.sum())
            safe_positive_pixels += int(safe_positive.sum())
            responsible_pixel_union += int(responsible_union.sum())
            active_images += int(responsible.flatten(1).any(dim=1).sum())
            reconstruction_error = max(
                reconstruction_error,
                float((evidence["z_reconstructed"] - z_full).abs().max()),
            )
            for scale in range(4):
                mask = responsible[:, scale : scale + 1]
                count = int(mask.sum())
                responsible_events[scale] += count
                if count:
                    responsible_contribution_sum[scale] += float(
                        contributions[:, scale : scale + 1][mask].sum()
                    )
                    responsible_full_margin_sum[scale] += float(z_full[mask].sum())
                    responsible_deleted_margin_sum[scale] += float(
                        (z_full - contributions[:, scale : scale + 1])[mask].sum()
                    )

    event_total = float(responsible_events.sum())
    false_alarm, detection_probability = component_metric.get()
    operating_curve = [
        {
            "probability_threshold": threshold,
            "iou": divide(intersections[index], unions[index]),
            "pd": float(detection_probability[index]),
            "fa_per_million": float(false_alarm[index] * 1e6),
        }
        for index, threshold in enumerate(thresholds)
    ]
    scale_rows = []
    for scale in range(4):
        count = float(responsible_events[scale])
        scale_rows.append(
            {
                "scale": scale,
                "responsible_events": int(count),
                "event_share": divide(count, event_total),
                "mean_contribution": divide(
                    float(responsible_contribution_sum[scale]), count
                ),
                "mean_full_margin": divide(
                    float(responsible_full_margin_sum[scale]), count
                ),
                "mean_deleted_margin": divide(
                    float(responsible_deleted_margin_sum[scale]), count
                ),
            }
        )

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "mode": args.mode,
        "images": len(dataset),
        "split_sha256": dataset.split_sha256,
        "safe_kernel": args.safe_kernel,
        "max_reconstruction_abs_error": reconstruction_error,
        "fusion_weight": model.final.weight.detach().cpu().flatten().tolist(),
        "fusion_bias": (
            model.final.bias.detach().cpu().flatten().tolist()
            if model.final.bias is not None else []
        ),
        "counts": {
            "pixels": total_pixels,
            "safe_pixels": safe_pixels,
            "target_pixels": target_pixels,
            "predicted_positive_pixels": predicted_pixels,
            "safe_positive_pixels": safe_positive_pixels,
            "responsible_pixel_union": responsible_pixel_union,
            "responsible_events": int(event_total),
            "active_images": active_images,
        },
        "ratios": {
            "safe_positive_per_safe_pixel": divide(
                safe_positive_pixels, safe_pixels
            ),
            "responsible_pixel_per_safe_positive": divide(
                responsible_pixel_union, safe_positive_pixels
            ),
            "events_per_responsible_pixel": divide(
                event_total, responsible_pixel_union
            ),
            "active_image_ratio": divide(active_images, len(dataset)),
        },
        "operating_curve": operating_curve,
        "scales": scale_rows,
        "component_linkage": {
            "predicted_components": predicted_components,
            "false_alarm_components": false_alarm_components,
            "matched_components": matched_components,
            "false_alarm_components_with_responsibility": (
                false_alarm_components_with_responsibility
            ),
            "matched_components_with_responsibility": (
                matched_components_with_responsibility
            ),
            "responsible_pixels_in_false_alarm_components": (
                responsible_pixels_in_false_alarm_components
            ),
            "responsible_pixels_in_matched_components": (
                responsible_pixels_in_matched_components
            ),
            "responsible_events_in_false_alarm_components": (
                responsible_events_in_false_alarm_components
            ),
            "responsible_events_in_matched_components": (
                responsible_events_in_matched_components
            ),
            "false_alarm_component_event_coverage": divide(
                false_alarm_components_with_responsibility,
                false_alarm_components,
            ),
            "responsible_pixel_false_alarm_precision": divide(
                responsible_pixels_in_false_alarm_components,
                responsible_pixel_union,
            ),
            "event_components": event_component_rows,
        },
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
