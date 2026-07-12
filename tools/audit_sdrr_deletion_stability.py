#!/usr/bin/env python3
"""Numerical and event-set audit for SDRR's fixed-logit scale deletion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines.mshnet_deterministic import MSHNet
from model.counterfactual_responsibility import build_safe_background
from model.mshnet_evidence_view import forward_mshnet_evidence
from model.scale_coalition_supervision import direct_zero_channel_coalitions
from utils.data import IRSTD_Dataset


DEFAULT_DELTAS = (0.0, 1e-6, 1e-4, 1e-3, 1e-2)
DEFAULT_THRESHOLDS = (-0.84729786, -0.40546511, 0.0, 0.40546511, 0.84729786)


def responsibility_mask(
    z_full: Tensor,
    deleted_logits: Tensor,
    safe_background: Tensor,
    *,
    threshold: float = 0.0,
    delta: float = 0.0,
) -> Tensor:
    if delta < 0.0:
        raise ValueError("delta must be non-negative")
    if z_full.ndim != 4 or z_full.shape[1] != 1:
        raise ValueError("z_full must have shape [B,1,H,W]")
    if deleted_logits.ndim != 4 or deleted_logits.shape[1] != 4:
        raise ValueError("deleted_logits must have shape [B,4,H,W]")
    if safe_background.shape != z_full.shape:
        raise ValueError("safe_background and z_full shapes differ")
    positive = z_full > threshold + delta
    if delta == 0.0:
        deleted_negative = deleted_logits <= threshold
    else:
        deleted_negative = deleted_logits < threshold - delta
    return (
        positive.expand_as(deleted_logits)
        & deleted_negative
        & safe_background.bool().expand_as(deleted_logits)
    )


def event_set_counts(first: Tensor, second: Tensor) -> dict[str, int]:
    if first.shape != second.shape:
        raise ValueError("event masks must have identical shapes")
    first = first.bool()
    second = second.bool()
    return {
        "first": int(first.sum()),
        "second": int(second.sum()),
        "intersection": int((first & second).sum()),
        "union": int((first | second).sum()),
        "mismatch": int((first ^ second).sum()),
    }


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _merge_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _load_clean_state(model: MSHNet, checkpoint_path: Path) -> list[str]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload["net"] if isinstance(payload, dict) and "net" in payload else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    expected = set(model.state_dict())
    missing = expected - set(state)
    if missing:
        raise ValueError(f"checkpoint missing clean MSHNet keys: {sorted(missing)[:5]}")
    extras = sorted(set(state) - expected)
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    return extras


def _quantiles(values: list[Tensor]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = torch.cat(values).double().numpy()
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.quantile(array, 0.5)),
        "q95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--deltas", type=float, nargs="+", default=DEFAULT_DELTAS)
    parser.add_argument(
        "--decision-thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
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
    dataset = IRSTD_Dataset(args, args.mode)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = MSHNet(args.input_channels).to(device).eval()
    extra_checkpoint_keys = _load_clean_state(model, args.checkpoint)

    deltas = tuple(sorted(set(float(value) for value in args.deltas)))
    thresholds = tuple(float(value) for value in args.decision_thresholds)
    if 0.0 not in deltas:
        raise ValueError("delta sweep must include 0")
    if 0.0 not in thresholds:
        raise ValueError("decision threshold sweep must include 0")

    direct_error_max = 0.0
    direct_error_sum = 0.0
    direct_error_count = 0
    delta_counts: dict[float, dict[str, int]] = {delta: {} for delta in deltas}
    threshold_counts: dict[float, dict[str, int]] = {
        threshold: {} for threshold in thresholds
    }
    degree_histogram: Counter[int] = Counter()
    combination_histogram: Counter[int] = Counter()
    per_scale_events = [0, 0, 0, 0]
    margin_values: dict[str, list[Tensor]] = {
        "full_logit": [],
        "deleted_logit": [],
        "contribution": [],
        "robust_margin": [],
    }

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            evidence = forward_mshnet_evidence(model, images, detach=True)
            z_full = evidence["z_base"]
            algebraic = evidence["z_without_scale"]
            direct = direct_zero_channel_coalitions(
                evidence["scale_logits"], model.final
            )
            contributions = evidence["contributions"]
            safe = build_safe_background(targets, args.safe_kernel)

            error = (algebraic - direct).abs()
            direct_error_max = max(direct_error_max, float(error.max()))
            direct_error_sum += float(error.double().sum())
            direct_error_count += error.numel()

            base_event = responsibility_mask(z_full, algebraic, safe)
            direct_base_event = responsibility_mask(z_full, direct, safe)
            degree = base_event.sum(dim=1)
            for value in range(1, 5):
                degree_histogram[value] += int((degree == value).sum())
            code = sum(
                base_event[:, scale].to(torch.int64) * (1 << scale)
                for scale in range(4)
            )
            for value in range(1, 16):
                combination_histogram[value] += int((code == value).sum())
            for scale in range(4):
                per_scale_events[scale] += int(base_event[:, scale].sum())

            if bool(base_event.any()):
                expanded_full = z_full.expand_as(algebraic)
                margin_values["full_logit"].append(
                    expanded_full[base_event].detach().cpu()
                )
                margin_values["deleted_logit"].append(
                    algebraic[base_event].detach().cpu()
                )
                margin_values["contribution"].append(
                    contributions[base_event].detach().cpu()
                )
                margin_values["robust_margin"].append(
                    torch.minimum(
                        expanded_full[base_event], -algebraic[base_event]
                    ).detach().cpu()
                )

            for delta in deltas:
                algebraic_event = responsibility_mask(
                    z_full, algebraic, safe, delta=delta
                )
                direct_event = responsibility_mask(
                    z_full, direct, safe, delta=delta
                )
                _merge_counts(
                    delta_counts[delta],
                    event_set_counts(algebraic_event, direct_event),
                )

            for threshold in thresholds:
                event = responsibility_mask(
                    z_full, algebraic, safe, threshold=threshold
                )
                _merge_counts(
                    threshold_counts[threshold], event_set_counts(event, base_event)
                )

    delta_rows = []
    for delta in deltas:
        row = {"delta": delta, **delta_counts[delta]}
        row["jaccard"] = _ratio(row["intersection"], row["union"])
        row["mismatch_per_union"] = _ratio(row["mismatch"], row["union"])
        delta_rows.append(row)
    threshold_rows = []
    for threshold in thresholds:
        row = {"logit_threshold": threshold, **threshold_counts[threshold]}
        row["jaccard_vs_zero"] = _ratio(row["intersection"], row["union"])
        threshold_rows.append(row)

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "mode": args.mode,
        "images": len(dataset),
        "split_sha256": dataset.split_sha256,
        "model_variant": "clean deterministic-backward MSHNet",
        "extra_checkpoint_keys_removed": extra_checkpoint_keys,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "safe_kernel": args.safe_kernel,
        "direct_zero_channel_error": {
            "max_abs": direct_error_max,
            "mean_abs": _ratio(direct_error_sum, direct_error_count),
            "values": direct_error_count,
        },
        "delta_sweep": delta_rows,
        "threshold_sweep": threshold_rows,
        "responsibility_degree_histogram": {
            str(key): degree_histogram[key] for key in range(1, 5)
        },
        "scale_combination_histogram": {
            format(key, "04b"): combination_histogram[key]
            for key in range(1, 16)
            if combination_histogram[key]
        },
        "per_scale_events": per_scale_events,
        "event_anatomy": {
            key: _quantiles(values) for key, values in margin_values.items()
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
