#!/usr/bin/env python3
"""Audit SDRR eligibility on UIUNet's native six-side linear fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.basicirstd_root.resolve()
    sys.path.insert(0, str(root))
    from dataset import TestSetLoader  # type: ignore[import-not-found]
    from net import Net  # type: ignore[import-not-found]

    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    net = Net("UIUNet", mode="train").to(device)
    net.load_state_dict(checkpoint["state_dict"], strict=True)
    net.eval()

    capture: dict[str, torch.Tensor] = {}

    def capture_input(_module: object, inputs: tuple[torch.Tensor, ...]) -> None:
        capture["input"] = inputs[0]

    def capture_output(
        _module: object,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        capture["output"] = output

    layer = net.model.outconv
    input_handle = layer.register_forward_pre_hook(capture_input)
    output_handle = layer.register_forward_hook(capture_output)
    dataset = TestSetLoader(
        str(args.dataset_root.resolve()), args.dataset, args.dataset
    )
    weights = layer.weight.reshape(1, layer.in_channels, 1, 1)
    bias = layer.bias.reshape(1, 1, 1, 1)
    per_scale = torch.zeros(layer.in_channels, dtype=torch.long)
    event_count = 0
    unique_pixel_count = 0
    active_image_count = 0
    reconstruction_max_error = 0.0
    minimum_full_positive_margin: float | None = None
    minimum_deleted_nonpositive_margin: float | None = None

    try:
        with torch.no_grad():
            for index in range(len(dataset)):
                image, mask, _size, _name = dataset[index]
                net(image.unsqueeze(0).to(device))
                fusion_input = capture["input"]
                native_z = capture["output"]
                contributions = fusion_input * weights
                reconstructed_z = bias + contributions.sum(dim=1, keepdim=True)
                reconstruction_max_error = max(
                    reconstruction_max_error,
                    float((native_z - reconstructed_z).abs().max()),
                )
                safe_background = F.max_pool2d(
                    (mask.unsqueeze(0).to(device) > 0.5).float(),
                    kernel_size=args.safe_kernel,
                    stride=1,
                    padding=args.safe_kernel // 2,
                ) < 0.5
                events = (
                    (native_z > 0.0)
                    & ((native_z - contributions) <= 0.0)
                    & safe_background.expand_as(contributions)
                )
                image_events = int(events.sum())
                if image_events > 0:
                    expanded_z = native_z.expand_as(contributions)
                    full_margin = float(expanded_z[events].min())
                    deleted_margin = float(
                        (-(expanded_z - contributions)[events]).min()
                    )
                    minimum_full_positive_margin = (
                        full_margin
                        if minimum_full_positive_margin is None
                        else min(minimum_full_positive_margin, full_margin)
                    )
                    minimum_deleted_nonpositive_margin = (
                        deleted_margin
                        if minimum_deleted_nonpositive_margin is None
                        else min(
                            minimum_deleted_nonpositive_margin, deleted_margin
                        )
                    )
                event_count += image_events
                unique_pixel_count += int(events.any(dim=1).sum())
                active_image_count += int(image_events > 0)
                per_scale += events.sum(dim=(0, 2, 3)).cpu()
    finally:
        input_handle.remove()
        output_handle.remove()

    result = {
        "backbone": "UIUNet",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root.resolve()),
        "safe_kernel": int(args.safe_kernel),
        "fusion_input_channels": int(layer.in_channels),
        "new_trainable_parameters": 0,
        "images": len(dataset),
        "active_images": active_image_count,
        "events": event_count,
        "unique_pixels": unique_pixel_count,
        "events_per_scale": per_scale.tolist(),
        "max_abs_native_vs_reconstructed_logit": reconstruction_max_error,
        "minimum_event_full_positive_margin": minimum_full_positive_margin,
        "minimum_event_deleted_nonpositive_margin": minimum_deleted_nonpositive_margin,
        "fusion_weights": weights.detach().cpu().flatten().tolist(),
        "fusion_bias": float(bias.detach().cpu()),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basicirstd-root", type=Path, default=Path("/home/md0/ly/BasicIRSTD")
    )
    parser.add_argument("--dataset", default="NUAA-SIRST")
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/md0/ly/DEA/datasets")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--safe-kernel", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.safe_kernel <= 0 or args.safe_kernel % 2 == 0:
        parser.error("--safe-kernel must be a positive odd integer")
    print(json.dumps(run_audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
