#!/usr/bin/env python3
"""Screen front-stage evidence families on frozen MSHNet components.

This is a correlational mechanism screen on a declared development benchmark,
not a model result.  It compares fixed, parameter-free statistics before any
candidate architecture is trained so that unsupported module construction can
be rejected cheaply.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

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
from model.jet_coherent_sufficient_lift import JetCoherentSufficientPool2d
from model.relative_trace_free_energy_lift import RelativeTraceFreeEnergyPool2d
from tools.audit_csl_stage0 import load_state, region_scores, sha256
from tools.audit_mshnet_stage_component_trace import distribution, probability_auc
from utils.data import IRSTD_Dataset
from utils.metric import match_connected_components


def _depthwise_filter(value: Tensor, kernel: Tensor) -> Tensor:
    channels = value.shape[1]
    kernel = kernel.to(device=value.device, dtype=value.dtype)
    kernel = kernel.view(1, 1, *kernel.shape).repeat(channels, 1, 1, 1)
    padded = F.pad(value, (1, 1, 1, 1), mode="replicate")
    return F.conv2d(padded, kernel, groups=channels)


def front_statistic_maps(
    image: Tensor,
    e0: Tensor,
    output_size: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Return fixed radiometric, diffusion, and differential evidence maps."""

    if image.ndim != 4 or e0.ndim != 4:
        raise ValueError("image and e0 must be BCHW tensors")
    binomial = e0.new_tensor(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]
    ) / 16.0
    sobel_x = e0.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ) / 8.0
    sobel_y = sobel_x.t().contiguous()
    dxx_kernel = e0.new_tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]]
    )
    dyy_kernel = dxx_kernel.t().contiguous()
    dxy_kernel = e0.new_tensor(
        [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]
    ) / 4.0
    eps = torch.finfo(e0.dtype).eps

    feature_energy = e0.square().mean(dim=1, keepdim=True).sqrt()
    diffused = _depthwise_filter(e0, binomial)
    diffusion_residual = e0 - diffused
    diffused_energy = diffused.square().mean(dim=1, keepdim=True).sqrt()
    residual_energy = diffusion_residual.square().mean(dim=1, keepdim=True).sqrt()
    smooth_energy = _depthwise_filter(feature_energy, binomial)

    grad_x = _depthwise_filter(feature_energy, sobel_x)
    grad_y = _depthwise_filter(feature_energy, sobel_y)
    gradient = torch.linalg.vector_norm(
        torch.stack([grad_x, grad_y], dim=0), dim=0
    )
    tangent_denominator = feature_energy + gradient
    tangent = torch.where(
        tangent_denominator > 0,
        gradient / tangent_denominator.clamp_min(torch.finfo(e0.dtype).tiny),
        torch.zeros_like(gradient),
    )
    dxx = _depthwise_filter(feature_energy, dxx_kernel)
    dyy = _depthwise_filter(feature_energy, dyy_kernel)
    dxy = _depthwise_filter(feature_energy, dxy_kernel)
    curvature = -(dxx + dyy)
    hessian_anisotropy = (
        ((dxx - dyy).square() + 4.0 * dxy.square() + eps).sqrt()
        / (dxx.abs() + dyy.abs() + eps)
    )
    blob_determinant = dxx * dyy - dxy.square()

    gray = image.mean(dim=1, keepdim=True)
    gray_low = _depthwise_filter(gray, binomial)
    gray_residual = gray - gray_low
    gray_scale = _depthwise_filter(gray.abs(), binomial)
    _, jcsl_state = JetCoherentSufficientPool2d()(e0, return_state=True)
    _, rtfe_state = RelativeTraceFreeEnergyPool2d()(e0, return_state=True)
    jcsl_factual_energy = jcsl_state["factual_maximum"].square().mean(
        dim=1, keepdim=True
    ).sqrt()
    jcsl_owned_jet_energy = jcsl_state["owned_jet"].square().mean(
        dim=1, keepdim=True
    ).sqrt()

    maps = {
        "feature_energy": feature_energy,
        "diffused_feature_energy": diffused_energy,
        "diffusion_residual_energy": residual_energy,
        "diffusion_survival_ratio": diffused_energy / (feature_energy + eps),
        "diffusion_residual_ratio": residual_energy / (feature_energy + eps),
        "energy_center_surround": feature_energy - smooth_energy,
        "energy_center_surround_ratio": (
            (feature_energy - smooth_energy) / (smooth_energy.abs() + eps)
        ),
        "energy_gradient_ratio": gradient / (feature_energy + eps),
        "snlt_tangent": tangent,
        "snlt_restricted_tangent": F.avg_pool2d(tangent, 2, 2),
        "positive_blob_curvature": curvature.clamp_min(0.0),
        "hessian_anisotropy": hessian_anisotropy,
        "positive_blob_determinant": blob_determinant.clamp_min(0.0),
        "gray_center_surround": gray_residual,
        "gray_abs_center_surround": gray_residual.abs(),
        "gray_center_surround_ratio": gray_residual / (gray_scale + eps),
        "jet_coherence": jcsl_state["coherence"],
        "bounded_jet_coherence": jcsl_state["bounded_coherence"],
        "jet_coherence_max_restricted": F.max_pool2d(
            jcsl_state["coherence"], 2, 2
        ),
        "jcsl_owned_coherence": jcsl_state["owned_coherence"].mean(
            dim=1, keepdim=True
        ),
        "jcsl_owned_jet_ratio": jcsl_owned_jet_energy
        / (jcsl_factual_energy + eps),
        "rtfe_coordinate": rtfe_state["coordinate"],
        "rtfe_max_restricted_coordinate": F.max_pool2d(
            rtfe_state["coordinate"], 2, 2
        ),
        "rtfe_restricted_coordinate": rtfe_state["restricted_coordinate"],
    }
    return {
        name: F.interpolate(value, size=output_size, mode="bilinear", align_corners=True)[
            0, 0
        ]
        .detach()
        .cpu()
        .numpy()
        for name, value in maps.items()
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
    dataset = IRSTD_Dataset(dataset_args, mode="test")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    model = MSHNet(3).to(device).eval()
    state, checkpoint_metadata = load_state(args.checkpoint)
    model.load_state_dict(state, strict=True)

    groups = (
        "matched_prediction",
        "false_prediction",
        "matched_target",
        "missed_target",
    )
    values: dict[str, dict[str, list[float]]] | None = None
    counts: Counter[str] = Counter()
    with torch.no_grad():
        for image, target in tqdm(loader, desc="front-evidence-screen"):
            image_device = image.to(device)
            e0 = model.encoder_0(model.conv_init(image_device))
            _, prediction = model(image_device, True)
            predicted = (prediction[0, 0].cpu().numpy() > 0.0).astype(np.int64)
            target_array = (target[0, 0].numpy() > 0.5).astype(np.int64)
            match = match_connected_components(
                predicted,
                target_array,
                max_centroid_distance=args.max_centroid_distance,
            )
            maps = front_statistic_maps(image_device, e0, predicted.shape)
            if values is None:
                values = {
                    group: {name: [] for name in maps} for group in groups
                }
            matched_predictions = {item[1] for item in match.matches}
            matched_targets = {item[0] for item in match.matches}
            for index, region in enumerate(match.prediction_regions):
                group = (
                    "matched_prediction"
                    if index in matched_predictions
                    else "false_prediction"
                )
                counts[group] += 1
                for name, score in region_scores(maps, region.coords).items():
                    values[group][name].append(score)
            for index, region in enumerate(match.target_regions):
                group = "matched_target" if index in matched_targets else "missed_target"
                counts[group] += 1
                for name, score in region_scores(maps, region.coords).items():
                    values[group][name].append(score)
    if values is None:
        raise RuntimeError("empty evidence screen")

    statistics = {}
    for name in values[groups[0]]:
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
            "fixed front-evidence mechanism screen on the declared NUAA development "
            "benchmark; no trained candidate and no confirmatory claim"
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata.get("epoch"),
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
