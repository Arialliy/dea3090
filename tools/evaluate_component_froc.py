#!/usr/bin/env python3
"""Evaluate a fixed checkpoint under component-level low-FPPI budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.counterfactual_conflict_diffusion import CCFDMSHNet
from model.support_persistence_transport import SupportPersistenceMSHNet
from utils.component_froc import ComponentFROC, DEFAULT_COMPONENT_BUDGETS
from utils.data import IRSTD_Dataset


MODEL_VARIANTS = {
    "deterministic": DeterministicMSHNet,
    "ccfd": CCFDMSHNet,
    "spt0": SupportPersistenceMSHNet,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    state = payload.get("net") if isinstance(payload, dict) and "net" in payload else payload
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a non-empty state_dict")
    if all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("checkpoint state_dict contains non-tensor values")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(MODEL_VARIANTS), required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--test-split-file", required=True)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-thresholds", type=int, default=51)
    parser.add_argument(
        "--threshold-space", choices=("probability", "logit"), default="logit"
    )
    parser.add_argument("--min-logit", type=float, default=-20.0)
    parser.add_argument("--max-logit", type=float, default=160.0)
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=DEFAULT_COMPONENT_BUDGETS,
    )
    parser.add_argument("--max-centroid-distance", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "component FROC keeps batch-size=1 to match the public evaluation numerics"
        )
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
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = MODEL_VARIANTS[args.variant](3).to(device).eval()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = extract_state_dict(payload)
    model.load_state_dict(state, strict=True)
    if args.threshold_space == "logit":
        if not args.min_logit < args.max_logit:
            raise ValueError("--min-logit must be smaller than --max-logit")
        thresholds = torch.linspace(
            args.min_logit, args.max_logit, args.num_thresholds,
            dtype=torch.float64,
        ).tolist()
    else:
        thresholds = None
    metric = ComponentFROC(
        thresholds=thresholds,
        num_thresholds=args.num_thresholds,
        max_centroid_distance=args.max_centroid_distance,
        threshold_space=args.threshold_space,
    )
    with torch.no_grad():
        for image, mask in tqdm(loader, desc="component-FROC"):
            _, prediction = model(image.to(device), True)
            metric.update(prediction.cpu(), mask)

    curve = metric.get_curve()
    budget_points = metric.at_budgets(args.budgets)
    checkpoint_meta = payload if isinstance(payload, dict) else {}
    result = {
        "metric": "component_froc",
        "false_alarm_unit": "unmatched_prediction_components_per_image",
        "detection_unit": "matched_target_instances_over_target_instances",
        "variant": args.variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_iou": checkpoint_meta.get("iou"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "test_split": str(Path(dataset.list_dir).resolve()),
        "test_split_sha256_normalized": dataset.split_sha256,
        "num_images": curve.num_images,
        "num_targets": curve.num_targets,
        "num_thresholds": len(curve.thresholds),
        "threshold_space": curve.threshold_space,
        "logit_range": (
            [args.min_logit, args.max_logit]
            if args.threshold_space == "logit"
            else None
        ),
        "max_centroid_distance": args.max_centroid_distance,
        "budgets": [
            {
                "budget_fppi": point.budget,
                "detection_probability": point.detection_probability,
                "threshold": point.threshold,
                "achieved_fppi": point.achieved_fppi,
            }
            for point in budget_points
        ],
        "mean_low_budget_detection": metric.mean_low_budget_detection(args.budgets),
        "curve": {
            "thresholds": curve.thresholds.tolist(),
            "detection_probability": curve.detection_probability.tolist(),
            "fppi": curve.false_positive_components_per_image.tolist(),
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
