#!/usr/bin/env python3
"""Evaluate a fixed checkpoint under component-level low-FPPI budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.counterfactual_conflict_diffusion import CCFDMSHNet
from model.birth_constrained_scale_filtration import (
    BirthConstrainedScaleFiltrationMSHNet,
)
from model.support_persistence_transport import SupportPersistenceMSHNet
from model.mshnet_evidence_view import forward_mshnet_evidence
from utils.component_froc import ComponentFROC, DEFAULT_COMPONENT_BUDGETS
from utils.data import IRSTD_Dataset
from utils.scale_subset import reconstruct_scale_subset


MODEL_VARIANTS = {
    "bcsf": BirthConstrainedScaleFiltrationMSHNet,
    "deterministic": DeterministicMSHNet,
    "ccfd": CCFDMSHNet,
    "spt0": SupportPersistenceMSHNet,
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


def _validated_sha256(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    normalized = value.lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    return normalized


def validate_checkpoint_identity(
    payload: Any,
    *,
    requested_variant: str,
    checkpoint_sha256: str,
    test_split_sha256: str,
    expected_checkpoint_sha256: str | None = None,
    expected_test_split_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind CLI forward semantics and data to a checkpoint artifact.

    Parameter-identical MSHNet variants can strict-load one another's state
    dicts, so strict=True is not an architecture identity check.  Modern
    checkpoints are verified through method_meta.  A legacy checkpoint without
    sufficient metadata remains usable only when the caller supplies explicit
    artifact and split hashes; the output records that weaker validation mode.
    """

    if requested_variant not in MODEL_VARIANTS:
        raise ValueError(f"unsupported requested variant: {requested_variant}")
    checkpoint_sha256 = _validated_sha256(
        checkpoint_sha256, label="checkpoint SHA-256"
    ) or ""
    test_split_sha256 = _validated_sha256(
        test_split_sha256, label="test split SHA-256"
    ) or ""
    expected_checkpoint_sha256 = _validated_sha256(
        expected_checkpoint_sha256,
        label="--expected-checkpoint-sha256",
    )
    expected_test_split_sha256 = _validated_sha256(
        expected_test_split_sha256,
        label="--expected-test-split-sha256",
    )
    if (
        expected_checkpoint_sha256 is not None
        and expected_checkpoint_sha256 != checkpoint_sha256
    ):
        raise ValueError(
            "checkpoint SHA-256 does not match --expected-checkpoint-sha256"
        )
    if (
        expected_test_split_sha256 is not None
        and expected_test_split_sha256 != test_split_sha256
    ):
        raise ValueError(
            "test split SHA-256 does not match --expected-test-split-sha256"
        )

    metadata = payload.get("method_meta") if isinstance(payload, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    metadata_variant = metadata.get("mshnet_variant")
    metadata_test_hash = metadata.get("test_split_sha256")
    warnings: list[str] = []

    if metadata_variant is not None and metadata_variant != requested_variant:
        raise ValueError(
            "checkpoint variant mismatch: method_meta has "
            f"{metadata_variant!r}, CLI requested {requested_variant!r}"
        )
    if metadata_variant is None:
        if expected_checkpoint_sha256 is None:
            raise ValueError(
                "checkpoint has no mshnet_variant metadata; legacy evaluation "
                "requires --expected-checkpoint-sha256 in addition to the "
                "explicit --variant"
            )
        warnings.append(
            "legacy checkpoint variant is asserted by --variant and bound only "
            "to the explicitly supplied checkpoint hash"
        )

    if metadata_test_hash is not None:
        metadata_test_hash = _validated_sha256(
            str(metadata_test_hash), label="checkpoint test_split_sha256"
        )
        if metadata_test_hash != test_split_sha256:
            raise ValueError(
                "checkpoint/test split mismatch: method_meta test_split_sha256 "
                f"{metadata_test_hash} != loaded split {test_split_sha256}"
            )
    else:
        if expected_test_split_sha256 is None:
            raise ValueError(
                "checkpoint has no test_split_sha256 metadata; legacy evaluation "
                "requires --expected-test-split-sha256"
            )
        warnings.append(
            "legacy checkpoint split identity is bound only to the explicitly "
            "supplied normalized split hash"
        )

    protocol = metadata.get("evaluation_protocol")
    if protocol is not None and protocol != "official_train_test":
        raise ValueError(
            "component FROC formal evaluation requires an official_train_test "
            f"checkpoint, got {protocol!r}"
        )
    return {
        "mode": "method_meta" if not warnings else "legacy_explicit_binding",
        "requested_variant": requested_variant,
        "metadata_variant": metadata_variant,
        "metadata_test_split_sha256": metadata_test_hash,
        "checkpoint_sha256_verified": expected_checkpoint_sha256 is not None,
        "test_split_sha256_verified": (
            metadata_test_hash is not None or expected_test_split_sha256 is not None
        ),
        "warnings": warnings,
    }


def threshold_grid_metadata(thresholds: Any, threshold_space: str) -> dict[str, Any]:
    values = np.asarray(thresholds, dtype=np.float64)
    differences = np.diff(values)
    uniform = bool(
        differences.size > 0
        and np.allclose(differences, differences[0], rtol=0.0, atol=1e-12)
    )
    canonical = values.astype("<f8", copy=False).tobytes()
    return {
        "space": threshold_space,
        "construction": "linear_inclusive" if uniform else "explicit",
        "count": int(values.size),
        "minimum": float(values[0]),
        "maximum": float(values[-1]),
        "uniform_step": float(differences[0]) if uniform else None,
        "float64_le_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=sorted(MODEL_VARIANTS),
        required=True,
        help=(
            "Expected checkpoint forward semantics; verified against "
            "checkpoint method_meta when available."
        ),
    )
    parser.add_argument(
        "--expected-checkpoint-sha256",
        help=(
            "Optional artifact binding for modern checkpoints; mandatory when "
            "legacy checkpoint metadata cannot verify the requested variant."
        ),
    )
    parser.add_argument(
        "--expected-test-split-sha256",
        help=(
            "Expected normalized test-ID hash; mandatory when absent from "
            "legacy checkpoint metadata."
        ),
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--test-split-file", required=True)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-thresholds", type=int, default=181)
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
    parser.add_argument(
        "--kept-scales",
        type=int,
        nargs="+",
        choices=(0, 1, 2, 3),
        help=(
            "Audit-only deterministic MSHNet scale subset. The deployed model "
            "is unchanged; exact native fusion contributions are reconstructed."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "component FROC keeps batch-size=1 to match the public evaluation numerics"
        )
    if args.kept_scales is not None:
        if args.variant != "deterministic":
            raise ValueError("--kept-scales is only valid for deterministic MSHNet")
        if len(args.kept_scales) != len(set(args.kept_scales)):
            raise ValueError("--kept-scales must not contain duplicates")
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

    checkpoint_digest = sha256(args.checkpoint)
    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_validation = validate_checkpoint_identity(
        payload,
        requested_variant=args.variant,
        checkpoint_sha256=checkpoint_digest,
        test_split_sha256=dataset.split_sha256,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_test_split_sha256=args.expected_test_split_sha256,
    )
    for warning in checkpoint_validation["warnings"]:
        print(f"[legacy checkpoint warning] {warning}", file=sys.stderr)

    model = MODEL_VARIANTS[args.variant](3).to(device).eval()
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
            image = image.to(device)
            if args.kept_scales is None:
                _, prediction = model(image, True)
            else:
                evidence = forward_mshnet_evidence(model, image, detach=True)
                subset = sum(1 << scale for scale in args.kept_scales)
                prediction = reconstruct_scale_subset(
                    evidence["contributions"],
                    evidence["fusion_bias"],
                    subset,
                    z_base=evidence["z_base"],
                )
            if not bool(torch.isfinite(prediction).all().detach().cpu()):
                raise ValueError("model produced non-finite logits")
            metric.update(prediction.cpu(), mask)

    curve = metric.get_curve()
    if (
        curve.detection_probability[-1] != 0.0
        or curve.false_positive_components_per_image[-1] != 0.0
    ):
        raise ValueError(
            "highest threshold does not produce an empty prediction; expand "
            "the threshold grid before reporting component FROC"
        )
    budget_points = metric.at_budgets(args.budgets)
    checkpoint_meta = payload if isinstance(payload, dict) else {}
    grid_metadata = threshold_grid_metadata(curve.thresholds, curve.threshold_space)
    result = {
        "metric": "component_froc",
        "false_alarm_unit": "unmatched_prediction_components_per_image",
        "detection_unit": "matched_target_instances_over_target_instances",
        "variant": args.variant,
        "kept_scales": (
            sorted(args.kept_scales) if args.kept_scales is not None else None
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_digest,
        "checkpoint_validation": checkpoint_validation,
        "checkpoint_epoch": checkpoint_meta.get("epoch"),
        "checkpoint_iou": checkpoint_meta.get("iou"),
        "checkpoint_iou_scope": (
            "source checkpoint metadata; not recomputed for kept-scales"
            if args.kept_scales is not None
            else "source checkpoint metadata; not recomputed by component-FROC"
        ),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "test_split": str(Path(dataset.list_dir).resolve()),
        "test_split_sha256_normalized": dataset.split_sha256,
        "base_size": args.base_size,
        "crop_size": args.crop_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "num_images": curve.num_images,
        "num_targets": curve.num_targets,
        "num_thresholds": len(curve.thresholds),
        "threshold_space": curve.threshold_space,
        "threshold_grid": grid_metadata,
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
