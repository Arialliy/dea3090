#!/usr/bin/env python3
"""Ten-epoch gain-only rejection gate for front-stage JCPT.

This is explicitly exploratory: it starts from the official baseline's own
best checkpoint, freezes every canonical parameter and all BN statistics, and
trains only the 16-dimensional jet potential.  It may reject JCPT cheaply but
cannot establish the final model or replace a paired 400-epoch best-vs-best run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adagrad
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.baselines.mshnet_deterministic import MSHNet
from model.jet_coherent_potential_transport import (
    JetCoherentPotentialTransportMSHNet,
)
from model.loss import SLSIoULoss
from utils.component_froc import ComponentFROC, DEFAULT_COMPONENT_BUDGETS
from utils.data import IRSTD_Dataset
from utils.metric import PD_FA, mIoU, match_connected_components


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def extract_baseline_state(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict) or "net" not in payload:
        raise ValueError("gate requires a full baseline checkpoint with metadata")
    metadata = payload.get("method_meta", {})
    if metadata.get("mshnet_variant") != "deterministic":
        raise ValueError("gate checkpoint is not deterministic baseline MSHNet")
    state = payload["net"]
    if all(key.startswith("module.") for key in state):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--train-split-file", required=True)
    parser.add_argument("--test-split-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def deep_supervision_loss(
    criterion: SLSIoULoss,
    masks: list[torch.Tensor],
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epoch: int,
) -> torch.Tensor:
    loss = criterion(prediction, target, 5, epoch)
    scale_target = target
    for index, side in enumerate(masks):
        if index > 0:
            scale_target = F.max_pool2d(scale_target, 2, 2)
        loss = loss + criterion(side, scale_target, 5, epoch)
    return loss / (len(masks) + 1)


def evaluate(
    candidate: JetCoherentPotentialTransportMSHNet,
    baseline: MSHNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    candidate.eval()
    baseline.eval()
    iou_metric = mIoU(1)
    pd_fa_metric = PD_FA(1, 10, 256)
    thresholds = np.linspace(-20.0, 160.0, 181).tolist()
    froc = ComponentFROC(thresholds=thresholds, threshold_space="logit")
    changed_pixels = 0
    matched_targets = 0
    false_components = 0
    target_count = 0
    with torch.no_grad():
        for image, target in tqdm(loader, desc="JCPT-gate-test"):
            image_device = image.to(device)
            target_device = target.to(device)
            _, prediction = candidate(image_device, True)
            _, baseline_prediction = baseline(image_device, True)
            changed_pixels += int(
                ((prediction > 0) != (baseline_prediction > 0)).sum().cpu()
            )
            iou_metric.update(prediction, target_device)
            pd_fa_metric.update(prediction.cpu(), target)
            froc.update(prediction.cpu(), target)
            predicted = (prediction[0, 0].cpu().numpy() > 0).astype(np.int64)
            truth = (target[0, 0].numpy() > 0.5).astype(np.int64)
            match = match_connected_components(predicted, truth)
            matched_targets += len(match.matches)
            target_count += len(match.target_regions)
            false_components += len(match.unmatched_prediction_indices)
    _, iou = iou_metric.get()
    false_alarm, detection_probability = pd_fa_metric.get()
    budget_points = froc.at_budgets(DEFAULT_COMPONENT_BUDGETS)
    return {
        "iou": float(iou),
        "pd": float(detection_probability[0]),
        "fa_per_million": float(false_alarm[0] * 1e6),
        "matched_targets": matched_targets,
        "target_count": target_count,
        "false_components": false_components,
        "fppi_at_logit_zero": false_components / len(loader.dataset),
        "binary_changed_pixels_vs_baseline": changed_pixels,
        "mean_low_budget_detection": froc.mean_low_budget_detection(
            DEFAULT_COMPONENT_BUDGETS
        ),
        "component_froc_budgets": [
            {
                "budget_fppi": point.budget,
                "detection_probability": point.detection_probability,
                "threshold": point.threshold,
                "achieved_fppi": point.achieved_fppi,
            }
            for point in budget_points
        ],
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = extract_baseline_state(payload)

    dataset_args = SimpleNamespace(
        dataset_dir=str(args.dataset_dir.resolve()),
        evaluation_protocol="official_train_test",
        train_split_file=args.train_split_file,
        val_split_file="",
        test_split_file=args.test_split_file,
        val_fraction=0.2,
        split_seed=0,
        seed=args.seed,
        crop_size=args.crop_size,
        base_size=args.base_size,
        return_instance_map=False,
    )
    trainset = IRSTD_Dataset(dataset_args, mode="train")
    testset = IRSTD_Dataset(dataset_args, mode="test")
    if set(trainset.names).intersection(testset.names):
        raise ValueError("train/test manifests overlap")
    train_generator = torch.Generator().manual_seed(args.seed)
    test_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    test_loader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=test_generator,
    )

    candidate = JetCoherentPotentialTransportMSHNet(3).to(device)
    candidate.load_canonical_state_dict(state)
    baseline = MSHNet(3).to(device).eval()
    baseline.load_state_dict(state, strict=True)
    for name, parameter in candidate.named_parameters():
        parameter.requires_grad_(name == "jet_potential")
    optimizer = Adagrad([candidate.jet_potential], lr=args.lr)
    criterion = SLSIoULoss()
    losses = []
    for local_epoch in range(args.epochs):
        # All canonical BN statistics remain exactly frozen during the gate.
        candidate.eval()
        epoch_losses = []
        for image, target in tqdm(
            train_loader, desc=f"JCPT-gate-train-{local_epoch:02d}"
        ):
            image = image.to(device)
            target = target.to(device)
            masks, prediction = candidate(image, True)
            loss = deep_supervision_loss(
                criterion,
                masks,
                prediction,
                target,
                epoch=400 + local_epoch,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(candidate.jet_potential.grad).all():
                raise FloatingPointError("non-finite jet-potential gradient")
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    metrics = evaluate(candidate, baseline, test_loader, device)
    gain = candidate.jet_potential.detach().cpu()
    baseline_iou = float(payload["iou"])
    baseline_pd = float(payload["pd"])
    baseline_fa = float(payload["fa"])
    gate = {
        "finite_nonzero_gain": bool(torch.isfinite(gain).all() and torch.count_nonzero(gain)),
        "changed_prediction": metrics["binary_changed_pixels_vs_baseline"] > 0,
        "matched_targets_at_least_baseline": metrics["matched_targets"] >= 249,
        "iou_within_0p002": metrics["iou"] >= baseline_iou - 0.002,
        "false_components_reduced": metrics["false_components"] < 15,
        "low_budget_edge_improved": any(
            point["detection_probability"] > reference
            for point, reference in zip(
                metrics["component_froc_budgets"][:2],
                (0.09125475285171103, 0.19391634980988592),
                strict=True,
            )
        ),
    }
    gate["allow_formal_training"] = bool(
        gate["finite_nonzero_gain"]
        and gate["changed_prediction"]
        and gate["matched_targets_at_least_baseline"]
        and gate["iou_within_0p002"]
        and (gate["false_components_reduced"] or gate["low_budget_edge_improved"])
    )
    report = {
        "scope": "gain-only exploratory rejection gate; not a formal model result",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "train_split_sha256": trainset.split_sha256,
        "test_split_sha256": testset.split_sha256,
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "trainable_parameters": int(candidate.jet_potential.numel()),
        "train_losses": losses,
        "jet_potential": {
            "values": gain.flatten().tolist(),
            "l1": float(gain.abs().sum()),
            "l2": float(gain.square().sum().sqrt()),
            "max_abs": float(gain.abs().max()),
        },
        "baseline_best": {
            "iou": baseline_iou,
            "pd": baseline_pd,
            "fa_per_million": baseline_fa,
        },
        "candidate": metrics,
        "gate": gate,
    }
    torch.save(
        {
            "net": candidate.state_dict(),
            "epoch": args.epochs - 1,
            "method_meta": {
                "method": "JCPT-gain-only-gate",
                "mshnet_variant": "jcpt0_gate",
                "scope": report["scope"],
            },
        },
        args.output_dir / "checkpoint_gate.pkl",
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "gate_report.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
