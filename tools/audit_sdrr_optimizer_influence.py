#!/usr/bin/env python3
"""One-step Adagrad influence audit for SDRR and attribution controls."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.MSHNet import MSHNet as WorkbenchMSHNet
from model.baselines.mshnet_deterministic import MSHNet as CleanMSHNet
from model.counterfactual_responsibility import (
    counterfactual_responsibility_suppression,
    magnitude_matched_nonpivotal_suppression,
    same_pixel_random_scale_suppression,
)
from model.loss import SLSIoULoss
from model.scale_coalition_supervision import leave_one_scale_out_coalitions
from tools.optimizer_counterfactual import optimizer_counterfactual
from utils.data import IRSTD_Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", default="datasets/NUAA-SIRST")
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
    parser.add_argument("--epoch", type=int, default=-1)
    parser.add_argument("--warm-epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--sdrr-lambda", type=float, default=0.05)
    parser.add_argument("--safe-kernel", type=int, default=15)
    parser.add_argument("--max-train-batches", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def canonical_loss(
    model: nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    loss_function: SLSIoULoss,
    *,
    warm_epoch: int,
    epoch: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    masks, prediction = model(images, True)
    total = loss_function(prediction, targets, warm_epoch, epoch)
    scale_target = targets
    for index, mask in enumerate(masks):
        if index > 0:
            scale_target = torch.nn.functional.max_pool2d(scale_target, 2, 2)
        total = total + loss_function(mask, scale_target, warm_epoch, epoch)
    return total / (len(masks) + 1), tuple(masks), prediction


def _scalar_logs(logs: dict[str, torch.Tensor]) -> dict[str, float]:
    return {
        key: float(value.detach().float().mean().cpu())
        for key, value in logs.items()
        if torch.is_tensor(value)
    }


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "net" not in payload or "optimizer" not in payload:
        raise ValueError("checkpoint must contain net and optimizer states")
    state = payload["net"]
    has_workbench_head = any(key.startswith("decidability_head.") for key in state)
    model: nn.Module = (
        WorkbenchMSHNet(3) if has_workbench_head else CleanMSHNet(3)
    )
    model.load_state_dict(state, strict=True)
    model.to(device)
    optimizer = Adagrad(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = args.lr

    epoch = args.epoch if args.epoch >= 0 else int(payload.get("epoch", 0)) + 1
    train_dataset = IRSTD_Dataset(args, "train")
    val_dataset = IRSTD_Dataset(args, "val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loss_function = SLSIoULoss()

    selected_train: tuple[torch.Tensor, torch.Tensor] | None = None
    selected_batch_index = -1
    selection_logs: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, targets) in enumerate(train_loader):
            if batch_index >= args.max_train_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            _, masks, prediction = canonical_loss(
                model,
                images,
                targets,
                loss_function,
                warm_epoch=args.warm_epoch,
                epoch=epoch,
            )
            coalition = leave_one_scale_out_coalitions(
                masks, prediction, model.final
            )
            _, logs = counterfactual_responsibility_suppression(
                prediction,
                coalition["contributions"],
                targets,
                safe_kernel=args.safe_kernel,
            )
            if float(logs["responsible_count"]) > 0:
                selected_train = (images.detach(), targets.detach())
                selected_batch_index = batch_index
                selection_logs = _scalar_logs(logs)
                break
    if selected_train is None:
        raise RuntimeError(
            "no active SDRR batch found within --max-train-batches"
        )
    probe_images, probe_targets = next(iter(val_loader))
    probe_images = probe_images.to(device)
    probe_targets = probe_targets.to(device)
    inference_images, inference_targets = selected_train
    # Tensors created under inference_mode cannot later be saved for backward.
    # Copy them outside that context while preserving exact values/device.
    train_images = torch.empty_like(inference_images)
    train_targets = torch.empty_like(inference_targets)
    train_images.copy_(inference_images)
    train_targets.copy_(inference_targets)

    branch_logs: dict[str, dict[str, float]] = {}

    def base_closure(current: nn.Module) -> torch.Tensor:
        current.train()
        loss, _, _ = canonical_loss(
            current,
            train_images,
            train_targets,
            loss_function,
            warm_epoch=args.warm_epoch,
            epoch=epoch,
        )
        return loss

    def probe_closure(current: nn.Module) -> torch.Tensor:
        current.eval()
        loss, _, _ = canonical_loss(
            current,
            probe_images,
            probe_targets,
            loss_function,
            warm_epoch=args.warm_epoch,
            epoch=epoch,
        )
        return loss

    regularizers: dict[
        str,
        Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]],
    ] = {
        "sdrr": counterfactual_responsibility_suppression,
        "m3_magnitude_nonpivotal": magnitude_matched_nonpivotal_suppression,
        "m4_same_pixel_random_scale": same_pixel_random_scale_suppression,
    }
    results = {
        "canonical_identity": dataclasses.asdict(
            optimizer_counterfactual(
                model,
                optimizer,
                with_edge_loss=base_closure,
                without_edge_loss=base_closure,
                probe_loss=probe_closure,
            )
        )
    }
    for name, regularizer in regularizers.items():
        def with_regularizer(
            current: nn.Module,
            *,
            _name: str = name,
            _regularizer: Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]] = regularizer,
        ) -> torch.Tensor:
            current.train()
            canonical, masks, prediction = canonical_loss(
                current,
                train_images,
                train_targets,
                loss_function,
                warm_epoch=args.warm_epoch,
                epoch=epoch,
            )
            coalition = leave_one_scale_out_coalitions(
                masks, prediction, current.final
            )
            kwargs = {
                "safe_kernel": args.safe_kernel,
                "normalization": "event",
            }
            if _name == "m4_same_pixel_random_scale":
                kwargs["salt"] = args.seed + epoch * 9176 + selected_batch_index
            regularization, logs = _regularizer(
                prediction,
                coalition["contributions"],
                train_targets,
                **kwargs,
            )
            branch_logs[_name] = _scalar_logs(logs)
            active_count = logs.get(
                "control_selected_count", logs["responsible_count"]
            )
            if bool((active_count == 0).detach().cpu()):
                return canonical
            return canonical + args.sdrr_lambda * regularization

        result = optimizer_counterfactual(
            model,
            optimizer,
            with_edge_loss=with_regularizer,
            without_edge_loss=base_closure,
            probe_loss=probe_closure,
        )
        results[name] = dataclasses.asdict(result)

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "audit_epoch": epoch,
        "model_variant": "workbench" if has_workbench_head else "deterministic_clean",
        "train_split_sha256": train_dataset.split_sha256,
        "val_split_sha256": val_dataset.split_sha256,
        "selected_train_batch_index": selected_batch_index,
        "selection_logs_eval_mode": selection_logs,
        "branch_logs_train_mode": branch_logs,
        "comparisons_against_canonical": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
