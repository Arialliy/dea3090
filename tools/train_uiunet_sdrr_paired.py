#!/usr/bin/env python3
"""Paired UIUNet baseline/SDRR training with one shared augmented batch stream."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def exact_state_identity(
    baseline: torch.nn.Module, candidate: torch.nn.Module
) -> tuple[bool, float, int]:
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    if tuple(baseline_state) != tuple(candidate_state):
        return False, float("inf"), -1
    maximum = 0.0
    mismatches = 0
    for key, baseline_tensor in baseline_state.items():
        candidate_tensor = candidate_state[key]
        if not torch.equal(baseline_tensor, candidate_tensor):
            mismatches += 1
            maximum = max(
                maximum,
                float((baseline_tensor - candidate_tensor).abs().max()),
            )
    return mismatches == 0, maximum, mismatches


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    pd_fa_class: type[Any],
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    intersection = 0
    union = 0
    pd_fa = pd_fa_class()
    with torch.no_grad():
        for image, mask, size, _name in loader:
            prediction = model(image.to(device))[0]
            height = int(size[0].item())
            width = int(size[1].item())
            prediction = prediction[:, :, :height, :width]
            mask = mask[:, :, :height, :width]
            binary = prediction > 0.5
            target = mask.to(device) > 0.5
            intersection += int((binary & target).sum())
            union += int((binary | target).sum())
            pd_fa.update(binary[0, 0].cpu(), target[0, 0].cpu(), (height, width))
    pd, fa = pd_fa.get()
    model.train(was_training)
    return {
        "iou": float(intersection / max(union, 1)),
        "pd": float(pd),
        "fa_per_million": float(fa * 1_000_000.0),
    }


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    baseline: torch.nn.Module,
    candidate: torch.nn.Module,
    baseline_optimizer: torch.optim.Optimizer,
    candidate_optimizer: torch.optim.Optimizer,
    baseline_scheduler: torch.optim.lr_scheduler.LRScheduler,
    candidate_scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "baseline_state_dict": baseline.state_dict(),
        "candidate_state_dict": candidate.state_dict(),
        "baseline_optimizer": baseline_optimizer.state_dict(),
        "candidate_optimizer": candidate_optimizer.state_dict(),
        "baseline_scheduler": baseline_scheduler.state_dict(),
        "candidate_scheduler": candidate_scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "loader_generator_state": loader_generator.get_state(),
        "method": "UIUNet-native-six-side-SDRR-paired",
        "args": vars(args),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basicirstd-root", type=Path, default=Path("/home/md0/ly/BasicIRSTD")
    )
    parser.add_argument("--dataset", default="NUAA-SIRST")
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/home/md0/ly/DEA/datasets")
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--sdrr-lambda", type=float, default=0.05)
    parser.add_argument("--sdrr-start-ratio", type=float, default=0.625)
    parser.add_argument("--sdrr-ramp-ratio", type=float, default=0.125)
    parser.add_argument("--safe-kernel", type=int, default=15)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--save-interval", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.safe_kernel <= 0 or args.safe_kernel % 2 == 0:
        parser.error("--safe-kernel must be a positive odd integer")
    if not 0 <= args.sdrr_start_ratio < 1:
        parser.error("--sdrr-start-ratio must be in [0,1)")
    if not 0 < args.sdrr_ramp_ratio <= 1:
        parser.error("--sdrr-ramp-ratio must be in (0,1]")
    if args.run_dir.exists() and any(args.run_dir.iterdir()) and args.resume is None:
        parser.error("--run-dir must be absent/empty unless --resume is provided")
    if args.resume is not None and not args.resume.is_file():
        parser.error("--resume must point to a checkpoint file")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.run_dir / "run_config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    root = args.basicirstd_root.resolve()
    sys.path.insert(0, str(root))
    from dataset import TestSetLoader, TrainSetLoader  # type: ignore[import-not-found]
    from loss import SoftIoULoss  # type: ignore[import-not-found]
    from metrics import PD_FA  # type: ignore[import-not-found]
    from model import UIUNet  # type: ignore[import-not-found]

    seed_everything(args.seed)
    device = torch.device(args.device)
    train_dataset = TrainSetLoader(
        str(args.dataset_root.resolve()), args.dataset, args.patch_size
    )
    test_dataset = TestSetLoader(
        str(args.dataset_root.resolve()), args.dataset, args.dataset
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0
    )

    baseline = UIUNet(mode="train").to(device)
    candidate = copy.deepcopy(baseline).to(device)
    criterion = SoftIoULoss()
    baseline_optimizer = torch.optim.Adam(baseline.parameters(), lr=args.lr)
    candidate_optimizer = torch.optim.Adam(candidate.parameters(), lr=args.lr)
    milestones = [epoch for epoch in (200, 300) if epoch < args.epochs]
    baseline_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        baseline_optimizer, milestones=milestones, gamma=0.1
    )
    candidate_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        candidate_optimizer, milestones=milestones, gamma=0.1
    )
    first_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        for key in (
            "dataset",
            "dataset_root",
            "seed",
            "batch_size",
            "patch_size",
            "lr",
            "sdrr_lambda",
            "sdrr_start_ratio",
            "sdrr_ramp_ratio",
            "safe_kernel",
        ):
            if checkpoint["args"][key] != getattr(args, key):
                raise ValueError(f"resume configuration mismatch for {key}")
        baseline.load_state_dict(checkpoint["baseline_state_dict"], strict=True)
        candidate.load_state_dict(checkpoint["candidate_state_dict"], strict=True)
        baseline_optimizer.load_state_dict(checkpoint["baseline_optimizer"])
        candidate_optimizer.load_state_dict(checkpoint["candidate_optimizer"])
        baseline_scheduler.load_state_dict(checkpoint["baseline_scheduler"])
        candidate_scheduler.load_state_dict(checkpoint["candidate_scheduler"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        generator.set_state(checkpoint["loader_generator_state"])
        first_epoch = int(checkpoint["epoch"]) + 1

    capture: dict[str, torch.Tensor] = {}

    def capture_input(_module: object, inputs: tuple[torch.Tensor, ...]) -> None:
        capture["input"] = inputs[0]

    def capture_output(
        _module: object,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        capture["output"] = output

    fusion = candidate.outconv
    input_handle = fusion.register_forward_pre_hook(capture_input)
    output_handle = fusion.register_forward_hook(capture_output)
    activation_start_epoch = int(round(args.sdrr_start_ratio * args.epochs))
    ramp_epochs = max(1, int(round(args.sdrr_ramp_ratio * args.epochs)))
    metrics_path = args.run_dir / "metrics.jsonl"

    try:
        for epoch in range(first_epoch, args.epochs):
            baseline.train()
            candidate.train()
            baseline_loss_sum = 0.0
            candidate_loss_sum = 0.0
            event_count = 0
            active_batches = 0
            ramp = max(
                0.0,
                min(1.0, float(epoch - activation_start_epoch) / ramp_epochs),
            )
            for image, target in train_loader:
                if image.shape[0] == 1:
                    continue
                image = image.to(device)
                target = target.to(device)

                baseline_prediction = baseline(image)
                baseline_loss = criterion(baseline_prediction, target)
                baseline_optimizer.zero_grad(set_to_none=True)
                baseline_loss.backward()
                baseline_optimizer.step()

                candidate_prediction = candidate(image)
                canonical_loss = criterion(candidate_prediction, target)
                candidate_loss = canonical_loss
                if ramp > 0.0:
                    native_z = capture["output"]
                    weights = fusion.weight.reshape(1, fusion.in_channels, 1, 1)
                    contributions = capture["input"] * weights
                    safe_background = F.max_pool2d(
                        (target > 0.5).float(),
                        kernel_size=args.safe_kernel,
                        stride=1,
                        padding=args.safe_kernel // 2,
                    ) < 0.5
                    with torch.no_grad():
                        responsibility = (
                            (native_z.detach() > 0.0)
                            & ((native_z.detach() - contributions.detach()) <= 0.0)
                            & safe_background.expand_as(contributions)
                        )
                    count = int(responsibility.sum())
                    if count > 0:
                        sdrr_loss = (
                            F.softplus(contributions) * responsibility
                        ).sum() / responsibility.sum()
                        candidate_loss = canonical_loss + (
                            args.sdrr_lambda * ramp * sdrr_loss
                        )
                        event_count += count
                        active_batches += 1
                candidate_optimizer.zero_grad(set_to_none=True)
                candidate_loss.backward()
                candidate_optimizer.step()
                baseline_loss_sum += float(baseline_loss.detach())
                candidate_loss_sum += float(candidate_loss.detach())

            baseline_scheduler.step()
            candidate_scheduler.step()
            record: dict[str, Any] = {
                "epoch": epoch,
                "ramp": ramp,
                "baseline_train_loss_sum": baseline_loss_sum,
                "candidate_train_loss_sum": candidate_loss_sum,
                "responsibility_events": event_count,
                "responsibility_active_batches": active_batches,
                "lr": baseline_optimizer.param_groups[0]["lr"],
            }
            if epoch == activation_start_epoch:
                identical, maximum, mismatches = exact_state_identity(
                    baseline, candidate
                )
                record.update(
                    {
                        "shared_prefix_bitwise_identical": identical,
                        "shared_prefix_max_abs_difference": maximum,
                        "shared_prefix_mismatched_tensors": mismatches,
                    }
                )
                if not identical:
                    raise RuntimeError("baseline/candidate diverged before SDRR activation")
            if (
                epoch % args.eval_interval == 0
                or epoch == activation_start_epoch
                or epoch == args.epochs - 1
            ):
                record["baseline"] = evaluate(
                    baseline, test_loader, device, PD_FA
                )
                record["candidate"] = evaluate(
                    candidate, test_loader, device, PD_FA
                )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
                save_checkpoint(
                    args.run_dir / "checkpoint.pt",
                    epoch=epoch,
                    baseline=baseline,
                    candidate=candidate,
                    baseline_optimizer=baseline_optimizer,
                    candidate_optimizer=candidate_optimizer,
                    baseline_scheduler=baseline_scheduler,
                    candidate_scheduler=candidate_scheduler,
                    loader_generator=generator,
                    args=args,
                )
    finally:
        input_handle.remove()
        output_handle.remove()


if __name__ == "__main__":
    main()
