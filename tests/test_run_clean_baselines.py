from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_clean_baselines as runner


def make_args(**overrides):
    values = {
        "val_fraction": 0.2,
        "split_seed": 20260711,
        "epochs": 400,
        "batch_size": 4,
        "num_workers": 0,
        "lr": 0.05,
        "warm_epoch": 5,
        "resume": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_job(tmp_path: Path) -> dict:
    return {
        "dataset_dir": str(tmp_path / "NUAA-SIRST"),
        "train_file": "img_idx/train_NUAA-SIRST.txt",
        "test_file": "img_idx/test_NUAA-SIRST.txt",
        "seed": 20260711,
        "run_dir": str(tmp_path / "run"),
        "job_id": "mshnet__nuaa-sirst__seed_20260711",
    }


def command_value(command: list[str], flag: str) -> str:
    positions = [index for index, item in enumerate(command) if item == flag]
    assert len(positions) == 1
    return command[positions[0] + 1]


def test_build_command_is_explicit_canonical(tmp_path):
    command = runner.build_command(make_args(), make_job(tmp_path))

    expected = {
        "--mode": "train",
        "--model-type": "mshnet",
        "--mshnet-variant": "deterministic",
        "--evaluation-protocol": "internal_holdout",
        "--deep-supervision": "legacy_exact",
        "--fusion-regularizer": "none",
        "--deterministic": "true",
        "--evaluation-interval": "10",
        "--skip-final-evaluation": "false",
    }
    for flag, value in expected.items():
        assert command_value(command, flag) == value
    for forbidden in (
        "--if-checkpoint",
        "--checkpoint-dir",
        "--reset-optimizer",
        "--init-from-baseline",
    ):
        assert forbidden not in command

    assert runner.CANONICAL_PROTOCOL["checkpoint_resume"] is False
    assert runner.CANONICAL_SOURCE_COMMIT == (
        "46cdfd46802629da51f70124662af7335be74b56"
    )


def test_resume_is_rejected_before_any_launch(monkeypatch):
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(resume=True))

    with pytest.raises(RuntimeError, match="must start at epoch 0"):
        runner.main()


def test_dry_run_never_overwrites_an_existing_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"frozen": true}\n', encoding="utf-8")

    runner.install_manifest(manifest, {"frozen": False}, dry_run=True)

    assert manifest.read_text(encoding="utf-8") == '{"frozen": true}\n'


def test_real_manifest_install_is_fail_closed(tmp_path):
    manifest = tmp_path / "manifest.json"
    runner.install_manifest(manifest, {"batch": 1}, dry_run=False)

    with pytest.raises(FileExistsError, match="fresh --batch-id"):
        runner.install_manifest(manifest, {"batch": 2}, dry_run=False)
    assert json.loads(manifest.read_text(encoding="utf-8")) == {"batch": 1}
