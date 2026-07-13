from __future__ import annotations

from pathlib import Path

import pytest

from tools.summarize_sdrr_formal import parse_metric_log, summarize


def _write_run(
    root: Path,
    method: str,
    dataset: str,
    seed: int,
    rows: list[tuple[float, float, float]],
) -> None:
    run = root / f"formal_{method}_{dataset}_seed{seed}_e{len(rows)}"
    run.mkdir(parents=True)
    lines = [
        f"2026-07-12-00-00-00 - {epoch:04d}\t - IoU {iou:.4f}\t "
        f"- PD {pd:.4f}\t - FA {fa:.4f}"
        for epoch, (iou, pd, fa) in enumerate(rows)
    ]
    (run / "epoch_metric.log").write_text("\n".join(lines) + "\n")


def test_summarize_uses_paired_per_run_best_iou(tmp_path: Path) -> None:
    for seed, offset in ((11, 0.00), (12, 0.01)):
        _write_run(
            tmp_path,
            "baseline",
            "toy",
            seed,
            [(0.4, 0.8, 4.0), (0.6 + offset, 0.9, 3.0)],
        )
        _write_run(
            tmp_path,
            "sdrr",
            "toy",
            seed,
            [(0.5, 0.8, 4.0), (0.7 + offset, 0.9, 2.0)],
        )

    result = summarize(tmp_path, "toy", [11, 12], expected_epochs=2)

    assert result["pairs"][0]["baseline"]["epoch"] == 1
    assert result["pairs"][0]["sdrr"]["epoch"] == 1
    assert result["aggregate"]["paired_delta"]["IoU"]["mean"] == pytest.approx(0.1)
    assert result["aggregate"]["paired_delta"]["FA"]["mean"] == pytest.approx(-1.0)
    assert result["aggregate"]["paired_delta"]["IoU"]["positive_seeds"] == 2
    assert result["pairs"][0]["trajectory"]["baseline"]["final_epoch"]["IoU"] == 0.6
    assert result["pairs"][0]["trajectory_delta"]["final_epoch"]["IoU"] == pytest.approx(0.1)
    assert result["trajectory_aggregate"]["last_20_mean"]["paired_delta"]["IoU"]["mean"] == pytest.approx(0.1)


def test_metric_parser_rejects_incomplete_epochs(tmp_path: Path) -> None:
    path = tmp_path / "epoch_metric.log"
    path.write_text(
        "2026-07-12-00-00-00 - 0000 - IoU 0.1 - PD 0.2 - FA 3.0\n"
        "2026-07-12-00-00-00 - 0002 - IoU 0.2 - PD 0.3 - FA 2.0\n"
    )

    with pytest.raises(ValueError, match="incomplete or non-canonical"):
        parse_metric_log(path, expected_epochs=3)


def test_metric_parser_rejects_explicitly_invalid_run(tmp_path: Path) -> None:
    run = tmp_path / "invalid_run"
    run.mkdir()
    metric = run / "epoch_metric.log"
    metric.write_text(
        "2026-01-01 - 0000 - IoU 0.1000 - PD 0.2000 - FA 3.0000\n",
        encoding="utf-8",
    )
    (run / "INVALID_RUN.json").write_text(
        '{"valid": false, "reason": "non-finite loss"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="explicitly marked invalid"):
        parse_metric_log(metric, expected_epochs=1)
