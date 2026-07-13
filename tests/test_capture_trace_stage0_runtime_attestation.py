from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import tools.capture_trace_stage0_runtime_attestation as attest


def split_fixture(tmp_path: Path, monkeypatch):
    dataset_dir = tmp_path / "datasets" / attest.EXPECTED_DATASET
    (dataset_dir / "img_idx").mkdir(parents=True)
    (dataset_dir / "images").mkdir()
    (dataset_dir / "masks").mkdir()
    train_names = [f"train_{index}" for index in range(5)]
    test_names = ["test_pixels_must_not_be_opened"]
    train_file = dataset_dir / "img_idx" / "train_NUAA-SIRST.txt"
    test_file = dataset_dir / "img_idx" / "test_NUAA-SIRST.txt"
    train_file.write_text("\n".join(train_names) + "\n", encoding="utf-8")
    test_file.write_text("\n".join(test_names) + "\n", encoding="utf-8")
    for index, name in enumerate(train_names):
        (dataset_dir / "images" / f"{name}.png").write_bytes(b"image" + bytes([index]))
        (dataset_dir / "masks" / f"{name}.png").write_bytes(b"mask" + bytes([index]))
    fit, val = attest.deterministic_fit_val(train_names, 7, 0.4)
    expected = {
        "official_train_count": len(train_names),
        "fit_count": len(fit),
        "val_count": len(val),
        "official_test_count": len(test_names),
        "official_train_sha256": attest.split_semantic_hash(train_names),
        "fit_sha256": attest.split_semantic_hash(fit),
        "val_sha256": attest.split_semantic_hash(val),
        "official_test_sha256": attest.split_semantic_hash(test_names),
    }
    monkeypatch.setattr(attest, "EXPECTED_SPLIT", expected)
    manifest = {
        "args": {"split_seed": 7, "val_fraction": 0.4},
        "datasets": {
            attest.EXPECTED_DATASET: {
                "dataset_dir": str(dataset_dir),
                "train_file": "img_idx/train_NUAA-SIRST.txt",
                "test_file": "img_idx/test_NUAA-SIRST.txt",
                **expected,
            }
        },
    }
    return manifest, fit, val, test_names, dataset_dir


def test_dataset_hashes_only_official_train_pixels(tmp_path, monkeypatch):
    manifest, fit, val, test_names, dataset_dir = split_fixture(tmp_path, monkeypatch)
    # There are deliberately no image/mask files for the official-test id.
    assert not (dataset_dir / "images" / f"{test_names[0]}.png").exists()
    payload, names = attest.capture_dataset_files(manifest)

    assert names == {"fit": fit, "val": val, "test": test_names}
    assert payload["official_test"]["image_or_mask_files_opened"] == 0
    assert payload["official_test"]["policy"].startswith("identifiers-only")
    assert payload["fit"]["count"] + payload["validation"]["count"] == 5
    opened_ids = {
        record["sample_id"]
        for group in (payload["fit"]["samples"], payload["validation"]["samples"])
        for record in group
    }
    assert opened_ids == {f"train_{index}" for index in range(5)}


def test_dataset_hash_fails_closed_on_missing_train_mask(tmp_path, monkeypatch):
    manifest, *_rest, dataset_dir = split_fixture(tmp_path, monkeypatch)
    (dataset_dir / "masks" / "train_2.png").unlink()
    with pytest.raises(attest.AttestationError, match="required regular file"):
        attest.capture_dataset_files(manifest)


def test_recursive_import_closure_includes_relative_module_and_namespace_marker(tmp_path):
    project = tmp_path / "project"
    (project / "pkg").mkdir(parents=True)
    (project / "main.py").write_text(
        "from pkg.mod import Public\nimport json\n", encoding="utf-8"
    )
    (project / "pkg" / "mod.py").write_text(
        "from .helper import VALUE\nPublic = VALUE\n", encoding="utf-8"
    )
    (project / "pkg" / "helper.py").write_text("VALUE = 3\n", encoding="utf-8")

    closure = attest.discover_local_import_closure(project, project / "main.py")
    observed = {record["path"] for record in closure["files"]}
    missing = {record["path"] for record in closure["missing_namespace_initializers"]}

    assert observed == {"main.py", "pkg/mod.py", "pkg/helper.py"}
    assert "pkg/__init__.py" in missing
    assert "model/__init__.py" in missing
    assert len(closure["aggregate_sha256"]) == 64


def make_manifest(project: Path) -> dict:
    dataset_dir = project / "datasets" / attest.EXPECTED_DATASET
    jobs = []
    for seed in attest.EXPECTED_SEEDS:
        run_dir = (
            project
            / "weight"
            / "clean"
            / attest.EXPECTED_BATCH_ID
            / attest.EXPECTED_DATASET
            / f"seed_{seed}"
        )
        jobs.append(
            {
                "dataset": attest.EXPECTED_DATASET,
                "dataset_dir": str(dataset_dir),
                "job_id": f"mshnet__nuaa-sirst__seed_{seed}",
                "run_dir": str(run_dir),
                "seed": seed,
                "train_file": "img_idx/train_NUAA-SIRST.txt",
                "test_file": "img_idx/test_NUAA-SIRST.txt",
            }
        )
    return {
        "batch_id": attest.EXPECTED_BATCH_ID,
        "canonical_source_commit": attest.EXPECTED_SOURCE_COMMIT,
        "canonical_protocol": dict(attest.EXPECTED_PROTOCOL),
        "stage": "development_holdout_baseline",
        "official_test_policy": "loaded only for disjoint/hash audit; not iterated",
        "args": {
            "batch_id": attest.EXPECTED_BATCH_ID,
            "datasets": attest.EXPECTED_DATASET,
            "seeds": ",".join(str(seed) for seed in attest.EXPECTED_SEEDS),
            "gpus": ",".join(attest.EXPECTED_GPUS),
            **attest.EXPECTED_RUN_ARGS,
        },
        "datasets": {attest.EXPECTED_DATASET: {"dataset_dir": str(dataset_dir)}},
        "jobs": jobs,
        "provenance": {"repository_head": attest.EXPECTED_REPOSITORY_HEAD},
    }


def test_manifest_validation_is_batch_and_protocol_strict(tmp_path):
    project = tmp_path / "project"
    manifest = make_manifest(project)
    path = (
        project
        / "repro_runs"
        / "clean"
        / attest.EXPECTED_BATCH_ID
        / "manifest.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    observed, record = attest.validate_manifest(project, attest.EXPECTED_BATCH_ID)
    assert observed["batch_id"] == attest.EXPECTED_BATCH_ID
    assert record["sha256"]

    manifest["canonical_protocol"]["evaluation_interval"] = 1
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(attest.AttestationError, match="protocol mismatch"):
        attest.validate_manifest(project, attest.EXPECTED_BATCH_ID)


def write_fake_process(
    proc_root: Path,
    pid: int,
    ppid: int,
    cmdline: list[str],
    environment: dict[str, str],
    cwd: Path,
    exe: Path,
    *,
    start_ticks: int,
    state: str = "R",
) -> None:
    root = proc_root / str(pid)
    root.mkdir(parents=True)
    # tail[0]=field 3 (state), tail[1]=field 4 (ppid), tail[19]=field 22.
    fillers = " ".join("0" for _ in range(17))
    (root / "stat").write_text(
        f"{pid} (python worker) {state} {ppid} {fillers} {start_ticks} 0\n",
        encoding="utf-8",
    )
    (root / "cmdline").write_bytes(b"\0".join(value.encode() for value in cmdline) + b"\0")
    (root / "environ").write_bytes(
        b"\0".join(f"{key}={value}".encode() for key, value in environment.items())
        + b"\0"
    )
    (root / "cwd").symlink_to(cwd, target_is_directory=True)
    (root / "exe").symlink_to(exe)


def make_fake_proc(tmp_path: Path, manifest: dict, wrong_omp: bool = False):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    (project / "main.py").write_text("# training fixture\n", encoding="utf-8")
    python = tmp_path / "python3"
    python.write_text("fixture executable\n", encoding="utf-8")
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text("cpu 0 0 0\nbtime 1700000000\n", encoding="utf-8")
    scheduler_pid = 900
    write_fake_process(
        proc,
        scheduler_pid,
        1,
        [
            str(python),
            str(project / "tools" / "run_clean_baselines.py"),
            "--batch-id",
            attest.EXPECTED_BATCH_ID,
        ],
        {},
        project,
        python,
        start_ticks=100,
    )
    identities = {}
    for index, job in enumerate(manifest["jobs"]):
        options = attest.expected_worker_options(manifest, job)
        cmdline = [str(python), str(project / "main.py")]
        for key, value in options.items():
            cmdline.extend([key, value])
        pid = 1000 + index
        env = {
            "CUDA_VISIBLE_DEVICES": attest.EXPECTED_GPUS[index],
            "PYTHONHASHSEED": str(job["seed"]),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "OMP_NUM_THREADS": "2" if wrong_omp and index == 1 else "1",
            "MKL_NUM_THREADS": "1",
        }
        write_fake_process(
            proc,
            pid,
            scheduler_pid,
            cmdline,
            env,
            project,
            python,
            start_ticks=200 + index,
        )
        identities[pid] = 200 + index
    return project, proc, identities


def test_process_discovery_checks_three_commands_environments_and_liveness(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    manifest = make_manifest(project)
    project, proc, identities = make_fake_proc(tmp_path, manifest)
    monkeypatch.setattr(attest, "git_output", lambda *_args: attest.EXPECTED_REPOSITORY_HEAD)

    payload, observed_identities = attest.discover_and_validate_processes(
        project, proc, manifest
    )
    endings = attest.assert_processes_still_alive(proc, observed_identities)

    assert observed_identities == identities
    assert payload["worker_count"] == 3
    assert [worker["seed"] for worker in payload["workers"]] == list(attest.EXPECTED_SEEDS)
    assert all(item["alive_at_capture_end"] for item in endings)


def test_process_discovery_fails_closed_on_thread_environment(tmp_path, monkeypatch):
    project = tmp_path / "project"
    manifest = make_manifest(project)
    project, proc, _ = make_fake_proc(tmp_path, manifest, wrong_omp=True)
    monkeypatch.setattr(attest, "git_output", lambda *_args: attest.EXPECTED_REPOSITORY_HEAD)
    with pytest.raises(attest.AttestationError, match="wrong OMP_NUM_THREADS"):
        attest.discover_and_validate_processes(project, proc, manifest)


def test_end_recheck_detects_pid_reuse(tmp_path, monkeypatch):
    project = tmp_path / "project"
    manifest = make_manifest(project)
    project, proc, identities = make_fake_proc(tmp_path, manifest)
    victim = next(iter(identities))
    stat_path = proc / str(victim) / "stat"
    text = stat_path.read_text(encoding="utf-8")
    stat_path.write_text(text.replace(" 200 0\n", " 999 0\n"), encoding="utf-8")
    with pytest.raises(attest.AttestationError, match="pid identity changed"):
        attest.assert_processes_still_alive(proc, identities)


def test_atomic_json_write_refuses_overwrite(tmp_path):
    output = tmp_path / "nested" / "attestation.json"
    attest.atomic_write_json(output, {"status": "PASS"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert not list(output.parent.glob("*.tmp"))
    with pytest.raises(attest.AttestationError, match="refusing to overwrite"):
        attest.atomic_write_json(output, {"status": "different"})


def test_nvidia_csv_parser_is_fail_closed():
    rows = attest.parse_nvidia_csv("0, GPU-a\n1, GPU-b", ("index", "uuid"))
    assert rows[1] == {"index": "1", "uuid": "GPU-b"}
    with pytest.raises(attest.AttestationError, match="CSV row"):
        attest.parse_nvidia_csv("0, GPU-a, extra", ("index", "uuid"))
