# DEA-lite 0.005 after NUAA negative: next-step plan and audit-only code changes

> Canonical repo root: `/home/ly/DEA`  
> Current branch recommendation: `dea-lite-0p005-nuaa-negative-archive`  
> Current decision: **NUAA DEA-lite 0.005 is a negative paired result. Do not change the model, do not run 0.01 now, and do not write DEA-lite 0.005 as a universal positive result.**

---

## 0. Verdict

The reported NUAA result should be treated as a **valid negative / dataset-dependent failure** unless retest reveals a checkpoint or evaluation mistake.

```text
NUAA baseline best-IoU:
  IoU 0.7462 / PD 0.9620 / FA 25.31

NUAA DEA-lite 0.005 best-IoU:
  IoU 0.7126 / PD 0.9354 / FA 27.52

Delta:
  IoU -0.0336
  PD  -0.0266
  FA  +2.21

checkpoint_pd_fa_best.pkl:
  absent
```

This means:

```text
NO-GO:
  claim NUAA positive
  claim DEA-lite 0.005 universally improves all datasets
  run 0.01 immediately as a rescue
  modify model/MSHNet.py
  modify model/loss.py
  change NUAA split / metric / threshold

GO:
  archive NUAA baseline + DEA-lite 0.005 artifacts
  retest both checkpoints under fixed test settings
  write machine-readable negative evidence JSON
  update the evidence matrix and paper claim
  keep NUDT as positive, IRSTD-1K as FA-control signal, NUAA as dataset-dependent failure
```

---

## 1. Updated evidence status

| Dataset | Current evidence | Decision |
|---|---|---|
| NUDT-SIRST | DEA-lite 0.005 improves IoU/PD and lowers FA versus MSHNet baseline. | Positive anchor. |
| IRSTD-1K | DEA-lite 0.005 has FA-control / PD-FA trade-off signal. | Supportive but should be summarized carefully. |
| NUAA-SIRST | DEA-lite 0.005 is worse than MSHNet baseline on IoU, PD, and FA. | Valid negative / dataset-dependent failure. |
| 0.01 | Not run in this branch. | Deferred; only allowed as a separately predeclared sensitivity, not an immediate rescue. |

The paper claim must become:

```text
DEA-lite 0.005 shows promising false-alarm control on NUDT-SIRST and IRSTD-1K,
but NUAA-SIRST exposes dataset-dependent failure under the current configuration.
```

Do not write:

```text
DEA-lite universally improves IRSTD.
DEA-lite 0.005 improves all three datasets.
DEA-lite solves false alarms.
DEA-lite is ready as a broad AAAI positive claim.
```

---

## 2. Immediate execution order

```text
R0. Freeze code: no model/loss/metric/split changes.
R1. Create a NUAA-negative archive branch.
R2. Archive NUAA baseline best-IoU checkpoint and weight.
R3. Archive NUAA DEA-lite 0.005 best-IoU checkpoint and weight.
R4. Record absence of DEA-lite PD/FA-best artifact.
R5. Retest baseline best-IoU and DEA-lite best-IoU.
R6. Parse retest logs into JSON.
R7. Compute paired delta and write gate_fail JSON.
R8. Analyze DEA-lite epoch_metric.log to confirm whether it ever passed the baseline gate.
R9. Update evidence-status markdown/JSON.
R10. Only after archive/retest is stable, decide whether to open a separate 0.01 sensitivity protocol.
```

---

## 3. Branch and guard commands

```bash
cd /home/ly/DEA

git checkout -b dea-lite-0p005-nuaa-negative-archive

# Confirm that large artifacts are not already staged.
git status --short | grep -E '(^A|^M|^\?\?)\s+(weight/|datasets/|repro_runs/|.*\.pkl|.*\.pth|.*\.tar)' && {
  echo 'ERROR: large data/checkpoint artifacts appear in git status. Do not commit them.' >&2
  exit 1
} || true
```

---

## 4. Add audit-only helper scripts

Create directories:

```bash
cd /home/ly/DEA
mkdir -p tools/official scripts/official docs/internal/dea_lite_0p005
```

### 4.1 `tools/official/parse_dea_test_log.py`

```bash
cat > tools/official/parse_dea_test_log.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_metric(text: str, names: list[str]) -> float | None:
    for name in names:
        patterns = [
            rf"\b{name}\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
            rf"\b{name}\b\s+([0-9]+(?:\.[0-9]+)?)",
            rf"{name}\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        ]
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            if matches:
                return float(matches[-1])
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--checkpoint_role", required=True, choices=["best_iou", "pdfa_best", "final", "baseline"])
    p.add_argument("--checkpoint_epoch", required=True, type=int)
    p.add_argument("--weight_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--extra", action="append", default=[], help="Optional key=value metadata entries.")
    args = p.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    weight_path = Path(args.weight_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.is_file():
        raise SystemExit(f"missing log: {log_path}")
    if not weight_path.is_file():
        raise SystemExit(f"missing weight: {weight_path}")

    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Prefer final test summary lines such as "mIoU: ..." over tqdm progress
    # descriptions such as "Epoch 1, IoU ...".
    iou = parse_metric(text, ["mIoU", "miou", "IoU", "iou"])
    pd = parse_metric(text, ["PD", "Pd", "pd"])
    fa = parse_metric(text, ["FA", "Fa", "fa"])

    extra: dict[str, Any] = {}
    for item in args.extra:
        if "=" not in item:
            raise SystemExit(f"invalid --extra entry, expected key=value: {item}")
        key, value = item.split("=", 1)
        extra[key] = value

    metrics_found = all(v is not None for v in (iou, pd, fa))
    result: dict[str, Any] = {
        "dataset": args.dataset,
        "method": args.method,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_epoch": args.checkpoint_epoch,
        "weight_path": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "metrics_found": metrics_found,
        "IoU": iou,
        "PD": pd,
        "FA": fa,
        "extra": extra,
    }

    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if not metrics_found:
        raise SystemExit(
            "Could not parse IoU/PD/FA from log. Inspect log manually and update parser patterns. "
            f"Output written to {output_path}"
        )


if __name__ == "__main__":
    main()
PY
chmod +x tools/official/parse_dea_test_log.py
```

### 4.2 `tools/official/write_dea_checkpoint_summary.py`

```bash
cat > tools/official/write_dea_checkpoint_summary.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_torch(path: Path) -> Any:
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"torch is required to read checkpoint: {exc}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weight", default="")
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--checkpoint_role", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--source_run_dir", default="")
    p.add_argument("--lambda_single", type=float, default=None)
    args = p.parse_args()

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"missing checkpoint: {ckpt_path}")

    data = load_torch(ckpt_path)
    if not isinstance(data, dict):
        raise SystemExit(f"checkpoint is not a dict: {ckpt_path}")

    def f(key: str) -> float | None:
        value = data.get(key)
        return None if value is None else float(value)

    epoch = data.get("epoch")
    if epoch is None:
        raise SystemExit(f"checkpoint missing epoch: {ckpt_path}")

    result: dict[str, Any] = {
        "dataset": args.dataset,
        "method": args.method,
        "checkpoint_role": args.checkpoint_role,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": int(epoch),
        "IoU": f("iou"),
        "PD": f("pd"),
        "FA": f("fa"),
        "best_iou": f("best_iou"),
        "best_pd_fa": f("best_pd_fa"),
        "best_pd_fa_iou": f("best_pd_fa_iou"),
        "best_pd_fa_pd": f("best_pd_fa_pd"),
        "best_pd_fa_epoch": data.get("best_pd_fa_epoch"),
        "source_run_dir": args.source_run_dir,
        "lambda_single": args.lambda_single,
    }

    if args.weight:
        weight_path = Path(args.weight).expanduser().resolve()
        if not weight_path.is_file():
            raise SystemExit(f"missing weight: {weight_path}")
        result["weight_path"] = str(weight_path)
        result["weight_sha256"] = sha256_file(weight_path)

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
PY
chmod +x tools/official/write_dea_checkpoint_summary.py
```

### 4.3 `tools/official/compare_dea_lite_against_baseline.py`

```bash
cat > tools/official/compare_dea_lite_against_baseline.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: str) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise SystemExit(f"missing json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def metric(d: dict[str, Any], key: str) -> float:
    value = d.get(key)
    if value is None:
        raise SystemExit(f"missing metric {key} in {d}")
    return float(value)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_json", required=True)
    p.add_argument("--candidate_json", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_delta_iou", type=float, default=0.0)
    p.add_argument("--min_delta_pd", type=float, default=0.0)
    p.add_argument("--max_delta_fa", type=float, default=0.0)
    p.add_argument("--allow_gate_fail", action="store_true", help="Write negative evidence JSON and return 0 even if the gate fails.")
    args = p.parse_args()

    base = read_json(args.baseline_json)
    cand = read_json(args.candidate_json)

    b_iou, c_iou = metric(base, "IoU"), metric(cand, "IoU")
    b_pd, c_pd = metric(base, "PD"), metric(cand, "PD")
    b_fa, c_fa = metric(base, "FA"), metric(cand, "FA")

    delta = {
        "IoU": c_iou - b_iou,
        "PD": c_pd - b_pd,
        "FA": c_fa - b_fa,
    }

    gate_pass = bool(
        delta["IoU"] >= args.min_delta_iou
        and delta["PD"] >= args.min_delta_pd
        and delta["FA"] <= args.max_delta_fa
    )

    decision = "DEA_LITE_POSITIVE" if gate_pass else "DEA_LITE_NEGATIVE_DATASET_DEPENDENT"
    result = {
        "baseline": base,
        "candidate": cand,
        "delta": delta,
        "thresholds": {
            "min_delta_iou": args.min_delta_iou,
            "min_delta_pd": args.min_delta_pd,
            "max_delta_fa": args.max_delta_fa,
        },
        "gate_pass": gate_pass,
        "decision": decision,
        "interpretation": (
            "candidate improves the paired baseline under the declared gate"
            if gate_pass else
            "candidate fails the paired gate; treat as dataset-dependent negative evidence unless audit invalidates the run"
        ),
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if (not gate_pass) and (not args.allow_gate_fail):
        raise SystemExit(3)


if __name__ == "__main__":
    main()
PY
chmod +x tools/official/compare_dea_lite_against_baseline.py
```

### 4.4 `tools/official/analyze_dea_epoch_metrics.py`

```bash
cat > tools/official/analyze_dea_epoch_metrics.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

LINE_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\s+-\s+"
    r"(?P<epoch>\d+)\s+\t\s+-\s+IoU\s+(?P<iou>[0-9.]+)\s+\t\s+-\s+PD\s+(?P<pd>[0-9.]+)\s+\t\s+-\s+FA\s+(?P<fa>[0-9.]+)"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epoch_metric_log", required=True)
    p.add_argument("--baseline_iou", type=float, required=True)
    p.add_argument("--baseline_pd", type=float, required=True)
    p.add_argument("--baseline_fa", type=float, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min_delta_iou", type=float, default=0.0)
    p.add_argument("--min_delta_pd", type=float, default=0.0)
    p.add_argument("--max_delta_fa", type=float, default=0.0)
    args = p.parse_args()

    path = Path(args.epoch_metric_log).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"missing epoch_metric.log: {path}")

    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        rec = {
            "epoch": int(m.group("epoch")),
            "IoU": float(m.group("iou")),
            "PD": float(m.group("pd")),
            "FA": float(m.group("fa")),
        }
        rec["delta_IoU"] = rec["IoU"] - args.baseline_iou
        rec["delta_PD"] = rec["PD"] - args.baseline_pd
        rec["delta_FA"] = rec["FA"] - args.baseline_fa
        rec["gate_pass"] = bool(
            rec["delta_IoU"] >= args.min_delta_iou
            and rec["delta_PD"] >= args.min_delta_pd
            and rec["delta_FA"] <= args.max_delta_fa
        )
        records.append(rec)

    if not records:
        raise SystemExit(f"no metric records parsed from {path}")

    best_iou = max(records, key=lambda x: x["IoU"])
    lowest_fa = min(records, key=lambda x: x["FA"])
    gate_pass_epochs = [r for r in records if r["gate_pass"]]

    result = {
        "epoch_metric_log": str(path),
        "num_records": len(records),
        "baseline": {
            "IoU": args.baseline_iou,
            "PD": args.baseline_pd,
            "FA": args.baseline_fa,
        },
        "thresholds": {
            "min_delta_iou": args.min_delta_iou,
            "min_delta_pd": args.min_delta_pd,
            "max_delta_fa": args.max_delta_fa,
        },
        "best_iou_epoch": best_iou,
        "lowest_fa_epoch": lowest_fa,
        "num_gate_pass_epochs": len(gate_pass_epochs),
        "gate_pass_epochs": gate_pass_epochs[:20],
        "decision": "HAS_GATE_PASS_EPOCH" if gate_pass_epochs else "NO_GATE_PASS_EPOCH",
    }

    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
PY
chmod +x tools/official/analyze_dea_epoch_metrics.py
```

---

## 5. Add NUAA archive/retest script

Create:

```bash
cat > scripts/official/archive_retest_nuaa_dea_lite_0p005_negative.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets/NUAA-SIRST}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/dea_lite_0p005_nuaa_negative_archive}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-false}
SEED=${SEED:-20260706}
DETERMINISTIC=${DETERMINISTIC:-true}

: "${BASE_RUN:?BASE_RUN is required, e.g. /home/ly/DEA/weight/MSHNet-... for NUAA baseline}"
: "${DEA_RUN:?DEA_RUN is required, e.g. /home/ly/DEA/weight/MSHNet-... for NUAA DEA-lite 0.005}"

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

for d in "${BASE_RUN}" "${DEA_RUN}" "${DATASET_DIR}"; do
  if [[ ! -e "${d}" ]]; then
    echo "ERROR: missing path: ${d}" >&2
    exit 2
  fi
done

for f in weight.pkl checkpoint_best_iou.pkl epoch_metric.log; do
  if [[ ! -s "${BASE_RUN}/${f}" ]]; then
    echo "ERROR: baseline artifact missing or empty: ${BASE_RUN}/${f}" >&2
    exit 3
  fi
  if [[ ! -s "${DEA_RUN}/${f}" ]]; then
    echo "ERROR: DEA artifact missing or empty: ${DEA_RUN}/${f}" >&2
    exit 4
  fi
done

read BASE_EPOCH BASE_IOU BASE_PD BASE_FA < <( "${PYTHON}" - <<'PY'
import os, torch
ck = torch.load(os.path.join(os.environ["BASE_RUN"], "checkpoint_best_iou.pkl"), map_location="cpu", weights_only=False)
print(int(ck["epoch"]), float(ck["iou"]), float(ck["pd"]), float(ck["fa"]))
PY
)
read DEA_EPOCH DEA_IOU DEA_PD DEA_FA < <( "${PYTHON}" - <<'PY'
import os, torch
ck = torch.load(os.path.join(os.environ["DEA_RUN"], "checkpoint_best_iou.pkl"), map_location="cpu", weights_only=False)
print(int(ck["epoch"]), float(ck["iou"]), float(ck["pd"]), float(ck["fa"]))
PY
)

BASE_WEIGHT="${BASE_RUN}/weight_nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}.pkl"
BASE_CKPT="${BASE_RUN}/checkpoint_nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}.pkl"
DEA_WEIGHT="${DEA_RUN}/weight_nuaa_lambda_single_0p005_best_iou_e${DEA_EPOCH}.pkl"
DEA_CKPT="${DEA_RUN}/checkpoint_nuaa_lambda_single_0p005_best_iou_e${DEA_EPOCH}.pkl"

cp -n "${BASE_RUN}/weight.pkl" "${BASE_WEIGHT}"
cp -n "${BASE_RUN}/checkpoint_best_iou.pkl" "${BASE_CKPT}"
cp -n "${DEA_RUN}/weight.pkl" "${DEA_WEIGHT}"
cp -n "${DEA_RUN}/checkpoint_best_iou.pkl" "${DEA_CKPT}"

PDFA_STATUS="absent"
if [[ -s "${DEA_RUN}/checkpoint_pd_fa_best.pkl" || -s "${DEA_RUN}/weight_pd_fa_best.pkl" ]]; then
  PDFA_STATUS="present"
fi

sha256sum \
  "${BASE_WEIGHT}" "${BASE_CKPT}" \
  "${DEA_WEIGHT}" "${DEA_CKPT}" \
  > "${OUT_DIR}/nuaa_dea_lite_0p005_archived_artifacts.sha256"

cat > "${OUT_DIR}/nuaa_dea_lite_0p005_archive_manifest.json" <<JSON
{
  "dataset": "NUAA-SIRST",
  "candidate": "DEA-lite-0.005",
  "baseline": "MSHNet-baseline",
  "baseline_run_dir": "${BASE_RUN}",
  "candidate_run_dir": "${DEA_RUN}",
  "dataset_dir": "${DATASET_DIR}",
  "baseline_best_iou": {"epoch": ${BASE_EPOCH}, "IoU": ${BASE_IOU}, "PD": ${BASE_PD}, "FA": ${BASE_FA}},
  "candidate_best_iou": {"epoch": ${DEA_EPOCH}, "IoU": ${DEA_IOU}, "PD": ${DEA_PD}, "FA": ${DEA_FA}},
  "candidate_pdfa_best_artifact_status": "${PDFA_STATUS}",
  "decision": "ARCHIVED_PENDING_RETEST",
  "interpretation": "NUAA DEA-lite 0.005 is reported negative; retest will confirm evidence status."
}
JSON

BASE_LOG="${OUT_DIR}/nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}_retest.log"
DEA_LOG="${OUT_DIR}/nuaa_dea_lite_0p005_best_iou_e${DEA_EPOCH}_retest.log"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory "${PIN_MEMORY}" \
  --mode test \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --weight-path "${BASE_WEIGHT}" \
  2>&1 | tee "${BASE_LOG}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON}" -u main.py \
  --dataset-dir "${DATASET_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --pin-memory "${PIN_MEMORY}" \
  --mode test \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --weight-path "${DEA_WEIGHT}" \
  2>&1 | tee "${DEA_LOG}"

"${PYTHON}" tools/official/parse_dea_test_log.py \
  --log "${BASE_LOG}" \
  --dataset NUAA-SIRST \
  --method MSHNet-baseline \
  --checkpoint_role baseline \
  --checkpoint_epoch "${BASE_EPOCH}" \
  --weight_path "${BASE_WEIGHT}" \
  --extra run_dir="${BASE_RUN}" \
  --output "${OUT_DIR}/nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}_retest_summary.json"

"${PYTHON}" tools/official/parse_dea_test_log.py \
  --log "${DEA_LOG}" \
  --dataset NUAA-SIRST \
  --method DEA-lite-0.005 \
  --checkpoint_role best_iou \
  --checkpoint_epoch "${DEA_EPOCH}" \
  --weight_path "${DEA_WEIGHT}" \
  --extra run_dir="${DEA_RUN}" \
  --extra pdfa_best_artifact_status="${PDFA_STATUS}" \
  --output "${OUT_DIR}/nuaa_dea_lite_0p005_best_iou_e${DEA_EPOCH}_retest_summary.json"

"${PYTHON}" tools/official/compare_dea_lite_against_baseline.py \
  --baseline_json "${OUT_DIR}/nuaa_mshnet_baseline_best_iou_e${BASE_EPOCH}_retest_summary.json" \
  --candidate_json "${OUT_DIR}/nuaa_dea_lite_0p005_best_iou_e${DEA_EPOCH}_retest_summary.json" \
  --output "${OUT_DIR}/nuaa_dea_lite_0p005_vs_mshnet_delta.json" \
  --min_delta_iou 0.0 \
  --min_delta_pd 0.0 \
  --max_delta_fa 0.0 \
  --allow_gate_fail

"${PYTHON}" tools/official/analyze_dea_epoch_metrics.py \
  --epoch_metric_log "${DEA_RUN}/epoch_metric.log" \
  --baseline_iou "${BASE_IOU}" \
  --baseline_pd "${BASE_PD}" \
  --baseline_fa "${BASE_FA}" \
  --output "${OUT_DIR}/nuaa_dea_lite_0p005_epoch_metric_audit.json" \
  --min_delta_iou 0.0 \
  --min_delta_pd 0.0 \
  --max_delta_fa 0.0

cat > "${OUT_DIR}/NUAA_NEGATIVE_README.md" <<MD
# NUAA-SIRST DEA-lite 0.005 negative evidence

Baseline best-IoU:

\`\`\`text
IoU=${BASE_IOU}
PD=${BASE_PD}
FA=${BASE_FA}
epoch=${BASE_EPOCH}
\`\`\`

DEA-lite 0.005 best-IoU:

\`\`\`text
IoU=${DEA_IOU}
PD=${DEA_PD}
FA=${DEA_FA}
epoch=${DEA_EPOCH}
\`\`\`

PD/FA-best artifact status:

\`\`\`text
${PDFA_STATUS}
\`\`\`

Decision:

\`\`\`text
NUAA-SIRST fails the DEA-lite 0.005 paired gate.
Treat as dataset-dependent negative evidence.
Do not claim universal DEA-lite improvement.
Do not run 0.01 as an immediate post-hoc rescue.
\`\`\`
MD

echo "DONE: NUAA negative archive/retest outputs written to ${OUT_DIR}"
BASH
chmod +x scripts/official/archive_retest_nuaa_dea_lite_0p005_negative.sh
```

---

## 6. Run NUAA negative archive/retest

You must provide the exact baseline and DEA run directories. Do not auto-select with `ls -td weight/MSHNet-* | head -1` because another training process may create a newer run.

```bash
cd /home/ly/DEA

export ROOT=/home/ly/DEA
export PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python
export DATASET_DIR=/home/ly/DEA/datasets/NUAA-SIRST

# Completed NUAA paired runs.
export BASE_RUN=/home/ly/DEA/weight/MSHNet-2026-07-08-02-11-49
export DEA_RUN=/home/ly/DEA/weight/MSHNet-2026-07-08-04-35-33

CUDA_DEVICE=0 \
OUT_DIR=/home/ly/DEA/repro_runs/dea_lite_0p005_nuaa_negative_archive \
bash scripts/official/archive_retest_nuaa_dea_lite_0p005_negative.sh
```

Expected final paired delta:

```text
IoU delta < 0
PD delta  < 0
FA delta  > 0
gate_pass = false
decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT
```

If retest gives a materially different result, stop and audit checkpoint/run-dir selection before making any paper claim.

---

## 7. Update evidence-status document

Create:

```bash
cat > docs/internal/dea_lite_0p005/evidence_status_after_nuaa.md <<'MD'
# DEA-lite 0.005 evidence status after NUAA

## Decision

DEA-lite 0.005 is not a universal positive result across NUDT-SIRST, IRSTD-1K, and NUAA-SIRST.

## Dataset summary

| Dataset | DEA-lite 0.005 status | Paper interpretation |
|---|---|---|
| NUDT-SIRST | Positive | Improves IoU/PD and reduces FA compared with MSHNet baseline. |
| IRSTD-1K | Positive FA-control signal | Use as supportive evidence; report exact paired metrics. |
| NUAA-SIRST | Negative | Dataset-dependent failure: IoU/PD decrease and FA increases. |

## NUAA result

```text
Baseline best-IoU:
  IoU 0.7462 / PD 0.9620 / FA 25.31

DEA-lite 0.005 best-IoU:
  IoU 0.7126 / PD 0.9354 / FA 27.52

PD/FA-best:
  not generated
```

## Forbidden claims

Do not claim:

```text
DEA-lite 0.005 improves all datasets.
DEA-lite universally reduces false alarms.
DEA-lite is globally robust.
NUAA supports the main positive claim.
```

## Allowed claim

```text
DEA-lite 0.005 shows promising false-alarm control on NUDT-SIRST and IRSTD-1K,
while NUAA-SIRST reveals dataset-dependent limitations under the current configuration.
```

## Next action

```text
Archive and retest NUAA negative evidence.
Do not modify the model or loss.
Do not run lambda=0.01 until the 0.005 evidence matrix is frozen.
```
MD
```

---

## 8. Optional paper-claim checker

This prevents positive wording from reappearing in local paper notes.

```bash
cat > tools/official/check_dea_no_universal_positive_claims.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FORBIDDEN_PATTERNS = [
    r"DEA-lite\s+0\.005\s+improves\s+all",
    r"universally\s+improves",
    r"universally\s+reduces\s+false\s+alarms",
    r"solves\s+false\s+alarms",
    r"robust\s+across\s+all\s+datasets",
    r"NUAA\s+.*positive",
]

DEFAULT_SCAN_DIRS = ["docs", "repro_runs"]


def mask_forbidden_example_blocks(text: str) -> str:
    """Avoid flagging explicitly forbidden examples inside audit documents."""
    lines = text.splitlines(keepends=True)
    masked = []
    in_skip = False
    fence_count = 0
    for line in lines:
        lower = line.lower()
        if "forbidden claims" in lower or "do not claim" in lower:
            in_skip = True
            fence_count = 0
            masked.append("\n")
            continue
        if in_skip:
            if line.lstrip().startswith("```"):
                fence_count += 1
                if fence_count >= 2:
                    in_skip = False
            masked.append("\n")
            continue
        masked.append(line)
    return "".join(masked)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/ly/DEA")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    violations = []
    for rel in DEFAULT_SCAN_DIRS:
        d = root / rel
        if not d.exists():
            continue
        for path in d.rglob("*.md"):
            text = path.read_text(encoding="utf-8", errors="replace")
            text = mask_forbidden_example_blocks(text)
            for pat in FORBIDDEN_PATTERNS:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    line = text[: m.start()].count("\n") + 1
                    violations.append({"file": str(path), "line": line, "pattern": pat})

    result = {"pass": len(violations) == 0, "violations": violations}
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if violations:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
PY
chmod +x tools/official/check_dea_no_universal_positive_claims.py
```

Run:

```bash
cd /home/ly/DEA
/home/ly/BasicIRSTD/infrarenet/bin/python tools/official/check_dea_no_universal_positive_claims.py \
  --root /home/ly/DEA \
  --output docs/internal/dea_lite_0p005/no_universal_positive_claims_check.json
```

---

## 9. What not to do next

Do not do these now:

```text
1. Do not run lambda_single=0.01 immediately.
2. Do not modify DEA loss to rescue NUAA.
3. Do not switch NUAA checkpoint selection rule.
4. Do not use a different NUAA split.
5. Do not write NUAA as positive.
6. Do not write a universal three-dataset AAAI claim.
```

Why: after a known negative result, running 0.01 immediately would be a post-hoc rescue unless it is explicitly opened as a new sensitivity protocol. That may be useful later, but it should not be mixed into the current 0.005 evidence chain.

---

## 10. When to consider lambda sensitivity

Only after this checklist is complete:

```text
1. NUDT 0.005 positive result is archived and retested.
2. IRSTD-1K 0.005 result is summarized in the same JSON format.
3. NUAA 0.005 negative result is archived and retested.
4. Evidence-status document is updated.
5. Paper claim is downgraded to dataset-dependent behavior.
6. A new sensitivity protocol explicitly says any new lambda is not a replacement for failed 0.005 NUAA evidence.
```

For NUAA diagnosis, the first sensitivity should be weaker than 0.005:

```text
preferred diagnostic: lambda_single=0.0025
defer: lambda_single=0.01
```

Reason: NUAA 0.005 already decreases IoU/PD and increases FA. A stronger 0.01 penalty is more likely to worsen the same failure mode, so it should not be the immediate next run.

If any lambda sensitivity is later run, treat it as:

```text
secondary sensitivity analysis / new protocol
```

not as:

```text
same 0.005 main evidence chain
```

---

## 11. Validation commands

```bash
cd /home/ly/DEA

python3 -m py_compile \
  tools/official/parse_dea_test_log.py \
  tools/official/write_dea_checkpoint_summary.py \
  tools/official/compare_dea_lite_against_baseline.py \
  tools/official/analyze_dea_epoch_metrics.py \
  tools/official/check_dea_no_universal_positive_claims.py

bash -n scripts/official/archive_retest_nuaa_dea_lite_0p005_negative.sh

git diff --check
```

Before commit:

```bash
git status --short | grep -E 'weight/|datasets/|\.pkl|\.pth|\.tar' && {
  echo 'ERROR: large/data artifacts are visible for commit. Do not commit them.' >&2
  exit 1
} || true
```

Commit only scripts and docs:

```bash
git add \
  tools/official/parse_dea_test_log.py \
  tools/official/write_dea_checkpoint_summary.py \
  tools/official/compare_dea_lite_against_baseline.py \
  tools/official/analyze_dea_epoch_metrics.py \
  tools/official/check_dea_no_universal_positive_claims.py \
  scripts/official/archive_retest_nuaa_dea_lite_0p005_negative.sh \
  docs/internal/dea_lite_0p005/evidence_status_after_nuaa.md

git commit -m "Archive NUAA negative evidence for DEA-lite 0.005"
```

---

## 12. One-line conclusion

```text
NUAA-SIRST falsifies the universal DEA-lite 0.005 claim.
The next step is not 0.01 or model changes; it is archive, retest, summarize, and downgrade the evidence claim.
```
