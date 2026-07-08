# DEA-lite 0.005：NUAA 稳定负结果后的下一步方案与代码修改

> Canonical repo root: `/home/ly/DEA`  
> 当前状态：`NUDT-SIRST` 正，`IRSTD-1K` 有 FA-control 正向信号，`NUAA-SIRST` 稳定负。  
> 当前结论：不要改模型，不要改 loss，不要补跑 `0.01`，先把 `0.005` 的 dataset-dependent evidence 固化为机器可审计证据。

---

## 0. 当前结果判定

你刚复测确认：

```text
NUAA DEA-lite 0.005:
  gate_pass = false
  decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT

Delta vs NUAA MSHNet baseline:
  IoU  -0.0336
  PD   -0.0266
  FA   +2.2104

epoch audit:
  num_gate_pass_epochs = 0
```

这说明 NUAA 不是偶然 checkpoint 问题，也不是 final epoch 没选好，而是当前 `lambda_single=0.005` 在 NUAA 上没有任何 epoch 通过 paired gate。

因此：

```text
Do not write:
  DEA-lite 0.005 improves NUAA.
  DEA-lite 0.005 improves all three datasets.
  DEA-lite universally reduces false alarms.
  DEA-lite is dataset-agnostic.

Write instead:
  DEA-lite 0.005 improves NUDT-SIRST and shows FA-control signal on IRSTD-1K,
  but NUAA-SIRST exposes dataset-dependent failure under the current configuration.
```

---

## 1. Immediate decision

### GO

```text
1. Freeze NUAA negative artifacts.
2. Write NUAA negative gate JSON.
3. Update three-dataset evidence matrix.
4. Add positive-claim checker / lambda-0.01 guard.
5. Prepare a downgraded paper claim or internal report.
6. Decide later whether to open a new predeclared DEA sensitivity protocol.
```

### NO-GO

```text
1. Do not modify model/MSHNet.py.
2. Do not modify model/loss.py.
3. Do not modify utils/metric.py.
4. Do not modify dataset split.
5. Do not synthesize PD/FA-best checkpoint.
6. Do not run lambda 0.01 now.
7. Do not present NUAA as positive.
8. Do not continue claiming universal three-dataset improvement.
```

---

## 2. Evidence status after NUAA

| Dataset | DEA-lite 0.005 status | Interpretation |
|---|---:|---|
| `NUDT-SIRST` | Positive | Main positive anchor: best-IoU improves IoU/PD/FA; PD/FA-best strongly lowers FA. |
| `IRSTD-1K` | Positive signal | FA-control / PD-FA trade-off signal. Keep as supporting evidence after manifest cleanup. |
| `NUAA-SIRST` | Negative | Stable dataset-dependent failure: IoU/PD drop and FA increases; no gate-pass epoch. |

Recommended project status:

```text
decision = DEA_LITE_0P005_DATASET_DEPENDENT
aaai_positive_universal_claim_allowed = false
run_lambda_0p01_now = false
model_changes_allowed = false
next_allowed_action = archive_and_summarize_evidence
```

---

## 3. Required directory layout

```bash
ROOT=/home/ly/DEA
PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python

NUAA_OUT=/home/ly/DEA/repro_runs/dea_lite_0p005_nuaa_negative_archive
EVIDENCE_DIR=/home/ly/DEA/docs/internal/dea_lite_0p005
```

Do not commit these local heavy directories:

```text
datasets/
weight/
repro_runs/
*.pkl
*.pth
*.tar
```

Commit only:

```text
tools/official/*.py
scripts/official/*.sh
docs/internal/dea_lite_0p005/*.md
docs/internal/dea_lite_0p005/*.json
```

---

## 4. Step 1：固化 NUAA negative decision

新建：

```text
docs/internal/dea_lite_0p005/NUAA_NEGATIVE_DATASET_DEPENDENT.md
```

内容：

```markdown
# NUAA-SIRST DEA-lite 0.005 Negative Result

## Decision

```text
decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT
gate_pass = false
num_gate_pass_epochs = 0
```

## Paired comparison

Baseline best-IoU:

```text
IoU 0.7462
PD  0.9620
FA  25.31
```

DEA-lite 0.005 best-IoU:

```text
IoU 0.7126
PD  0.9354
FA  27.52
```

Delta:

```text
IoU -0.0336
PD  -0.0266
FA  +2.2104
```

## Interpretation

NUAA-SIRST is a stable negative result for DEA-lite 0.005 under the current paired protocol.

This result must not be presented as a positive dataset result. It should be treated as dataset-dependent failure evidence.

## Allowed claims

```text
DEA-lite 0.005 shows positive behavior on NUDT-SIRST and FA-control signal on IRSTD-1K,
but NUAA-SIRST reveals dataset-dependent limitations.
```

## Forbidden claims

```text
DEA-lite 0.005 improves all datasets.
DEA-lite universally reduces false alarms.
DEA-lite is robust across NUAA/NUDT/IRSTD-1K.
NUAA supports the main positive claim.
```

## Action

Do not run lambda 0.01 until a separate predeclared sensitivity protocol is written and approved.
```
```

创建命令：

```bash
cd /home/ly/DEA

mkdir -p docs/internal/dea_lite_0p005

cat > docs/internal/dea_lite_0p005/NUAA_NEGATIVE_DATASET_DEPENDENT.md <<'MD'
# NUAA-SIRST DEA-lite 0.005 Negative Result

## Decision

```text
decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT
gate_pass = false
num_gate_pass_epochs = 0
```

## Paired comparison

Baseline best-IoU:

```text
IoU 0.7462
PD  0.9620
FA  25.31
```

DEA-lite 0.005 best-IoU:

```text
IoU 0.7126
PD  0.9354
FA  27.52
```

Delta:

```text
IoU -0.0336
PD  -0.0266
FA  +2.2104
```

## Interpretation

NUAA-SIRST is a stable negative result for DEA-lite 0.005 under the current paired protocol.

This result must not be presented as a positive dataset result. It should be treated as dataset-dependent failure evidence.

## Allowed claims

```text
DEA-lite 0.005 shows positive behavior on NUDT-SIRST and FA-control signal on IRSTD-1K,
but NUAA-SIRST reveals dataset-dependent limitations.
```

## Forbidden claims

```text
DEA-lite 0.005 improves all datasets.
DEA-lite universally reduces false alarms.
DEA-lite is robust across NUAA/NUDT/IRSTD-1K.
NUAA supports the main positive claim.
```

## Action

Do not run lambda 0.01 until a separate predeclared sensitivity protocol is written and approved.
MD
```

---

## 5. Step 2：写入机器可读 NUAA gate JSON

新建：

```text
docs/internal/dea_lite_0p005/nuaa_dea_lite_0p005_negative_gate.json
```

命令：

```bash
cd /home/ly/DEA

mkdir -p docs/internal/dea_lite_0p005

cat > docs/internal/dea_lite_0p005/nuaa_dea_lite_0p005_negative_gate.json <<'JSON'
{
  "dataset": "NUAA-SIRST",
  "method": "DEA-lite",
  "lambda_single": 0.005,
  "paired_protocol": true,
  "baseline": {
    "model": "MSHNet",
    "checkpoint_role": "best_iou",
    "IoU": 0.7462,
    "PD": 0.9620,
    "FA": 25.31
  },
  "candidate": {
    "model": "MSHNet + DEA-lite",
    "checkpoint_role": "best_iou",
    "lambda_single": 0.005,
    "epoch": 324,
    "IoU": 0.7126,
    "PD": 0.9354,
    "FA": 27.52,
    "pd_fa_best_checkpoint_generated": false
  },
  "delta": {
    "IoU": -0.0336,
    "PD": -0.0266,
    "FA": 2.2104
  },
  "epoch_audit": {
    "num_gate_pass_epochs": 0
  },
  "gate": {
    "min_delta_iou": 0.0,
    "min_delta_pd": -0.002,
    "max_delta_fa": 0.0,
    "gate_pass": false
  },
  "decision": "DEA_LITE_NEGATIVE_DATASET_DEPENDENT",
  "aaai_positive_claim_allowed": false,
  "run_lambda_0p01_now": false,
  "model_or_loss_changes_allowed": false
}
JSON
```

Validate:

```bash
python3 -m json.tool docs/internal/dea_lite_0p005/nuaa_dea_lite_0p005_negative_gate.json >/dev/null
```

---

## 6. Step 3：归档 NUAA negative artifacts

新建：

```text
scripts/official/freeze_nuaa_dea_lite_0p005_negative.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/dea_lite_0p005_nuaa_negative_archive}
BASE_RUN=${BASE_RUN:?BASE_RUN is required, e.g. /home/ly/DEA/weight/MSHNet-... baseline run}
DEA_RUN=${DEA_RUN:?DEA_RUN is required, e.g. /home/ly/DEA/weight/MSHNet-... DEA-lite run}

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

if [[ ! -d "${BASE_RUN}" ]]; then
  echo "ERROR: BASE_RUN not found: ${BASE_RUN}" >&2
  exit 2
fi

if [[ ! -d "${DEA_RUN}" ]]; then
  echo "ERROR: DEA_RUN not found: ${DEA_RUN}" >&2
  exit 3
fi

required_base=(
  "weight.pkl"
  "checkpoint_best_iou.pkl"
)

required_dea=(
  "weight.pkl"
  "checkpoint_best_iou.pkl"
)

for f in "${required_base[@]}"; do
  if [[ ! -s "${BASE_RUN}/${f}" ]]; then
    echo "ERROR: missing baseline artifact: ${BASE_RUN}/${f}" >&2
    exit 4
  fi
done

for f in "${required_dea[@]}"; do
  if [[ ! -s "${DEA_RUN}/${f}" ]]; then
    echo "ERROR: missing DEA artifact: ${DEA_RUN}/${f}" >&2
    exit 5
  fi
done

BASE_WEIGHT="${BASE_RUN}/weight_nuaa_mshnet_baseline_best_iou.pkl"
BASE_CKPT="${BASE_RUN}/checkpoint_nuaa_mshnet_baseline_best_iou.pkl"

DEA_WEIGHT="${DEA_RUN}/weight_nuaa_lambda_single_0p005_best_iou_negative.pkl"
DEA_CKPT="${DEA_RUN}/checkpoint_nuaa_lambda_single_0p005_best_iou_negative.pkl"

cp -n "${BASE_RUN}/weight.pkl" "${BASE_WEIGHT}"
cp -n "${BASE_RUN}/checkpoint_best_iou.pkl" "${BASE_CKPT}"

cp -n "${DEA_RUN}/weight.pkl" "${DEA_WEIGHT}"
cp -n "${DEA_RUN}/checkpoint_best_iou.pkl" "${DEA_CKPT}"

PDFA_WEIGHT_PRESENT=false
PDFA_CKPT_PRESENT=false

if [[ -s "${DEA_RUN}/weight_pd_fa_best.pkl" ]]; then
  PDFA_WEIGHT_PRESENT=true
fi

if [[ -s "${DEA_RUN}/checkpoint_pd_fa_best.pkl" ]]; then
  PDFA_CKPT_PRESENT=true
fi

sha256sum \
  "${BASE_WEIGHT}" \
  "${BASE_CKPT}" \
  "${DEA_WEIGHT}" \
  "${DEA_CKPT}" \
  > "${OUT_DIR}/nuaa_dea_lite_0p005_negative_artifacts.sha256"

cat > "${OUT_DIR}/nuaa_dea_lite_0p005_negative_manifest.json" <<JSON
{
  "dataset": "NUAA-SIRST",
  "decision": "DEA_LITE_NEGATIVE_DATASET_DEPENDENT",
  "baseline_run_dir": "${BASE_RUN}",
  "candidate_run_dir": "${DEA_RUN}",
  "baseline_archived_weight": "${BASE_WEIGHT}",
  "baseline_archived_checkpoint": "${BASE_CKPT}",
  "candidate_archived_weight": "${DEA_WEIGHT}",
  "candidate_archived_checkpoint": "${DEA_CKPT}",
  "candidate_pd_fa_best_weight_present": ${PDFA_WEIGHT_PRESENT},
  "candidate_pd_fa_best_checkpoint_present": ${PDFA_CKPT_PRESENT},
  "reported_baseline": {
    "IoU": 0.7462,
    "PD": 0.9620,
    "FA": 25.31
  },
  "reported_candidate": {
    "checkpoint_role": "best_iou",
    "epoch": 324,
    "IoU": 0.7126,
    "PD": 0.9354,
    "FA": 27.52
  },
  "delta": {
    "IoU": -0.0336,
    "PD": -0.0266,
    "FA": 2.2104
  },
  "epoch_audit": {
    "num_gate_pass_epochs": 0
  },
  "allowed_next_actions": [
    "update evidence matrix",
    "downgrade claims",
    "write limitation",
    "predeclare new sensitivity protocol if needed"
  ],
  "forbidden_next_actions": [
    "claim NUAA positive",
    "run lambda 0.01 as immediate rescue",
    "modify model or loss under the same 0.005 protocol"
  ]
}
JSON

echo "Wrote ${OUT_DIR}/nuaa_dea_lite_0p005_negative_manifest.json"
echo "Wrote ${OUT_DIR}/nuaa_dea_lite_0p005_negative_artifacts.sha256"
```

授权：

```bash
chmod +x scripts/official/freeze_nuaa_dea_lite_0p005_negative.sh
```

运行时不要自动猜 run dir。手动指定：

```bash
cd /home/ly/DEA

ROOT=/home/ly/DEA \
BASE_RUN=/home/ly/DEA/weight/<NUAA_BASELINE_RUN_DIR> \
DEA_RUN=/home/ly/DEA/weight/<NUAA_DEA_0P005_RUN_DIR> \
OUT_DIR=/home/ly/DEA/repro_runs/dea_lite_0p005_nuaa_negative_archive \
bash scripts/official/freeze_nuaa_dea_lite_0p005_negative.sh
```

---

## 7. Step 4：更新三数据集 evidence matrix

新建：

```text
docs/internal/dea_lite_0p005/evidence_matrix_after_nuaa_negative.md
```

内容：

```markdown
# DEA-lite 0.005 Evidence Matrix After NUAA

## Decision

```text
decision = DEA_LITE_0P005_DATASET_DEPENDENT
universal_positive_claim_allowed = false
run_lambda_0p01_now = false
```

## Dataset matrix

| Dataset | Status | Key evidence | Interpretation |
|---|---:|---|---|
| NUDT-SIRST | Positive | best-IoU improves IoU/PD/FA; PD/FA-best lowers FA further | Main positive anchor |
| IRSTD-1K | Positive signal | FA-control / PD-FA trade-off | Supporting signal |
| NUAA-SIRST | Negative | IoU -0.0336, PD -0.0266, FA +2.2104; 0 gate-pass epochs | Dataset-dependent failure |

## Paper implication

Do not write a three-dataset universal positive claim.

Safe wording:

```text
DEA-lite 0.005 shows promising false-alarm control on NUDT-SIRST and IRSTD-1K, while NUAA-SIRST exposes dataset-dependent limitations under the current fixed configuration.
```

Unsafe wording:

```text
DEA-lite improves all datasets.
DEA-lite universally reduces false alarms.
DEA-lite is robust across NUDT/IRSTD/NUAA.
```

## Next decision

Do not run lambda 0.01 until a separate predeclared sensitivity protocol is written.
```
```

创建命令：

```bash
cd /home/ly/DEA

mkdir -p docs/internal/dea_lite_0p005

cat > docs/internal/dea_lite_0p005/evidence_matrix_after_nuaa_negative.md <<'MD'
# DEA-lite 0.005 Evidence Matrix After NUAA

## Decision

```text
decision = DEA_LITE_0P005_DATASET_DEPENDENT
universal_positive_claim_allowed = false
run_lambda_0p01_now = false
```

## Dataset matrix

| Dataset | Status | Key evidence | Interpretation |
|---|---:|---|---|
| NUDT-SIRST | Positive | best-IoU improves IoU/PD/FA; PD/FA-best lowers FA further | Main positive anchor |
| IRSTD-1K | Positive signal | FA-control / PD-FA trade-off | Supporting signal |
| NUAA-SIRST | Negative | IoU -0.0336, PD -0.0266, FA +2.2104; 0 gate-pass epochs | Dataset-dependent failure |

## Paper implication

Do not write a three-dataset universal positive claim.

Safe wording:

```text
DEA-lite 0.005 shows promising false-alarm control on NUDT-SIRST and IRSTD-1K, while NUAA-SIRST exposes dataset-dependent limitations under the current fixed configuration.
```

Unsafe wording:

```text
DEA-lite improves all datasets.
DEA-lite universally reduces false alarms.
DEA-lite is robust across NUDT/IRSTD/NUAA.
```

## Next decision

Do not run lambda 0.01 until a separate predeclared sensitivity protocol is written.
MD
```

---

## 8. Step 5：新增 positive-claim checker

新建：

```text
tools/official/check_dea_no_universal_claims.py
```

内容：

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_PATTERNS = [
    r"\bDEA-lite\b.*\b(improves|improved|outperforms|reduces)\b.*\b(all|across all|three datasets|NUAA)\b",
    r"\buniversally\b.*\b(DEA-lite|DEA)\b",
    r"\bDEA-lite\b.*\buniversally\b",
    r"\bDEA-lite\b.*\bsolves\b.*\b(false alarms|FA)\b",
    r"\bNUAA\b.*\bpositive\b",
    r"\bNUAA\b.*\bimproves\b",
    r"\b0\.005\b.*\bglobally optimal\b",
]


def should_scan(path: Path) -> bool:
    if path.is_dir():
        return False
    if path.suffix.lower() not in {".md", ".txt", ".tex", ".rst"}:
        return False
    parts = set(path.parts)
    if "weight" in parts or "datasets" in parts or "repro_runs" in parts:
        return False
    return True


def scan_file(path: Path, patterns: list[re.Pattern[str]]) -> list[dict[str, Any]]:
    out = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for pat in patterns:
            if pat.search(line):
                out.append(
                    {
                        "path": str(path),
                        "line": lineno,
                        "pattern": pat.pattern,
                        "text": line.strip(),
                    }
                )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/ly/DEA")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    patterns = [re.compile(pat, flags=re.IGNORECASE) for pat in DEFAULT_PATTERNS]

    violations = []
    for path in root.rglob("*"):
        if should_scan(path):
            violations.extend(scan_file(path, patterns))

    result = {
        "check": "dea_no_universal_positive_claims_after_nuaa_negative",
        "root": str(root),
        "pass": len(violations) == 0,
        "violations": violations,
        "decision_if_fail": "CLAIM_TEXT_CONTAINS_FORBIDDEN_UNIVERSAL_POSITIVE_STATEMENT",
    }

    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if violations:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
```

运行：

```bash
cd /home/ly/DEA

chmod +x tools/official/check_dea_no_universal_claims.py

python3 tools/official/check_dea_no_universal_claims.py \
  --root /home/ly/DEA \
  --output docs/internal/dea_lite_0p005/no_universal_claims_after_nuaa_negative.json
```

---

## 9. Step 6：新增 lambda 0.01 guard

新建：

```text
scripts/official/guard_dea_no_lambda_0p01_without_protocol.sh
```

内容：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PROTOCOL=${PROTOCOL:-${ROOT}/docs/internal/dea_lite_0p005/lambda_0p01_sensitivity_protocol.md}

cd "${ROOT}"

if [[ -f "${PROTOCOL}" ]]; then
  echo "ALLOW: lambda 0.01 protocol exists: ${PROTOCOL}"
  exit 0
fi

cat >&2 <<MSG
BLOCKED: Do not run lambda 0.01 yet.

Reason:
  NUAA-SIRST DEA-lite 0.005 is a stable negative result:
    gate_pass=false
    decision=DEA_LITE_NEGATIVE_DATASET_DEPENDENT
    num_gate_pass_epochs=0

Allowed next step:
  Archive and summarize the dataset-dependent evidence.
  If lambda 0.01 is needed, first write a separate predeclared sensitivity protocol:

    ${PROTOCOL}

This guard does not block code inspection or documentation.
It blocks treating lambda 0.01 as immediate post-hoc rescue.
MSG

exit 3
```

运行：

```bash
cd /home/ly/DEA

chmod +x scripts/official/guard_dea_no_lambda_0p01_without_protocol.sh

bash scripts/official/guard_dea_no_lambda_0p01_without_protocol.sh
```

预期：

```text
exit code 3
BLOCKED: Do not run lambda 0.01 yet.
```

---

## 10. Optional：写 0.01 sensitivity protocol 模板，但不要执行

只有你决定要单独评估 `0.01`，才新建：

```text
docs/internal/dea_lite_0p005/lambda_0p01_sensitivity_protocol.md
```

模板内容：

```markdown
# DEA-lite Lambda 0.01 Sensitivity Protocol

## Status

```text
predeclared = false
execution_allowed = false
```

## Why this exists

NUAA-SIRST showed stable negative behavior for lambda_single=0.005.

This protocol may later evaluate whether a different lambda improves dataset robustness, but it must not be used as retroactive rescue for the 0.005 claim.

## Required before execution

```text
1. Freeze all 0.005 evidence.
2. State whether 0.01 is sensitivity analysis or new main setting.
3. Define datasets.
4. Define paired baselines.
5. Define gate thresholds.
6. Define checkpoint selection.
7. Define whether NUAA is the primary rescue target.
8. Commit this protocol before running.
```

## Execution allowed?

```text
No.
```
```

Do not create this file unless you actually intend to open a new protocol.

---

## 11. Paper claim after NUAA negative

### Current safe claim

```text
DEA-lite 0.005 shows promising false-alarm control on NUDT-SIRST and IRSTD-1K, while NUAA-SIRST exposes dataset-dependent limitations under the current fixed configuration.
```

### Stronger claim not allowed

```text
DEA-lite 0.005 improves all three datasets.
DEA-lite 0.005 is robust across datasets.
DEA-lite 0.005 universally controls false alarms.
```

### AAAI implication

Current `0.005` evidence is not enough for a strong AAAI main-conference universal method claim.

Possible routes:

| Route | Action | Risk |
|---|---|---:|
| Conservative | Write a limitation-aware paper with NUDT/IRSTD positives and NUAA negative | High AAAI risk |
| Sensitivity protocol | Predeclare `0.01` or dataset-adaptive lambda study | Medium/high time risk |
| New method | Add dataset-adaptive FA-control mechanism | Highest time risk |
| Freeze as internal result | Use this as negative/limitation evidence | Lowest risk |

Recommended immediate route:

```text
Freeze 0.005 evidence first.
Do not decide AAAI claim until evidence matrix and no-positive-claim checker pass.
```

---

## 12. Final execution order

```text
R0. Stop current 0.005 universal-positive narrative.
R1. Freeze NUAA negative artifacts and manifest.
R2. Write NUAA negative gate JSON.
R3. Update evidence matrix after NUAA.
R4. Run no-universal-positive-claims checker.
R5. Add lambda 0.01 guard.
R6. Commit documentation + audit-only scripts.
R7. Decide whether to open a separate lambda 0.01 sensitivity protocol.
R8. Do not modify model/loss under the current 0.005 protocol.
```

---

## 13. Commit plan

```bash
cd /home/ly/DEA

git checkout -b dea-lite-0p005-nuaa-negative-freeze

python3 -m py_compile \
  tools/official/check_dea_no_universal_claims.py

bash -n \
  scripts/official/freeze_nuaa_dea_lite_0p005_negative.sh \
  scripts/official/guard_dea_no_lambda_0p01_without_protocol.sh

python3 -m json.tool docs/internal/dea_lite_0p005/nuaa_dea_lite_0p005_negative_gate.json >/dev/null

python3 tools/official/check_dea_no_universal_claims.py \
  --root /home/ly/DEA \
  --output docs/internal/dea_lite_0p005/no_universal_claims_after_nuaa_negative.json || true

git status --short | grep -E '(^|/)(weight|datasets|repro_runs)/|\.pkl|\.pth|\.tar' && {
  echo "ERROR: local artifacts/checkpoints are visible for commit. Do not commit them." >&2
  exit 1
} || true

git add \
  docs/internal/dea_lite_0p005/NUAA_NEGATIVE_DATASET_DEPENDENT.md \
  docs/internal/dea_lite_0p005/nuaa_dea_lite_0p005_negative_gate.json \
  docs/internal/dea_lite_0p005/evidence_matrix_after_nuaa_negative.md \
  docs/internal/dea_lite_0p005/no_universal_claims_after_nuaa_negative.json \
  scripts/official/freeze_nuaa_dea_lite_0p005_negative.sh \
  scripts/official/guard_dea_no_lambda_0p01_without_protocol.sh \
  tools/official/check_dea_no_universal_claims.py

git commit -m "Freeze DEA-lite 0.005 NUAA negative evidence and downgrade claims"
```

---

## 14. One-line answer

```text
下一步不是跑 0.01，也不是改模型。
下一步是把 NUAA 稳定负结果正式归档，更新三数据集 evidence matrix，
阻止 universal positive claim，然后再决定是否另开预声明 sensitivity protocol。
```
