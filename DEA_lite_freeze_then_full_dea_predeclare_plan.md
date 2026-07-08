# DEA-lite 先冻结，再新开 Full DEA 预声明方案

> Canonical repo root: `/home/ly/DEA`  
> Current branch: `dea-lite-0p005-nuaa-negative-archive`  
> Current decision: **先冻结 DEA-lite；不要在当前分支直接实现 Full DEA 代码。**

---

## 0. Verdict

这个判断是正确的。

当前应该拆成两步：

```text
Step 1:
  在当前分支 dea-lite-0p005-nuaa-negative-archive
  只完成 DEA-lite method-level freeze。

Step 2:
  freeze commit 后，另开 full-dea-predeclare-design 分支，
  只新增 FULL_DEA_PREDECLARE_PROTOCOL.md。
```

不要在当前分支直接加入：

```text
FullDEAHead
full_dea_loss
counterfactual branch
inference-time evidence gate
```

理由是：当前 DEA-lite 证据已经混合。NUDT-SIRST 正，IRSTD-1K 有 FA-control 信号，但 NUAA-SIRST 稳定负。因此当前实验只能支持：

```text
DEA-lite 是 MSHNet 上的 lightweight evidence-aware training regularizer。
DEA-lite 可作为 pilot / ablation / limitation evidence。
DEA-lite 不能被包装成完整 DEA 方法。
```

硬边界：

```text
当前所有 0.005 / 0.01 / lambda 相关实验都只能命名为 DEA-lite。
只有未来新增显式 evidence decomposition、counterfactual path、inference evidence gate 的结构实现，才允许命名为 Full DEA。
不要把 DEA-lite 的 NUDT 正结果写成 DEA 正结果。
不要用 DEA-lite 的 NUAA 负结果继续调 lambda 来 rescue DEA 主方法。
```

---

## 1. Evidence boundary after NUAA

当前证据边界如下：

| Dataset | DEA-lite 0.005 status | Interpretation |
|---|---:|---|
| NUDT-SIRST | Positive | IoU / PD / FA 同时优于 MSHNet baseline，是当前正结果 anchor。 |
| IRSTD-1K | Positive signal | 主要体现 FA-control / PD-FA trade-off。 |
| NUAA-SIRST | Stable negative | 暴露 dataset-dependent failure。 |

关键数值边界：

```text
NUDT-SIRST:
  MSHNet baseline best-IoU:
    IoU 0.7538765773 / PD 0.9449735450 / FA 24.3818903544
  DEA-lite 0.005 best-IoU:
    IoU 0.7632385950 / PD 0.9513227513 / FA 17.3269984234
  Interpretation:
    Positive anchor, but still DEA-lite only.

IRSTD-1K:
  MSHNet baseline best-IoU:
    IoU 0.6705 / PD 0.9150 / FA 9.2616
  DEA-lite 0.005 best-IoU:
    IoU 0.6718 / PD 0.9014 / FA 6.4527
  DEA-lite 0.005 PD/FA-best:
    IoU 0.6637 / PD 0.9218 / FA 6.6805
  Interpretation:
    FA-control / PD-FA trade-off signal, not a clean universal improvement.
```

NUAA 0.005 已复测确认：

```text
MSHNet baseline best-IoU:
  IoU 0.7461767423
  PD  0.9619771863
  FA  25.3124771831

DEA-lite 0.005 best-IoU:
  IoU 0.7126024590
  PD  0.9353612167
  FA  27.5228625146

gate_pass = false
decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT

Delta:
  IoU -0.0336
  PD  -0.0266
  FA  +2.2104

epoch audit:
  num_gate_pass_epochs = 0
```

因此当前 method-level conclusion 是：

```text
DEA-lite is not full DEA.
DEA-lite should be frozen as pilot / ablation / limitation evidence.
Full DEA must be predeclared separately before implementation.
```

---

## 2. Directory split

保留已有 0.005 证据链目录：

```text
docs/internal/dea_lite_0p005/
```

用途：

```text
lambda_single=0.005 的具体实验链：
  NUDT result
  IRSTD-1K result
  NUAA negative result
  evidence matrix
  gate JSON
  retest logs summary
```

新增 method-level 冻结目录：

```text
docs/internal/dea_lite/
```

用途：

```text
方法层冻结决策：
  DEA-lite 是否等价于 DEA？
  DEA-lite 当前角色是什么？
  哪些 claim 被禁止？
  什么时候才能进入 Full DEA？
```

这两个目录不冲突：

```text
dea_lite_0p005 = 0.005 experiment evidence chain
dea_lite        = method-level freeze decision
```

---

## 3. Immediate GO / NO-GO

### GO

```text
1. 新增 DEA-lite method-level 状态文档。
2. 新增 full-DEA claim guard。
3. 新增 freeze script。
4. 运行 guard，生成 DEA_LITE_FREEZE_DECISION.json。
5. 提交 freeze commit。
6. 新开 full-dea-predeclare-design 分支。
7. 只新增 FULL_DEA_PREDECLARE_PROTOCOL.md。
```

### NO-GO

```text
1. 不要在当前分支改 model/MSHNet.py。
2. 不要在当前分支改 model/loss.py。
3. 不要在当前分支实现 FullDEAHead。
4. 不要在当前分支实现 full_dea_loss。
5. 不要继续把 0.005 / 0.01 当作 DEA 主方法 rescue。
6. 不要把 DEA-lite 写成完整 DEA。
7. 不要提交 datasets/、weight/、repro_runs/、checkpoint 或大型日志。
```

---

## 4. Step A — add method-level freeze document

Create:

```text
docs/internal/dea_lite/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md
```

Content:

```markdown
# DEA-lite Method-Level Freeze After NUAA

## Decision

DEA-lite is frozen as pilot / ablation / limitation evidence.

It is not the full DEA method.

## Evidence status

| Dataset | Status | Interpretation |
|---|---:|---|
| NUDT-SIRST | Positive | DEA-lite 0.005 improves the MSHNet baseline under the recorded paired setting. |
| IRSTD-1K | Positive signal | DEA-lite 0.005 shows a false-alarm-control / PD-FA trade-off signal. |
| NUAA-SIRST | Stable negative | DEA-lite 0.005 fails the paired gate and exposes dataset-dependent behavior. |

## NUAA negative result

```text
gate_pass = false
decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT

Delta:
  IoU -0.0336
  PD  -0.0266
  FA  +2.2104

epoch audit:
  num_gate_pass_epochs = 0
```

## Method interpretation

The current implementation should be described as:

```text
DEA-lite = MSHNet + evidence-aware training regularization / loss constraints.
```

It should not be described as:

```text
full DEA architecture
explicit evidence decomposition model
counterfactual intervention method
inference-time evidence-control method
```

## Why this matters

The NUAA negative result shows that loss-level evidence regularization alone is insufficient.

This motivates a future Full DEA design with:

```text
target evidence branch
clutter evidence branch
counterfactual suppression or swapping path
dual real/counterfactual prediction
inference-time evidence gate or evidence-calibrated segmentation head
```

## Forbidden claims

Do not claim:

```text
DEA-lite validates full DEA.
DEA-lite is the proposed full DEA method.
DEA-lite universally improves IRSTD.
DEA-lite solves false alarms.
DEA-lite is AAAI-ready as a full method.
DEA-lite 0.005 is globally robust.
```

## Allowed claims

Allowed:

```text
DEA-lite is a lightweight evidence-aware training regularizer on top of MSHNet.
DEA-lite shows promising behavior on NUDT-SIRST and IRSTD-1K.
DEA-lite fails on NUAA-SIRST, revealing dataset-dependent limitations.
DEA-lite motivates the need for a full DEA architecture with explicit evidence decomposition and counterfactual control.
```

## Current route

```text
1. Freeze DEA-lite evidence and claims.
2. Stop presenting DEA-lite as full DEA.
3. Start a separate Full DEA predeclare protocol.
4. Implement Full DEA only after protocol review.
```
```

---

## 5. Step B — add full-DEA claim guard

Create:

```text
tools/official/check_no_full_dea_claim_from_dea_lite.py
```

Content:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FORBIDDEN_PATTERNS = [
    r"\bDEA\s+improves\b",
    r"\bDEA\s+reduces\s+false\s+alarms\b",
    r"\bDEA\s+solves\b",
    r"\bDEA\s+achieves\b",
    r"\bDEA\s+outperforms\b",
    r"\bDEA\s+is\s+effective\b",
    r"\bDEA\s+demonstrates\s+superior\b",
    r"\bfull\s+DEA\s+is\s+effective\b",
    r"\bFull\s+DEA\s+results\b",
    r"\bproposed\s+DEA\s+method\b",
    r"\bDEA-lite\s+validates\s+(the\s+)?full\s+DEA\b",
    r"\bDEA-lite\s+proves\s+(the\s+)?DEA\b",
    r"\bDEA-lite\s+is\s+(the\s+)?(proposed\s+)?full\s+DEA\b",
    r"\bDEA-lite\s+universally\s+improves\b",
    r"\buniversally\s+improves\s+IRSTD\b",
    r"\bDEA-lite\s+is\s+AAAI-ready\b",
    r"\bDEA-lite\s+is\s+our\s+main\s+method\b",
    r"\bDEA-lite\s+serves\s+as\s+the\s+full\s+DEA\b",
    r"DEA\s*显著提升",
    r"DEA\s*降低虚警",
    r"DEA\s*有效",
    r"DEA-lite\s*验证了\s*DEA",
    r"DEA-lite\s*就是\s*完整\s*DEA",
    r"DEA-lite\s*作为\s*主方法",
]

NEGATION_HINTS = [
    "do not claim",
    "do not write",
    "do not describe",
    "forbidden",
    "not allowed",
    "must not",
    "should not",
    "cannot claim",
    "不要",
    "不能",
    "禁止",
    "不应",
]

SUPPRESSED_SECTION_HINTS = [
    "forbidden claims",
    "forbidden_next_steps",
    "non-goals",
    "no-go",
    "do not claim",
    "it should not be described as",
    "full dea must not be implemented as merely",
    "不要",
    "不能",
    "禁止",
    "不应",
]

DEFAULT_SCAN_TARGETS = [
    "README.md",
    "docs",
]


def is_negated_context(lines: list[str], idx: int) -> bool:
    start = max(0, idx - 6)
    context = "\n".join(lines[start : idx + 1]).lower()
    return any(hint in context for hint in NEGATION_HINTS)


def is_suppressed_example_context(lines: list[str], idx: int) -> bool:
    start = max(0, idx - 20)
    context = "\n".join(lines[start : idx + 1]).lower()
    return any(hint in context for hint in SUPPRESSED_SECTION_HINTS)


def iter_markdown_files(root: Path, targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        p = (root / target).resolve()
        if not p.exists():
            continue
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".rst"}:
            files.append(p)
        elif p.is_dir():
            files.extend(
                q for q in p.rglob("*")
                if q.is_file() and q.suffix.lower() in {".md", ".txt", ".rst"}
            )
    return sorted(set(files))


def scan_file(path: Path, root: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    violations: list[dict[str, Any]] = []

    for idx, line in enumerate(lines):
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                if is_negated_context(lines, idx) or is_suppressed_example_context(lines, idx):
                    continue
                violations.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": idx + 1,
                        "pattern": pat,
                        "text": line.strip(),
                    }
                )

    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ly/DEA")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=DEFAULT_SCAN_TARGETS,
        help="Files or directories relative to root.",
    )
    parser.add_argument(
        "--output",
        default="docs/internal/dea_lite/DEA_LITE_FULL_DEA_CLAIM_GUARD.json",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = iter_markdown_files(root, args.targets)
    violations: list[dict[str, Any]] = []
    for path in files:
        violations.extend(scan_file(path, root))

    result: dict[str, Any] = {
        "guard": "check_no_full_dea_claim_from_dea_lite",
        "root": str(root),
        "scanned_files": [str(p.relative_to(root)) for p in files],
        "pass": len(violations) == 0,
        "violations": violations,
        "decision": "PASS_NO_FULL_DEA_CLAIM_FROM_DEA_LITE"
        if not violations
        else "FAIL_FULL_DEA_CLAIM_FROM_DEA_LITE_FOUND",
    }

    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if violations:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
```

Run:

```bash
cd /home/ly/DEA

chmod +x tools/official/check_no_full_dea_claim_from_dea_lite.py

python3 -m py_compile \
  tools/official/check_no_full_dea_claim_from_dea_lite.py

python3 tools/official/check_no_full_dea_claim_from_dea_lite.py \
  --root /home/ly/DEA \
  --output docs/internal/dea_lite/DEA_LITE_FULL_DEA_CLAIM_GUARD.json
```

---

## 6. Step C — add freeze script

Create:

```text
scripts/official/freeze_dea_lite_method_after_nuaa.sh
```

Content:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-python3}

cd "${ROOT}"

EXPECTED_BRANCH=${EXPECTED_BRANCH:-dea-lite-0p005-nuaa-negative-archive}
CURRENT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"

if [[ "${CURRENT_BRANCH}" != "${EXPECTED_BRANCH}" ]]; then
  echo "ERROR: wrong branch for DEA-lite freeze." >&2
  echo "  expected: ${EXPECTED_BRANCH}" >&2
  echo "  current : ${CURRENT_BRANCH}" >&2
  exit 6
fi

METHOD_DIR="docs/internal/dea_lite"
EVIDENCE_DIR="docs/internal/dea_lite_0p005"
DECISION_JSON="${METHOD_DIR}/DEA_LITE_FREEZE_DECISION.json"
CLAIM_GUARD_JSON="${METHOD_DIR}/DEA_LITE_FULL_DEA_CLAIM_GUARD.json"

mkdir -p "${METHOD_DIR}"

if [[ ! -f "${METHOD_DIR}/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md" ]]; then
  echo "ERROR: missing method freeze document: ${METHOD_DIR}/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md" >&2
  exit 2
fi

if [[ ! -d "${EVIDENCE_DIR}" ]]; then
  echo "ERROR: missing evidence directory: ${EVIDENCE_DIR}" >&2
  exit 3
fi

REQUIRED_EVIDENCE_FILES=(
  "docs/internal/dea_lite_0p005/evidence_status_after_nuaa.md"
  "docs/internal/dea_lite_0p005/evidence_status_after_nuaa.json"
  "docs/internal/dea_lite_0p005/no_universal_positive_claims_check.json"
)

for f in "${REQUIRED_EVIDENCE_FILES[@]}"; do
  if [[ ! -s "${f}" ]]; then
    echo "ERROR: missing required DEA-lite 0.005 evidence file: ${f}" >&2
    exit 7
  fi
done

if [[ ! -f "tools/official/check_no_full_dea_claim_from_dea_lite.py" ]]; then
  echo "ERROR: missing claim guard tool." >&2
  exit 4
fi

# Protect method code from accidental freeze-commit contamination.
# This catches staged, unstaged, and untracked implementation changes.
PROTECTED_CHANGED="$(git status --short -- main.py model utils 2>/dev/null || true)"
if [[ -n "${PROTECTED_CHANGED}" ]]; then
  echo "ERROR: protected implementation files have uncommitted changes:" >&2
  echo "${PROTECTED_CHANGED}" >&2
  echo "Freeze commit must not modify model/loss/main/utils implementation." >&2
  exit 5
fi

"${PYTHON}" tools/official/check_no_full_dea_claim_from_dea_lite.py \
  --root "${ROOT}" \
  --output "${CLAIM_GUARD_JSON}"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_STATUS_SHORT="$(git status --short 2>/dev/null || true)"
GIT_DIRTY_COUNT="$(printf "%s\n" "${GIT_STATUS_SHORT}" | sed '/^$/d' | wc -l | tr -d ' ')"
GIT_STATUS_SHORT_JSON="$(printf "%s" "${GIT_STATUS_SHORT}" | "${PYTHON}" -c 'import json, sys; print(json.dumps(sys.stdin.read()))')"

cat > "${DECISION_JSON}" <<JSON
{
  "decision": "DEA_LITE_FROZEN_AS_PILOT_ABLATION_LIMITATION_EVIDENCE",
  "root": "${ROOT}",
  "branch": "${GIT_BRANCH}",
  "commit": "${GIT_COMMIT}",
  "git_dirty_entry_count_at_decision_write": ${GIT_DIRTY_COUNT},
  "git_status_short_at_decision_write": ${GIT_STATUS_SHORT_JSON},
  "method_status": "DEA-lite is not full DEA",
  "evidence_status": {
    "NUDT-SIRST": "positive_anchor",
    "IRSTD-1K": "fa_control_positive_signal",
    "NUAA-SIRST": "stable_negative_dataset_dependent_failure"
  },
  "nuaa_negative": {
    "gate_pass": false,
    "decision": "DEA_LITE_NEGATIVE_DATASET_DEPENDENT",
    "delta_iou": -0.0336,
    "delta_pd": -0.0266,
    "delta_fa": 2.2104,
    "num_gate_pass_epochs": 0
  },
  "allowed_next_step": "open full-dea-predeclare-design branch and add protocol document only",
  "forbidden_next_steps": [
    "claim DEA-lite validates full DEA",
    "claim universal DEA-lite improvement",
    "run lambda 0.01 as DEA-main rescue",
    "mix Full DEA implementation code into the DEA-lite freeze branch"
  ],
  "claim_guard_json": "${CLAIM_GUARD_JSON}"
}
JSON

echo "DONE: wrote ${DECISION_JSON}"
```

Run:

```bash
cd /home/ly/DEA

chmod +x scripts/official/freeze_dea_lite_method_after_nuaa.sh
bash -n scripts/official/freeze_dea_lite_method_after_nuaa.sh

ROOT=/home/ly/DEA \
PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python \
bash scripts/official/freeze_dea_lite_method_after_nuaa.sh
```

---

## 7. Step D — commit DEA-lite freeze only

Before commit:

```bash
cd /home/ly/DEA

git status --short
```

Make sure no data / weight / large artifact is staged:

```bash
git status --short | grep -E '(^|/)(datasets|weight|repro_runs)/|\.pkl|\.pth|\.tar|\.pt$' && {
  echo "ERROR: data/weight/large artifacts are visible. Do not commit them." >&2
  exit 1
} || true
```

Stage only method-level freeze files:

```bash
git add \
  docs/internal/dea_lite/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md \
  docs/internal/dea_lite/DEA_LITE_FREEZE_DECISION.json \
  docs/internal/dea_lite/DEA_LITE_FULL_DEA_CLAIM_GUARD.json \
  tools/official/check_no_full_dea_claim_from_dea_lite.py \
  scripts/official/freeze_dea_lite_method_after_nuaa.sh
```

Commit:

```bash
git commit -m "Freeze DEA-lite as pilot evidence after NUAA negative"
```

Do **not** include Full DEA model code in this commit.

---

## 8. Step E — create separate Full DEA predeclare branch

After freeze commit:

```bash
cd /home/ly/DEA

git checkout -b full-dea-predeclare-design
```

Create:

```text
docs/internal/full_dea/FULL_DEA_PREDECLARE_PROTOCOL.md
```

Content:

```markdown
# Full DEA Predeclare Protocol

> This document is a protocol declaration only.  
> It does not implement Full DEA code.

## Motivation

DEA-lite is frozen as pilot / ablation / limitation evidence.

DEA-lite shows:

```text
NUDT-SIRST: positive anchor
IRSTD-1K: false-alarm-control signal
NUAA-SIRST: stable negative result
```

The NUAA failure suggests that loss-level evidence regularization is insufficient.

## Full DEA hypothesis

A full DEA method should explicitly model and intervene on evidence, rather than only regularizing training loss.

Full DEA is a new structural method on top of the MSHNet source code.
It is not a renamed DEA-lite run, not a new lambda value, and not a loss-only patch.

## MSHNet source-level insertion rule

The MSHNet encoder / multi-scale decoder should remain the base network.

Full DEA may add modules only at explicit structural insertion points:

```text
1. after multi-scale decoder feature fusion
2. before the final segmentation prediction head
3. optionally as an auxiliary branch from decoder features
```

Full DEA must not change the dataset split, metric implementation, test threshold, or checkpoint selection rule to obtain gains.

## Required structural components

A Full DEA implementation must include:

```text
1. target evidence branch E_t
2. clutter/background evidence branch E_c
3. counterfactual intervention path C(F, E_t, E_c)
4. real prediction head y_real
5. counterfactual prediction head y_cf
6. inference-time evidence gate or evidence-calibrated segmentation head
```

Minimum structural contract:

```text
F_dec = MSHNet decoder feature
E_t, E_c = EvidenceHead(F_dec)
F_cf = CounterfactualOperator(F_dec, E_t, E_c)
y_real = SegHead(F_dec, E_t, E_c)
y_cf = SegHead_cf(F_cf)
y_final = EvidenceCalibratedHead(y_real, E_t, E_c)
```

The exact operator can be revised during design review, but the implementation must expose separate target evidence, clutter evidence, and counterfactual prediction tensors.

## Non-goals

Full DEA must not be implemented as merely:

```text
MSHNet + another scalar loss
MSHNet + lambda tuning
DEA-lite with a new lambda
DEA-lite 0.0025 / 0.005 / 0.01 sensitivity rescue
post-hoc threshold adjustment
dataset-specific lambda selection
```

Loss terms may be used only after the structural branches exist.
If the method can be disabled by setting only `--dea-lambda-single 0`, it is still DEA-lite rather than Full DEA.

## First gate dataset

Use NUAA-SIRST first.

Reason:

```text
NUAA is the dataset where DEA-lite 0.005 failed stably.
A full DEA design should first show that explicit evidence decomposition and counterfactual control can fix this failure mode.
```

## Baselines

Compare:

```text
MSHNet baseline
DEA-lite 0.005 negative result
Full DEA candidate
```

## First gate

Full DEA on NUAA-SIRST must satisfy:

```text
MSHNet baseline reference:
  IoU 0.7461767423
  PD  0.9619771863
  FA  25.3124771831

DEA-lite 0.005 negative reference:
  IoU 0.7126024590
  PD  0.9353612167
  FA  27.5228625146

Full DEA first gate:
  IoU >= 0.7461767423
  PD  >= 0.9569771863
  FA  <= 25.3124771831
and Full DEA must outperform DEA-lite 0.005 on NUAA.
```

The PD tolerance is predeclared as 0.005 absolute PD.
If Full DEA fails this NUAA-first gate, stop and audit the structure before running NUDT-SIRST or IRSTD-1K.

## Evidence rules

Do not claim Full DEA works until:

```text
1. Full DEA code is predeclared and committed.
2. NUAA seed result passes the first gate.
3. The same protocol is reproduced on NUDT-SIRST and IRSTD-1K.
4. Failure analysis confirms reduced false alarms without target collapse.
```

## AAAI route

Full DEA may become an AAAI route only after it has:

```text
explicit architecture contribution
counterfactual/evidence-control mechanism
NUAA recovery evidence
multi-dataset paired evidence
ablation separating target evidence, clutter evidence, and counterfactual control
```
```

Commit only the protocol:

```bash
git add docs/internal/full_dea/FULL_DEA_PREDECLARE_PROTOCOL.md
git commit -m "Predeclare Full DEA design protocol"
```

Do not add `FullDEAHead` or `full_dea_loss` in this branch until the protocol is reviewed.

---

## 9. Exact next commands after freeze commit

Current freeze status:

```text
DEA-lite freeze files have been generated and committed.
Current freeze commit observed during planning:
  948b0ee Record DEA-lite freeze decision
```

Run the following commands after confirming you are still on the DEA-lite freeze branch.
This command block commits this plan document, creates the Full DEA predeclare branch, creates only the protocol document, and commits only that protocol.

````bash
cd /home/ly/DEA

# 1. Confirm current branch and workspace state.
git branch --show-current
git status --short

# 2. Commit only this route plan.
# Do not stage unrelated untracked plans unless you explicitly want them in this commit.
git add DEA_lite_freeze_then_full_dea_predeclare_plan.md
git commit -m "Document DEA-lite freeze to Full DEA predeclare route"

# 3. Create the Full DEA predeclare branch.
git checkout -b full-dea-predeclare-design

# 4. Create the Full DEA protocol directory.
mkdir -p docs/internal/full_dea

# 5. Create the protocol document.
tee docs/internal/full_dea/FULL_DEA_PREDECLARE_PROTOCOL.md > /dev/null <<'EOF'
# Full DEA Predeclare Protocol

> This document is a protocol declaration only.
> It does not implement Full DEA code.

## Motivation

DEA-lite is frozen as pilot / ablation / limitation evidence.

Current DEA-lite evidence:
- NUDT-SIRST: positive anchor.
- IRSTD-1K: false-alarm-control / PD-FA trade-off signal.
- NUAA-SIRST: stable negative result.

The NUAA failure shows that loss-level evidence regularization alone is insufficient.

## Full DEA Hypothesis

Full DEA should explicitly model and intervene on evidence rather than only regularizing training loss.

Full DEA is a structural method on top of MSHNet. It is not a renamed DEA-lite run, not a lambda change, and not a loss-only patch.

## MSHNet Insertion Rule

The MSHNet encoder and multi-scale decoder remain the base network.

Allowed insertion points:
1. after multi-scale decoder feature fusion
2. before the final segmentation prediction head
3. optionally as an auxiliary branch from decoder features

Do not change dataset splits, metric implementation, test threshold, or checkpoint selection rule.

## Required Structural Components

A Full DEA implementation must include:

1. target evidence branch E_t
2. clutter/background evidence branch E_c
3. counterfactual intervention path C(F, E_t, E_c)
4. real prediction head y_real
5. counterfactual prediction head y_cf
6. inference-time evidence gate or evidence-calibrated segmentation head

Minimum contract:

```text
F_dec = MSHNet decoder feature
E_t, E_c = EvidenceHead(F_dec)
F_cf = CounterfactualOperator(F_dec, E_t, E_c)
y_real = SegHead(F_dec, E_t, E_c)
y_cf = SegHead_cf(F_cf)
y_final = EvidenceCalibratedHead(y_real, E_t, E_c)
```

## Non-goals

Full DEA must not be:

```text
MSHNet + another scalar loss
MSHNet + lambda tuning
DEA-lite with a new lambda
DEA-lite 0.0025 / 0.005 / 0.01 sensitivity rescue
post-hoc threshold adjustment
dataset-specific lambda selection
```

If the method can be disabled only by setting `--dea-lambda-single 0`, it is still DEA-lite rather than Full DEA.

## First Gate Dataset

Use NUAA-SIRST first.

Reason: NUAA is where DEA-lite 0.005 failed stably. Full DEA must first show that explicit evidence decomposition and counterfactual control can recover this failure case.

## Baselines

Compare:

```text
MSHNet baseline
DEA-lite 0.005 negative result
Full DEA candidate
```

## NUAA First Gate

Reference MSHNet baseline:

```text
IoU 0.7461767423
PD  0.9619771863
FA  25.3124771831
```

Reference DEA-lite 0.005:

```text
IoU 0.7126024590
PD  0.9353612167
FA  27.5228625146
```

Full DEA must satisfy:

```text
IoU >= 0.7461767423
PD  >= 0.9569771863
FA  <= 25.3124771831
```

If Full DEA fails this NUAA-first gate, stop and audit the structure before running NUDT-SIRST or IRSTD-1K.

## Evidence Rules

Do not claim Full DEA works until:

1. Full DEA code is predeclared and committed.
2. NUAA seed result passes the first gate.
3. The same protocol is reproduced on NUDT-SIRST and IRSTD-1K.
4. Failure analysis confirms reduced false alarms without target collapse.

## AAAI Route

Full DEA may become an AAAI route only after it has:

```text
explicit architecture contribution
counterfactual/evidence-control mechanism
NUAA recovery evidence
multi-dataset paired evidence
ablation separating target evidence, clutter evidence, and counterfactual control
```
EOF

# 6. Commit only the protocol document.
git add docs/internal/full_dea/FULL_DEA_PREDECLARE_PROTOCOL.md
git commit -m "Predeclare Full DEA design protocol"

# 7. Final check. Stop here; do not implement FullDEAHead yet.
git status --short
git log --oneline -3
````

Stop after this command block. Do not implement `FullDEAHead`, `full_dea_loss`, or any MSHNet structural code until `FULL_DEA_PREDECLARE_PROTOCOL.md` is reviewed.

---

## 10. After protocol review

Only after `FULL_DEA_PREDECLARE_PROTOCOL.md` is accepted, create a new implementation branch:

```bash
git checkout -b full-dea-prototype-nuaa-first
```

Then, and only then, start implementing:

```text
model/full_dea_head.py
model/full_dea_mshnet.py
model/full_dea_loss.py
scripts/official/run_full_dea_nuaa_first.sh
tests/test_full_dea_shapes.py
tests/test_full_dea_counterfactual_path.py
```

This must be a separate commit series from the DEA-lite freeze.

---

## 11. Final execution order

```text
R0. Stay on branch: dea-lite-0p005-nuaa-negative-archive.
R1. Add method-level DEA-lite freeze document.
R2. Add full-DEA claim guard.
R3. Add freeze script.
R4. Run guard and freeze script.
R5. Generate DEA_LITE_FREEZE_DECISION.json.
R6. Commit: Freeze DEA-lite as pilot evidence after NUAA negative.
R7. New branch: full-dea-predeclare-design.
R8. Add FULL_DEA_PREDECLARE_PROTOCOL.md only.
R9. Commit protocol.
R10. Only after protocol review, open a new implementation branch for Full DEA code.
```

---

## 12. One-line conclusion

```text
Yes, follow this direction:
freeze DEA-lite first, commit the freeze cleanly, then open a separate Full DEA predeclare branch.
Do not mix Full DEA implementation code into the current DEA-lite negative-evidence branch.
```
