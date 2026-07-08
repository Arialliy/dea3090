# DEA-lite 冻结与 Full DEA 新路线方案及代码修改

> Canonical repo root: `/home/ly/DEA`  
> Current decision: **freeze DEA-lite as pilot / ablation / mixed evidence; do not package DEA-lite as the full DEA method.**  
> Next method route: **start a new full DEA protocol with explicit target/clutter evidence decomposition, counterfactual intervention, and inference-time evidence control.**

---

## 0. 一句话判断

当前结果不能继续被包装成“DEA 方法有效”。更准确的定性是：

```text
DEA-lite = MSHNet + evidence-aware training regularizer / loss constraints
```

不是：

```text
full DEA architecture
explicit target/clutter evidence decomposition
counterfactual intervention model
inference-time evidence-control mechanism
```

所以当前路线应改为：

```text
R0. 冻结 DEA-lite 0.005 结果。
R1. 把 DEA-lite 定义为 pilot / ablation / negative-and-positive mixed evidence。
R2. 停止把 0.0025 / 0.005 / 0.01 当作 DEA 主方法 rescue。
R3. 新开 full DEA 设计分支。
R4. 在 MSHNet 上实现结构性 DEA，而不是只改 loss。
R5. 首先用 NUAA 验证 full DEA 是否修复 DEA-lite 失败。
```

---

## 1. 当前 evidence 状态

| 数据集 | DEA-lite 0.005 状态 | 当前解释 |
|---|---:|---|
| NUDT-SIRST | 正结果 | best-IoU 和 PD/FA-best 均优于新的 MSHNet baseline。 |
| IRSTD-1K | 正向信号 | 有 FA-control / PD-FA trade-off 价值。 |
| NUAA-SIRST | 稳定负结果 | `gate_pass=false`，`decision=DEA_LITE_NEGATIVE_DATASET_DEPENDENT`。 |

NUAA 复测结果应正式记录为：

```text
Delta vs MSHNet baseline:
  IoU -0.0336
  PD  -0.0266
  FA  +2.2104

Epoch audit:
  num_gate_pass_epochs = 0
```

这说明当前 DEA-lite 不是稳定的通用方法，而是数据集相关的 loss-level regularization。

---

## 2. 立即停止事项

现在不要做：

```text
1. 不要继续把 0.0025 / 0.005 / 0.01 当成 DEA 主方法救火。
2. 不要把 NUAA 负结果隐去。
3. 不要写“DEA improves IRSTD”或“DEA reduces false alarms”。
4. 不要用 DEA-lite 结果证明 full DEA。
5. 不要在同一个 DEA-lite 协议里继续后验调 lambda。
```

`0.0025` 后续可以作为 diagnostic sensitivity，但不能作为当前 AAAI 主线救法。

---

## 3. 允许事项

现在允许做：

```text
1. 归档 DEA-lite 0.005 的 NUDT / IRSTD-1K / NUAA evidence。
2. 写明 DEA-lite 是 pilot / ablation / limitation evidence。
3. 添加 claim guard，防止把 DEA-lite 写成 full DEA。
4. 新开 full-DEA-design 分支。
5. 预声明 full DEA architecture 和 NUAA-first gate。
6. 只在新协议下实现结构性 DEA。
```

---

## 4. Paper claim 立即降级

当前安全表述：

```text
DEA-lite is a lightweight evidence-aware training regularizer built on top of MSHNet. It shows promising false-alarm control on NUDT-SIRST and IRSTD-1K, but it fails on NUAA-SIRST, revealing that loss-level evidence regularization alone is insufficient.
```

禁止表述：

```text
DEA is effective.
DEA improves all datasets.
DEA reduces false alarms universally.
DEA-lite validates the full DEA method.
DEA-lite is the proposed AAAI method.
```

AAAI 主线如果要继续，必须变成：

```text
DEA-lite motivates full DEA: explicit evidence decomposition and counterfactual evidence control are needed because loss-only evidence regularization fails on NUAA.
```

---

## 5. Git 分支建议

先冻结 DEA-lite，再新开 full DEA：

```bash
cd /home/ly/DEA

git status --short

git checkout -b dea-lite-freeze-after-nuaa
```

提交 DEA-lite 冻结相关文档和 guard 后，再新开：

```bash
git checkout -b full-dea-predeclare-design
```

不要把 full DEA 模型改动混进 DEA-lite 冻结分支。

---

## 6. 代码修改 A：新增 DEA-lite 状态文档

新建：

```text
docs/internal/dea_lite/DEA_LITE_STATUS_AFTER_NUAA.md
```

内容如下：

```bash
cd /home/ly/DEA
mkdir -p docs/internal/dea_lite

cat > docs/internal/dea_lite/DEA_LITE_STATUS_AFTER_NUAA.md <<'MD'
# DEA-lite status after NUAA

## Decision

DEA-lite is **not** the full DEA method.

Current DEA-lite is an MSHNet-based evidence-aware training regularizer. It does not implement a full DEA architecture with explicit target/clutter evidence decomposition, counterfactual intervention, or inference-time evidence control.

## Evidence summary

| Dataset | Status | Interpretation |
|---|---:|---|
| NUDT-SIRST | Positive | DEA-lite 0.005 improves IoU/PD/FA against the new MSHNet baseline. |
| IRSTD-1K | Positive signal | DEA-lite 0.005 shows FA-control / PD-FA trade-off behavior. |
| NUAA-SIRST | Stable negative | DEA-lite 0.005 fails the paired gate. |

## NUAA paired result

```text
gate_pass = false
decision = DEA_LITE_NEGATIVE_DATASET_DEPENDENT

Delta:
  IoU -0.0336
  PD  -0.0266
  FA  +2.2104

Epoch audit:
  num_gate_pass_epochs = 0
```

## Interpretation

DEA-lite behaves as a dataset-dependent loss-level regularizer. The NUAA failure indicates that loss-only evidence regularization is insufficient to support a full DEA method claim.

## Allowed role

DEA-lite may be used as:

```text
pilot evidence
ablation evidence
limitation evidence
motivation for full DEA
```

## Forbidden claims

Do not claim:

```text
DEA-lite validates full DEA.
DEA-lite universally improves IRSTD.
DEA-lite solves false alarms.
DEA-lite is the full AAAI method.
lambda_single=0.005 is globally robust.
```

## Next route

Start a new full DEA protocol with:

```text
explicit target evidence branch
explicit clutter evidence branch
counterfactual suppression / swapping path
dual prediction: factual and counterfactual
inference-time evidence gate or evidence-calibrated segmentation head
```

The first full DEA gate should be NUAA-SIRST, because NUAA is where DEA-lite fails.
MD
```

---

## 7. 代码修改 B：新增 claim guard

新建：

```text
tools/official/check_no_full_dea_claim_from_dea_lite.py
```

完整代码：

```bash
cd /home/ly/DEA
mkdir -p tools/official

cat > tools/official/check_no_full_dea_claim_from_dea_lite.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCAN_EXTS = {".md", ".tex", ".txt"}
DEFAULT_SCAN_DIRS = ["docs", "paper", "manuscript", "README.md"]

FORBIDDEN_PATTERNS = [
    r"\bDEA\s+improves\b",
    r"\bDEA\s+reduces\s+false\s+alarms\b",
    r"\bDEA\s+solves\b",
    r"\bDEA\s+is\s+effective\b",
    r"\bDEA\s+is\s+robust\b",
    r"\bDEA\s+universally\s+improves\b",
    r"\bDEA-lite\s+validates\s+full\s+DEA\b",
    r"\bDEA-lite\s+is\s+the\s+full\s+DEA\b",
    r"\bDEA-lite\s+solves\s+false\s+alarms\b",
    r"\b0\.005\s+is\s+globally\s+optimal\b",
]

ALLOW_CONTEXT_PATTERNS = [
    r"do\s+not\s+claim",
    r"forbidden",
    r"not\s+claim",
    r"cannot\s+claim",
    r"should\s+not\s+claim",
    r"not\s+the\s+full\s+DEA",
    r"does\s+not\s+validate",
    r"fails\b",
    r"negative",
    r"limitation",
    r"insufficient",
]


def iter_files(root: Path, scan_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in scan_paths:
        p = (root / item).resolve()
        if p.is_file() and p.suffix.lower() in SCAN_EXTS:
            files.append(p)
        elif p.is_dir():
            for child in p.rglob("*"):
                if child.is_file() and child.suffix.lower() in SCAN_EXTS:
                    files.append(child)
    return sorted(set(files))


def is_allowed_context(line: str) -> bool:
    text = line.lower()
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in ALLOW_CONTEXT_PATTERNS)


def scan_file(path: Path, root: Path) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if is_allowed_context(line):
            continue
        for pat in FORBIDDEN_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                violations.append(
                    {
                        "file": str(path.relative_to(root)),
                        "line": lineno,
                        "pattern": pat,
                        "text": line.strip(),
                    }
                )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/ly/DEA")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scan", nargs="*", default=DEFAULT_SCAN_DIRS)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    files = iter_files(root, args.scan)
    violations: list[dict[str, Any]] = []
    for f in files:
        violations.extend(scan_file(f, root))

    result = {
        "check": "no_full_dea_claim_from_dea_lite",
        "root": str(root),
        "files_scanned": [str(f.relative_to(root)) for f in files],
        "pass": len(violations) == 0,
        "num_violations": len(violations),
        "violations": violations,
        "decision": "PASS" if len(violations) == 0 else "FAIL_FULL_DEA_CLAIM_FROM_DEA_LITE",
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    if violations:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
PY

chmod +x tools/official/check_no_full_dea_claim_from_dea_lite.py
python3 -m py_compile tools/official/check_no_full_dea_claim_from_dea_lite.py
```

运行：

```bash
cd /home/ly/DEA

python3 tools/official/check_no_full_dea_claim_from_dea_lite.py \
  --root /home/ly/DEA \
  --output docs/internal/dea_lite/no_full_dea_claim_from_dea_lite.json
```

---

## 8. 代码修改 C：新增 DEA-lite 冻结脚本

新建：

```text
scripts/official/freeze_dea_lite_after_nuaa.sh
```

完整代码：

```bash
cd /home/ly/DEA
mkdir -p scripts/official

cat > scripts/official/freeze_dea_lite_after_nuaa.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-python3}
OUT_DIR=${OUT_DIR:-${ROOT}/docs/internal/dea_lite}

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

test -s "docs/internal/dea_lite/DEA_LITE_STATUS_AFTER_NUAA.md"

"${PYTHON}" tools/official/check_no_full_dea_claim_from_dea_lite.py \
  --root "${ROOT}" \
  --output "${OUT_DIR}/no_full_dea_claim_from_dea_lite.json"

cat > "${OUT_DIR}/DEA_LITE_FREEZE_DECISION.json" <<JSON
{
  "decision": "DEA_LITE_FROZEN_AS_PILOT_ABLATION_LIMITATION_EVIDENCE",
  "canonical_root": "${ROOT}",
  "allowed_next_route": "full_dea_predeclared_architecture_protocol",
  "forbidden_next_routes": [
    "package_dea_lite_as_full_dea",
    "claim_universal_dea_lite_improvement",
    "use_lambda_sweep_as_main_method_rescue",
    "hide_nuaa_negative_result"
  ],
  "datasets": {
    "NUDT-SIRST": "positive_anchor",
    "IRSTD-1K": "fa_control_signal",
    "NUAA-SIRST": "stable_negative_dataset_dependent_failure"
  }
}
JSON

echo "DONE: DEA-lite freeze decision written to ${OUT_DIR}"
SH

chmod +x scripts/official/freeze_dea_lite_after_nuaa.sh
bash -n scripts/official/freeze_dea_lite_after_nuaa.sh
```

运行：

```bash
cd /home/ly/DEA

ROOT=/home/ly/DEA \
PYTHON=/home/ly/BasicIRSTD/infrarenet/bin/python \
bash scripts/official/freeze_dea_lite_after_nuaa.sh
```

---

## 9. 代码修改 D：新增 Full DEA 预声明协议

新建：

```text
docs/internal/full_dea/FULL_DEA_PREDECLARE_PROTOCOL.md
```

内容：

```bash
cd /home/ly/DEA
mkdir -p docs/internal/full_dea

cat > docs/internal/full_dea/FULL_DEA_PREDECLARE_PROTOCOL.md <<'MD'
# Full DEA predeclared protocol

## Status

This protocol starts a new method route. It does not retroactively rescue DEA-lite.

## Motivation

DEA-lite is a loss-level evidence-aware regularizer on top of MSHNet. It shows positive behavior on NUDT-SIRST and IRSTD-1K but fails on NUAA-SIRST. Therefore, full DEA must include structural evidence modeling rather than only loss regularization.

## Method name

Working title:

```text
DEA: Decidable Evidence Aggregation with Counterfactual Evidence Control for Infrared Small Target Detection
```

## Architecture requirement

Full DEA must include all of the following:

```text
1. Target evidence branch E_t.
2. Clutter evidence branch E_c.
3. Evidence gate G = f(E_t, E_c).
4. Factual prediction y.
5. Counterfactual prediction y_cf.
6. Counterfactual intervention path, e.g. clutter suppression or target/clutter swapping.
7. Inference-time evidence-calibrated segmentation head.
```

## Not sufficient

The following are not enough to call the method full DEA:

```text
1. Adding only a scalar loss.
2. Adding only debug evidence tensors.
3. Adding only checkpoint selection logic.
4. Tuning lambda_single.
5. Reporting DEA-lite results as DEA.
```

## First dataset gate

Full DEA must first be tested on NUAA-SIRST because NUAA is where DEA-lite fails.

Baseline set:

```text
MSHNet baseline
DEA-lite 0.005 negative result
Full DEA candidate
```

NUAA first gate:

```text
Full DEA IoU >= MSHNet baseline IoU - 0.005
Full DEA PD  >= MSHNet baseline PD  - 0.005
Full DEA FA  <= MSHNet baseline FA
Full DEA must beat DEA-lite 0.005 on IoU, PD, and FA.
```

If NUAA first gate fails:

```text
Stop full DEA AAAI route.
Do not run NUDT / IRSTD-1K as rescue tables.
```

If NUAA first gate passes:

```text
Run NUDT-SIRST and IRSTD-1K paired experiments.
Then run multiseed only if all three datasets are non-negative.
```

## Evidence rules

```text
No threshold sweep for the main table.
No per-dataset lambda tuning as main claim.
No hiding NUAA failure.
No reuse of DEA-lite as full DEA evidence.
```

## Claim if successful

```text
Full DEA introduces explicit target/clutter evidence decomposition and counterfactual evidence control, improving false-alarm suppression over both MSHNet and DEA-lite on the dataset where loss-only evidence regularization fails.
```
MD
```

---

## 10. Full DEA 代码路线：不要现在直接污染 DEA-lite

Full DEA 应在新分支实现：

```bash
cd /home/ly/DEA

git checkout -b full-dea-predeclare-design
```

建议先只加新文件，不覆盖 DEA-lite 主路径：

```text
model/full_dea_heads.py
model/full_dea_wrapper.py
model/full_dea_loss.py
scripts/official/run_full_dea_nuaa_seed_first.sh
tools/official/summarize_full_dea_nuaa_first_gate.py
tests/test_full_dea_shapes_and_gate.py
```

第一阶段目标不是三数据集，而是：

```text
Can full DEA fix NUAA where DEA-lite fails?
```

---

## 11. Full DEA 最小结构设计

### 11.1 Evidence decomposition

从 decoder feature 中显式产生：

```text
E_t: target evidence map
E_c: clutter evidence map
```

### 11.2 Evidence gate

推理时使用：

```text
G = sigmoid(conv([E_t, 1 - E_c, F]))
F_gated = F * G
```

### 11.3 Counterfactual intervention

训练时构造：

```text
F_cf_clutter_suppressed = F * (1 - sigmoid(E_c))
y_cf = segmentation_head(F_cf_clutter_suppressed)
```

反事实约束：

```text
safe background 上 y_cf 应更低
GT target 上 y_cf 不应显著低于 factual prediction
```

### 11.4 Inference path

Full DEA 推理不能退化成 MSHNet 原输出。必须使用：

```text
image -> decoder feature -> E_t / E_c -> evidence gate -> segmentation head -> final mask
```

---

## 12. Full DEA 初版代码接口建议

不要马上把下面代码接入 `main.py` 主训练。先作为新文件和 shape test。

新建：

```text
model/full_dea_heads.py
```

内容：

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceDecompositionHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.target_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.clutter_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(feature)
        return {
            "target_evidence_logit": self.target_head(h),
            "clutter_evidence_logit": self.clutter_head(h),
        }


class EvidenceGate(nn.Module):
    def __init__(self, feature_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels + 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        target_evidence_logit: torch.Tensor,
        clutter_evidence_logit: torch.Tensor,
    ) -> torch.Tensor:
        target_prob = torch.sigmoid(target_evidence_logit)
        clutter_prob = torch.sigmoid(clutter_evidence_logit)
        gate_input = torch.cat([feature, target_prob, 1.0 - clutter_prob], dim=1)
        return torch.sigmoid(self.net(gate_input))


class FullDEASegmentationHead(nn.Module):
    def __init__(self, feature_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class FullDEAHead(nn.Module):
    def __init__(self, feature_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.evidence = EvidenceDecompositionHead(feature_channels, hidden_channels)
        self.gate = EvidenceGate(feature_channels, hidden_channels)
        self.seg = FullDEASegmentationHead(feature_channels, hidden_channels)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        evidence = self.evidence(feature)
        gate = self.gate(
            feature,
            evidence["target_evidence_logit"],
            evidence["clutter_evidence_logit"],
        )
        factual_feature = feature * gate
        factual_logit = self.seg(factual_feature)

        clutter_prob = torch.sigmoid(evidence["clutter_evidence_logit"])
        cf_feature = feature * (1.0 - clutter_prob)
        counterfactual_logit = self.seg(cf_feature)

        return {
            **evidence,
            "evidence_gate": gate,
            "factual_logit": factual_logit,
            "counterfactual_logit": counterfactual_logit,
        }
```

Shape test first:

```bash
python3 - <<'PY'
import torch
from model.full_dea_heads import FullDEAHead

head = FullDEAHead(feature_channels=16, hidden_channels=8)
x = torch.randn(2, 16, 128, 128)
out = head(x)
for k, v in out.items():
    print(k, tuple(v.shape))
assert out["factual_logit"].shape == (2, 1, 128, 128)
assert out["counterfactual_logit"].shape == (2, 1, 128, 128)
assert out["target_evidence_logit"].shape == (2, 1, 128, 128)
assert out["clutter_evidence_logit"].shape == (2, 1, 128, 128)
PY
```

---

## 13. Full DEA loss 设计，不要先接主训练

新建草案文件：

```text
model/full_dea_loss.py
```

第一版只定义 loss，不进主实验：

```python
from __future__ import annotations

import torch
import torch.nn.functional as F


def dilate_mask(mask: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    pad = kernel_size // 2
    return F.max_pool2d(mask.float(), kernel_size=kernel_size, stride=1, padding=pad)


def safe_background(mask: torch.Tensor, kernel_size: int = 15) -> torch.Tensor:
    return (dilate_mask(mask, kernel_size=kernel_size) < 0.5).float()


def full_dea_loss(
    out: dict[str, torch.Tensor],
    mask: torch.Tensor,
    lambda_target_evidence: float = 0.1,
    lambda_clutter_evidence: float = 0.1,
    lambda_cf_bg: float = 0.1,
    lambda_cf_target: float = 0.05,
    safe_bg_kernel: int = 15,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    y = mask.float()
    safe_bg = safe_background(y, kernel_size=safe_bg_kernel)

    factual = out["factual_logit"]
    counterfactual = out["counterfactual_logit"]
    target_ev = out["target_evidence_logit"]
    clutter_ev = out["clutter_evidence_logit"]

    loss_target_ev = F.binary_cross_entropy_with_logits(target_ev, y)
    loss_clutter_bg = F.binary_cross_entropy_with_logits(clutter_ev, safe_bg)

    loss_cf_bg_map = F.binary_cross_entropy_with_logits(
        counterfactual,
        torch.zeros_like(counterfactual),
        reduction="none",
    )
    loss_cf_bg = (loss_cf_bg_map * safe_bg).sum() / (safe_bg.sum() + 1e-6)

    with torch.no_grad():
        factual_target_prob = torch.sigmoid(factual) * y
    cf_target_prob = torch.sigmoid(counterfactual) * y
    loss_cf_target = F.relu(factual_target_prob - cf_target_prob).sum() / (y.sum() + 1e-6)

    total = (
        lambda_target_evidence * loss_target_ev
        + lambda_clutter_evidence * loss_clutter_bg
        + lambda_cf_bg * loss_cf_bg
        + lambda_cf_target * loss_cf_target
    )

    logs = {
        "loss_target_evidence": loss_target_ev.detach(),
        "loss_clutter_evidence": loss_clutter_bg.detach(),
        "loss_cf_bg": loss_cf_bg.detach(),
        "loss_cf_target": loss_cf_target.detach(),
        "safe_bg_ratio": safe_bg.mean().detach(),
    }
    return total, logs
```

注意：这只是 Full DEA 分支的初始 loss 草案。没有 NUAA-first preflight 和 shape tests 前，不要进入 400 epoch training。

---

## 14. Full DEA 第一阶段测试

新建：

```text
tests/test_full_dea_heads.py
```

内容：

```python
import torch

from model.full_dea_heads import FullDEAHead
from model.full_dea_loss import full_dea_loss


def test_full_dea_head_shapes():
    head = FullDEAHead(feature_channels=16, hidden_channels=8)
    x = torch.randn(2, 16, 64, 64)
    out = head(x)
    assert out["target_evidence_logit"].shape == (2, 1, 64, 64)
    assert out["clutter_evidence_logit"].shape == (2, 1, 64, 64)
    assert out["evidence_gate"].shape == (2, 1, 64, 64)
    assert out["factual_logit"].shape == (2, 1, 64, 64)
    assert out["counterfactual_logit"].shape == (2, 1, 64, 64)


def test_full_dea_loss_finite():
    head = FullDEAHead(feature_channels=16, hidden_channels=8)
    x = torch.randn(2, 16, 64, 64)
    mask = torch.zeros(2, 1, 64, 64)
    mask[:, :, 24:28, 24:28] = 1.0
    out = head(x)
    loss, logs = full_dea_loss(out, mask)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    assert "loss_cf_bg" in logs
```

运行：

```bash
cd /home/ly/DEA
python3 -m py_compile model/full_dea_heads.py model/full_dea_loss.py
python3 -m pytest tests/test_full_dea_heads.py
```

如果当前环境没有 `pytest`，先只跑：

```bash
python3 -m py_compile model/full_dea_heads.py model/full_dea_loss.py
python3 - <<'PY'
import torch
from model.full_dea_heads import FullDEAHead
from model.full_dea_loss import full_dea_loss
head = FullDEAHead(16, 8)
x = torch.randn(2, 16, 64, 64)
y = torch.zeros(2, 1, 64, 64)
y[:, :, 24:28, 24:28] = 1
out = head(x)
loss, logs = full_dea_loss(out, y)
print(float(loss), sorted(logs))
PY
```

---

## 15. Full DEA 后续接入 MSHNet 的原则

当前 `model/MSHNet.py` 已有 DEA-lite outputs，但它们仍然服务于 loss-level regularization。Full DEA 接入时应新增开关，不破坏旧接口：

```text
return_full_dea_feature=False by default
```

建议只在新分支中做如下扩展：

```text
MSHNet.forward(..., return_full_dea_feature=True)
```

返回：

```text
decoder_feature: x_d0
scale_logits: concat of multi-scale logits if available
base_logit: original MSHNet final logit
```

注意：这一步必须谨慎。不要在 DEA-lite freeze 分支做。

---

## 16. Full DEA 实验 gate

第一关只跑 NUAA：

```text
Dataset: NUAA-SIRST
Seed: same as current paired setting
Baseline: MSHNet
Negative anchor: DEA-lite 0.005
Candidate: Full DEA
Epochs: 400 only after smoke tests pass
```

Gate：

```text
Full DEA IoU >= MSHNet baseline IoU - 0.005
Full DEA PD  >= MSHNet baseline PD  - 0.005
Full DEA FA  <= MSHNet baseline FA
Full DEA must beat DEA-lite 0.005 on IoU, PD, and FA
```

如果 NUAA gate 不过：

```text
Stop full DEA AAAI route.
```

如果 NUAA gate 过：

```text
Run NUDT-SIRST and IRSTD-1K.
Only after all three datasets are non-negative should multiseed be considered.
```

---

## 17. 最终执行顺序

```text
R0. Freeze DEA-lite result status after NUAA.
R1. Run claim guard and commit DEA-lite freeze docs.
R2. Open full-dea-predeclare-design branch.
R3. Add FULL_DEA_PREDECLARE_PROTOCOL.md.
R4. Add FullDEAHead and full_dea_loss shape-test files.
R5. Run py_compile and shape tests only.
R6. Design MSHNet feature-return interface in new branch.
R7. Add FullDEAMSHNet wrapper.
R8. Smoke train on tiny subset only.
R9. If smoke passes, run NUAA first gate.
R10. If NUAA fails, stop AAAI route.
R11. If NUAA passes, run NUDT and IRSTD-1K.
R12. If all three datasets are non-negative, consider multiseed and paper writing.
```

---

## 18. Commit commands

Freeze DEA-lite branch:

```bash
cd /home/ly/DEA

git add \
  docs/internal/dea_lite/DEA_LITE_STATUS_AFTER_NUAA.md \
  docs/internal/dea_lite/no_full_dea_claim_from_dea_lite.json \
  docs/internal/dea_lite/DEA_LITE_FREEZE_DECISION.json \
  tools/official/check_no_full_dea_claim_from_dea_lite.py \
  scripts/official/freeze_dea_lite_after_nuaa.sh

git status --short | grep -E 'datasets/|weight/|repro_runs/|\.pkl|\.pth|\.tar' && {
  echo 'ERROR: large data/weight artifacts are visible for commit. Do not commit them.' >&2
  exit 1
} || true

git commit -m "Freeze DEA-lite as pilot evidence after NUAA negative"
```

Full DEA design branch:

```bash
git checkout -b full-dea-predeclare-design

git add \
  docs/internal/full_dea/FULL_DEA_PREDECLARE_PROTOCOL.md \
  model/full_dea_heads.py \
  model/full_dea_loss.py \
  tests/test_full_dea_heads.py

git commit -m "Predeclare full DEA architecture route after DEA-lite limitations"
```

---

## 19. 一句话结论

```text
DEA-lite 不再作为 DEA 主方法救火；它冻结为 pilot/ablation/limitation evidence。
真正下一步是新开 full DEA：显式 target/clutter evidence 分解 + counterfactual intervention + inference-time evidence gate。
第一关跑 NUAA，因为 NUAA 是 DEA-lite 暴露失败的地方。
```
