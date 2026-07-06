# DEA-lite 0.01 结果归档、复测与 0.005 强度验证方案

适配仓库：`https://github.com/Arialliy/DEA`  
本地路径：`/home/ly/DEA`  
当前阶段：**不再修改 `model/MSHNet.py`、`model/loss.py`、`main.py` 的训练逻辑，只新增低风险实验管理工具脚本。**

---

## 1. 当前判断

当前 `lambda_single=0.01` run 已经完成。

run 目录：

```bash
/home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58
```

关键文件已经存在：

```text
weight.pkl
checkpoint_best_iou.pkl
weight_pd_fa_best.pkl
checkpoint_pd_fa_best.pkl
```

结果如下：

| Setting | Checkpoint | Epoch | IoU | PD | FA |
|---|---:|---:|---:|---:|---:|
| paired DEA-off baseline | best-IoU | 222 | 0.6705 | 0.9150 | 9.2616 |
| DEA-lite 0.01 | best-IoU | 291 | 0.6705 | 0.9116 | 7.8951 |
| DEA-lite 0.01 | PD/FA-best | 333 | 0.6639 | 0.9286 | 7.8951 |
| DEA-lite 0.01 | final | 399 | 0.6508 | 0.9082 | 8.1988 |

结论：

```text
0.01 是有效 paired positive signal。
```

原因：

```text
best-IoU checkpoint:
    IoU 与 baseline 持平：0.6705 -> 0.6705
    PD 轻微下降：0.9150 -> 0.9116
    FA 降低：9.2616 -> 7.8951

PD/FA-best checkpoint:
    IoU 小幅下降：0.6705 -> 0.6639
    PD 提升：0.9150 -> 0.9286
    FA 降低：9.2616 -> 7.8951
```

这说明 `single-scale anti-sufficiency` 在 paired setting 下确实能降低 FA，并且存在一个更高 PD 的 operating point。

---

## 2. 当前执行优先级

现在执行顺序应该是：

```text
1. 归档 0.01 run 的 best-IoU 与 PD/FA-best 权重。
2. 单独复测两个归档权重，确认 test 能复现训练日志。
3. 新增两个低风险实验管理脚本：
   - tools/archive_dea_checkpoints.sh
   - tools/dea_run_report.py
4. 生成 0.01 run 的结果汇总。
5. 再跑 lambda_single=0.005 + tau=0.6，验证更保守强度。
```

现在不要做：

```text
不要改 model/MSHNet.py
不要改 model/loss.py
不要改 main.py 的训练逻辑
不要开 lambda_dec
不要开 lambda_empty
不要加 inference-time d gate
不要加 component selector
不要加 all-16 subset
不要引入新的评估协议
```

---

## 3. 先手工归档 0.01 结果

执行：

```bash
cd /home/ly/DEA

export RUN_DIR=/home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58

cp "$RUN_DIR/weight.pkl" \
   "$RUN_DIR/weight_lambda_single_0p01_best_iou_e291.pkl"

cp "$RUN_DIR/checkpoint_best_iou.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p01_best_iou_e291.pkl"

cp "$RUN_DIR/weight_pd_fa_best.pkl" \
   "$RUN_DIR/weight_lambda_single_0p01_pdfa_best_e333.pkl"

cp "$RUN_DIR/checkpoint_pd_fa_best.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p01_pdfa_best_e333.pkl"
```

确认文件存在：

```bash
ls -lh "$RUN_DIR"/*lambda_single_0p01*.pkl
```

预期至少看到：

```text
weight_lambda_single_0p01_best_iou_e291.pkl
checkpoint_lambda_single_0p01_best_iou_e291.pkl
weight_lambda_single_0p01_pdfa_best_e333.pkl
checkpoint_lambda_single_0p01_pdfa_best_e333.pkl
```

---

## 4. 复测 best-IoU 权重

执行：

```bash
cd /home/ly/DEA

export RUN_DIR=/home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/weight_lambda_single_0p01_best_iou_e291.pkl"
```

预期接近：

```text
IoU 0.6705 / PD 0.9116 / FA 7.8951
```

---

## 5. 复测 PD/FA-best 权重

执行：

```bash
cd /home/ly/DEA

export RUN_DIR=/home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/weight_lambda_single_0p01_pdfa_best_e333.pkl"
```

预期接近：

```text
IoU 0.6639 / PD 0.9286 / FA 7.8951
```

如果复测结果和训练日志差异较大，优先检查：

```text
1. weight 文件是否确实来自当前 run。
2. test split 是否还是 /home/ly/DEA/datasets/IRSTD-1K。
3. main.py 的 test loader 是否正确加载 raw state_dict。
4. base-size / crop-size 是否没有被额外修改。
```

---

## 6. 代码修改：新增实验管理脚本

这一节是实际代码修改。只新增 `tools/` 下两个脚本，不修改训练逻辑。

### 6.1 新增 `tools/archive_dea_checkpoints.sh`

在 `/home/ly/DEA` 下执行：

```bash
cd /home/ly/DEA
mkdir -p tools
```

将下面文件保存为：

```text
tools/archive_dea_checkpoints.sh
```

完整代码如下：

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/archive_dea_checkpoints.sh RUN_DIR TAG BEST_IOU_EPOCH PDFA_EPOCH

Example:
  tools/archive_dea_checkpoints.sh \
    /home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58 \
    lambda_single_0p01 \
    291 \
    333

This script copies:
  weight.pkl                  -> weight_${TAG}_best_iou_e${BEST_IOU_EPOCH}.pkl
  checkpoint_best_iou.pkl     -> checkpoint_${TAG}_best_iou_e${BEST_IOU_EPOCH}.pkl
  weight_pd_fa_best.pkl       -> weight_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl
  checkpoint_pd_fa_best.pkl   -> checkpoint_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl
USAGE
}

if [[ $# -ne 4 ]]; then
  usage
  exit 1
fi

RUN_DIR="$1"
TAG="$2"
BEST_IOU_EPOCH="$3"
PDFA_EPOCH="$4"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "[ERROR] RUN_DIR does not exist: $RUN_DIR" >&2
  exit 2
fi

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] required file missing: $path" >&2
    exit 3
  fi
}

require_file "$RUN_DIR/weight.pkl"
require_file "$RUN_DIR/checkpoint_best_iou.pkl"
require_file "$RUN_DIR/weight_pd_fa_best.pkl"
require_file "$RUN_DIR/checkpoint_pd_fa_best.pkl"

BEST_WEIGHT="$RUN_DIR/weight_${TAG}_best_iou_e${BEST_IOU_EPOCH}.pkl"
BEST_CKPT="$RUN_DIR/checkpoint_${TAG}_best_iou_e${BEST_IOU_EPOCH}.pkl"
PDFA_WEIGHT="$RUN_DIR/weight_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl"
PDFA_CKPT="$RUN_DIR/checkpoint_${TAG}_pdfa_best_e${PDFA_EPOCH}.pkl"

cp "$RUN_DIR/weight.pkl" "$BEST_WEIGHT"
cp "$RUN_DIR/checkpoint_best_iou.pkl" "$BEST_CKPT"
cp "$RUN_DIR/weight_pd_fa_best.pkl" "$PDFA_WEIGHT"
cp "$RUN_DIR/checkpoint_pd_fa_best.pkl" "$PDFA_CKPT"

echo "[OK] archived checkpoints:"
ls -lh "$BEST_WEIGHT" "$BEST_CKPT" "$PDFA_WEIGHT" "$PDFA_CKPT"
```

授权：

```bash
chmod +x tools/archive_dea_checkpoints.sh
```

检查 shell 语法：

```bash
bash -n tools/archive_dea_checkpoints.sh
```

使用它归档当前 0.01 run：

```bash
cd /home/ly/DEA

tools/archive_dea_checkpoints.sh \
  /home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58 \
  lambda_single_0p01 \
  291 \
  333
```

---

### 6.2 新增 `tools/dea_run_report.py`

将下面文件保存为：

```text
tools/dea_run_report.py
```

完整代码如下：

```python
#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, Optional, Tuple

import torch


def load_torch_file(path: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def read_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    obj = load_torch_file(path)
    if not isinstance(obj, dict):
        return None
    return obj


def get_metric(row: Dict[str, Any]) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[float]]:
    epoch = as_int(row.get("epoch"))
    iou = as_float(row.get("iou"))
    pd = as_float(row.get("pd"))
    fa = as_float(row.get("fa"))
    return epoch, iou, pd, fa


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def fmt_delta(value: Optional[float], base: Optional[float], digits: int = 4) -> str:
    if value is None or base is None:
        return "NA"
    delta = value - base
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{digits}f}"


def append_row(rows, label, ckpt_name, ckpt, baseline):
    if ckpt is None:
        rows.append({
            "label": label,
            "checkpoint": ckpt_name,
            "epoch": "NA",
            "iou": "NA",
            "pd": "NA",
            "fa": "NA",
            "delta_iou": "NA",
            "delta_pd": "NA",
            "delta_fa": "NA",
        })
        return

    epoch, iou, pd, fa = get_metric(ckpt)
    rows.append({
        "label": label,
        "checkpoint": ckpt_name,
        "epoch": str(epoch) if epoch is not None else "NA",
        "iou": fmt(iou),
        "pd": fmt(pd),
        "fa": fmt(fa),
        "delta_iou": fmt_delta(iou, baseline.get("iou")),
        "delta_pd": fmt_delta(pd, baseline.get("pd")),
        "delta_fa": fmt_delta(fa, baseline.get("fa")),
    })


def print_markdown(rows):
    headers = [
        "label",
        "checkpoint",
        "epoch",
        "IoU",
        "PD",
        "FA",
        "ΔIoU",
        "ΔPD",
        "ΔFA",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print(
            "| {label} | {checkpoint} | {epoch} | {iou} | {pd} | {fa} | {delta_iou} | {delta_pd} | {delta_fa} |".format(
                **r
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize DEA-lite run checkpoints.")
    parser.add_argument("--run-dir", required=True, help="Run directory under weight/.")
    parser.add_argument("--label", default="run", help="Label used in the report table.")
    parser.add_argument("--baseline-iou", type=float, default=None)
    parser.add_argument("--baseline-pd", type=float, default=None)
    parser.add_argument("--baseline-fa", type=float, default=None)
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    baseline = {
        "iou": args.baseline_iou,
        "pd": args.baseline_pd,
        "fa": args.baseline_fa,
    }

    checkpoints = [
        ("latest", "checkpoint.pkl"),
        ("best-IoU", "checkpoint_best_iou.pkl"),
        ("PD/FA-best", "checkpoint_pd_fa_best.pkl"),
    ]

    rows = []
    for ckpt_label, filename in checkpoints:
        ckpt = read_checkpoint(os.path.join(run_dir, filename))
        append_row(rows, args.label, ckpt_label, ckpt, baseline)

    print(f"# DEA run report: {args.label}")
    print()
    print(f"Run directory: `{run_dir}`")
    print()
    if all(value is not None for value in baseline.values()):
        print(
            "Baseline: "
            f"IoU={baseline['iou']:.4f}, "
            f"PD={baseline['pd']:.4f}, "
            f"FA={baseline['fa']:.4f}"
        )
        print()
    print_markdown(rows)


if __name__ == "__main__":
    main()
```

检查 Python 语法：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile \
  tools/dea_run_report.py
```

运行报告：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python tools/dea_run_report.py \
  --run-dir /home/ly/DEA/weight/MSHNet-2026-07-06-18-56-58 \
  --label DEA-lite-0p01 \
  --baseline-iou 0.6705 \
  --baseline-pd 0.9150 \
  --baseline-fa 9.2616
```

---

## 7. 继续跑 0.005 + tau=0.6

如果 0.01 的两个权重复测正常，再跑更保守版本：

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --epochs 400 \
  --lr 0.05 \
  --mode train \
  --seed 20260706 \
  --deterministic true \
  --dea-lambda-single 0.005 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.6 \
  --dea-detach-evidence \
  --save-dea-debug \
  --paired-baseline-iou 0.6705 \
  --pd-fa-min-pd 0.915 \
  --pd-fa-min-iou 0.6605 \
  --pd-fa-iou-margin 0.01
```

---

## 8. 0.005 跑完后的判断标准

paired baseline：

```text
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

0.005 如果满足：

```text
IoU >= 0.6605
PD  >= 0.915
FA  <  9.2616
```

则说明更弱 anti-sufficiency 仍然有效。

如果 0.005 结果为：

```text
IoU 更接近 baseline
PD 更接近或高于 baseline
FA 仍低于 baseline
```

那么 0.005 可能比 0.01 更适合作为主配置。

如果 0.005 的 FA 降幅明显小于 0.01，或者没有生成 `checkpoint_pd_fa_best.pkl`，则保留 0.01 作为主结果，0.005 作为强度消融。

---

## 9. 当前不要做的改动

当前阶段不要执行以下操作：

```text
不要改 model/MSHNet.py
不要改 model/loss.py
不要继续改 main.py 的训练逻辑
不要开 --dea-lambda-dec
不要开 --dea-lambda-empty
不要使用 inference-time d gate
不要增加 component selector
不要增加 all-16 subset
不要新增 positive necessity loss
不要替换原始 self.final
```

原因：

```text
0.01 已经给出 paired positive signal。
现在最重要的是固化、复测、汇总，然后做更弱强度验证。
继续改核心训练逻辑会引入新变量，不利于判断 DEA-lite 是否稳定有效。
```

---

## 10. 可写入实验记录的结论

当前可以记录为：

```text
On paired IRSTD-1K training with seed 20260706, DEA-lite with lambda_single=0.01 yields two useful operating points.
At the best-IoU checkpoint, it preserves IoU at 0.6705 while reducing FA from 9.2616 to 7.8951, with only a small PD drop from 0.9150 to 0.9116.
At the PD/FA-best checkpoint, it improves PD from 0.9150 to 0.9286 and reduces FA from 9.2616 to 7.8951, with a small IoU trade-off from 0.6705 to 0.6639.
```

中文版本：

```text
在 paired IRSTD-1K 设置下，DEA-lite 的 lambda_single=0.01 给出了两个有效 operating points。
best-IoU checkpoint 在保持 IoU=0.6705 的同时，将 FA 从 9.2616 降到 7.8951，PD 仅从 0.9150 小幅降到 0.9116。
PD/FA-best checkpoint 将 PD 从 0.9150 提高到 0.9286，同时将 FA 从 9.2616 降到 7.8951，代价是 IoU 从 0.6705 小幅降到 0.6639。
```

这不是 SOTA mIoU 结果，但已经支持 DEA-lite 当前主张：

```text
training-time single-scale counterfactual anti-sufficiency can reduce false alarms while maintaining a reasonable detection operating point.
```
