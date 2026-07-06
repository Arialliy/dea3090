# DEA-lite 0.01 Paired Positive Result：下一步方案与实际代码修改

> 适配仓库：`https://github.com/Arialliy/DEA`  
> 本文档基于当前 `/home/ly/DEA` 已经跑完的 paired DEA-off baseline 与 `lambda_single=0.01` 结果。  
> 当前阶段不再继续改模型结构，不开 full DEA，不开 `lambda_dec`、`lambda_empty`、inference-time gate、component selector、all-16 subset。  
> 当前目标：把 `0.01` 的正信号归档、复测、自动汇总；然后只补一个更保守的 `0.005 + tau=0.6` operating point。

---

## 1. 当前结果判断

### 1.1 paired DEA-off baseline

```text
paired DEA-off baseline best:
epoch 222
IoU = 0.6705
PD  = 0.9150
FA  = 9.2616
```

### 1.2 DEA-lite `lambda_single=0.01`

```text
final checkpoint, epoch 399:
IoU = 0.6508
PD  = 0.9082
FA  = 8.1988
```

```text
best-IoU checkpoint, epoch 291:
IoU = 0.6705
PD  = 0.9116
FA  = 7.8951
```

```text
PD/FA-best checkpoint, epoch 333:
IoU = 0.6639
PD  = 0.9286
FA  = 7.8951
```

### 1.3 结论

`lambda_single=0.01` 是有效 paired positive signal。

它没有提升 best IoU，但给出了两个有用 operating points：

| checkpoint | IoU | PD | FA | 结论 |
|---|---:|---:|---:|---|
| paired DEA-off best | 0.6705 | 0.9150 | 9.2616 | 主对照 |
| DEA 0.01 best-IoU | 0.6705 | 0.9116 | 7.8951 | IoU 持平，FA 降低，PD 小降 |
| DEA 0.01 PD/FA-best | 0.6639 | 0.9286 | 7.8951 | IoU 小降，PD 提高，FA 降低 |

可以写成：

```text
At the best-IoU operating point, DEA-lite keeps IoU unchanged
while reducing FA from 9.2616 to 7.8951 with only a 0.0034 PD drop.

At the PD/FA-best operating point, DEA-lite improves PD from 0.9150
to 0.9286 and reduces FA from 9.2616 to 7.8951, with a small IoU
trade-off of 0.0066.
```

这支持你的当前主线：

```text
single-scale anti-sufficiency 可以降低 MSHNet 的 false alarms；
在合适强度下，FA 下降不必伴随明显的 IoU / PD 崩坏。
```

但它还不能写成：

```text
DEA-lite 显著提升 mIoU / 达到 SOTA。
```

当前更准确的 claim 是：

```text
DEA-lite improves the FA/PD operating trade-off under a paired setting.
```

---

## 2. 当前不要继续改什么

现在不要改：

```text
model/MSHNet.py
model/loss.py
main.py 的 DEA 训练逻辑
```

也不要开：

```text
--dea-lambda-dec > 0
--dea-lambda-empty > 0
inference-time d gate
component selector
all-16 subset
positive necessity
candidate verifier
```

原因：

```text
1. 当前 0.01 已经得到有效 positive signal；
2. 继续加结构会引入新变量，反而不利于证明 single-scale anti-sufficiency 的独立作用；
3. lambda_dec 之前 d_prob 基本全 1，还没有可靠负样本学习信号；
4. lambda_empty 可能压 self.final.bias，影响 PD；
5. d gate 会直接影响推理输出，当前还没有必要。
```

当前只做两类实际代码补充：

```text
1. checkpoint 归档脚本；
2. run 结果自动汇总脚本。
```

这两个工具不会改变训练结果，只提升可复现性和实验管理。

---

## 3. 下一步执行顺序

```text
Step 1. 找到这次 lambda_single=0.01 run 目录。
Step 2. 归档 best-IoU 和 PD/FA-best 权重。
Step 3. 复测两个归档权重，确认结果可复现。
Step 4. 新增 tools/archive_dea_checkpoints.sh。
Step 5. 新增 tools/dea_run_report.py。
Step 6. 对 baseline 与 0.01 run 生成 summary markdown / json。
Step 7. 再跑 lambda_single=0.005 + tau=0.6。
Step 8. 比较 0.01 和 0.005，决定 IRSTD-1K 主配置。
```

---

## 4. 先找到当前 0.01 run 目录

如果这次 `0.01` 是刚跑完的最新目录，可以用：

```bash
cd /home/ly/DEA

ls -td /home/ly/DEA/weight/MSHNet-* | head -5
```

然后设置：

```bash
export RUN_DIR=/home/ly/DEA/weight/MSHNet-YYYY-MM-DD-HH-MM-SS
```

如果它就是最新目录，也可以：

```bash
cd /home/ly/DEA

export RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -1)

echo "$RUN_DIR"
```

确认里面有这些文件：

```bash
ls -lh "$RUN_DIR"/weight.pkl \
       "$RUN_DIR"/checkpoint_best_iou.pkl \
       "$RUN_DIR"/weight_pd_fa_best.pkl \
       "$RUN_DIR"/checkpoint_pd_fa_best.pkl \
       "$RUN_DIR"/checkpoint.pkl \
       "$RUN_DIR"/epoch_metric.log
```

---

## 5. 归档当前 0.01 checkpoint

执行：

```bash
cd /home/ly/DEA

export RUN_DIR=/home/ly/DEA/weight/MSHNet-YYYY-MM-DD-HH-MM-SS

cp "$RUN_DIR/weight.pkl" \
   "$RUN_DIR/weight_lambda_single_0p01_best_iou_e291.pkl"

cp "$RUN_DIR/checkpoint_best_iou.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p01_best_iou_e291.pkl"

cp "$RUN_DIR/weight_pd_fa_best.pkl" \
   "$RUN_DIR/weight_lambda_single_0p01_pdfa_best_e333.pkl"

cp "$RUN_DIR/checkpoint_pd_fa_best.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p01_pdfa_best_e333.pkl"

ls -lh "$RUN_DIR"/weight_lambda_single_0p01_best_iou_e291.pkl \
       "$RUN_DIR"/checkpoint_lambda_single_0p01_best_iou_e291.pkl \
       "$RUN_DIR"/weight_lambda_single_0p01_pdfa_best_e333.pkl \
       "$RUN_DIR"/checkpoint_lambda_single_0p01_pdfa_best_e333.pkl
```

如果某个文件不存在，先不要继续，说明保存逻辑或目录选错了。

---

## 6. 复测 0.01 best-IoU 权重

```bash
cd /home/ly/DEA

export RUN_DIR=/home/ly/DEA/weight/MSHNet-YYYY-MM-DD-HH-MM-SS

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
IoU = 0.6705
PD  = 0.9116
FA  = 7.8951
```

---

## 7. 复测 0.01 PD/FA-best 权重

```bash
cd /home/ly/DEA

export RUN_DIR=/home/ly/DEA/weight/MSHNet-YYYY-MM-DD-HH-MM-SS

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
IoU = 0.6639
PD  = 0.9286
FA  = 7.8951
```

---

## 8. 代码修改 1：新增 checkpoint 归档脚本

新增文件：

```text
tools/archive_dea_checkpoints.sh
```

执行下面命令直接创建：

```bash
cd /home/ly/DEA

mkdir -p tools

cat > tools/archive_dea_checkpoints.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash tools/archive_dea_checkpoints.sh <run_dir> <tag>"
  echo "Example: bash tools/archive_dea_checkpoints.sh /home/ly/DEA/weight/MSHNet-2026-07-06-xx-xx-xx lambda_single_0p01"
  exit 1
fi

RUN_DIR="$1"
TAG="$2"

if [ ! -d "$RUN_DIR" ]; then
  echo "[ERROR] RUN_DIR does not exist: $RUN_DIR"
  exit 2
fi

copy_if_exists() {
  local src="$1"
  local dst="$2"

  if [ -f "$src" ]; then
    cp "$src" "$dst"
    echo "[OK] copied: $src -> $dst"
  else
    echo "[WARN] missing: $src"
  fi
}

copy_if_exists "$RUN_DIR/weight.pkl" \
               "$RUN_DIR/weight_${TAG}_best_iou.pkl"

copy_if_exists "$RUN_DIR/checkpoint_best_iou.pkl" \
               "$RUN_DIR/checkpoint_${TAG}_best_iou.pkl"

copy_if_exists "$RUN_DIR/weight_pd_fa_best.pkl" \
               "$RUN_DIR/weight_${TAG}_pdfa_best.pkl"

copy_if_exists "$RUN_DIR/checkpoint_pd_fa_best.pkl" \
               "$RUN_DIR/checkpoint_${TAG}_pdfa_best.pkl"

echo "[DONE] archive complete."
SH

chmod +x tools/archive_dea_checkpoints.sh
```

使用方式：

```bash
cd /home/ly/DEA

bash tools/archive_dea_checkpoints.sh \
  "$RUN_DIR" \
  lambda_single_0p01
```

这个脚本不会改变原文件，只复制备份。

---

## 9. 代码修改 2：新增 run 结果汇总脚本

新增文件：

```text
tools/dea_run_report.py
```

执行下面命令直接创建：

```bash
cd /home/ly/DEA

mkdir -p tools

cat > tools/dea_run_report.py <<'PY'
import argparse
import csv
import json
import os
import re
from typing import Dict, List, Optional

import torch


def load_torch_file(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_checkpoint(path: str) -> Optional[Dict[str, float]]:
    if not os.path.exists(path):
        return None

    obj = load_torch_file(path)

    if not isinstance(obj, dict):
        return None

    out = {
        "path": path,
        "epoch": obj.get("epoch"),
        "iou": obj.get("iou"),
        "pd": obj.get("pd"),
        "fa": obj.get("fa"),
        "best_iou": obj.get("best_iou"),
        "best_pd_fa": obj.get("best_pd_fa"),
        "best_pd_fa_iou": obj.get("best_pd_fa_iou"),
        "best_pd_fa_pd": obj.get("best_pd_fa_pd"),
        "best_pd_fa_epoch": obj.get("best_pd_fa_epoch"),
    }

    return out


_METRIC_RE = re.compile(
    r".*-\s*(?P<epoch>\d+)\s*-\s*IoU\s*(?P<iou>[0-9.]+)\s*-\s*PD\s*(?P<pd>[0-9.]+)\s*-\s*FA\s*(?P<fa>[0-9.]+)"
)


def parse_metric_log(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []

    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _METRIC_RE.match(line.strip())
            if not m:
                continue
            rows.append(
                {
                    "epoch": int(m.group("epoch")),
                    "iou": float(m.group("iou")),
                    "pd": float(m.group("pd")),
                    "fa": float(m.group("fa")),
                    "source": os.path.basename(path),
                }
            )

    return rows


def format_float(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--baseline-iou", type=float, required=True)
    parser.add_argument("--baseline-pd", type=float, required=True)
    parser.add_argument("--baseline-fa", type=float, required=True)
    parser.add_argument("--out-md", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)

    checkpoints = {
        "latest": read_checkpoint(os.path.join(run_dir, "checkpoint.pkl")),
        "best_iou": read_checkpoint(os.path.join(run_dir, "checkpoint_best_iou.pkl")),
        "pd_fa_best": read_checkpoint(os.path.join(run_dir, "checkpoint_pd_fa_best.pkl")),
    }

    epoch_metrics = parse_metric_log(os.path.join(run_dir, "epoch_metric.log"))
    best_iou_log = parse_metric_log(os.path.join(run_dir, "metric.log"))
    pd_fa_log = parse_metric_log(os.path.join(run_dir, "metric_pd_fa_best.log"))

    summary_rows = []

    baseline = {
        "name": "paired_baseline",
        "epoch": "",
        "iou": args.baseline_iou,
        "pd": args.baseline_pd,
        "fa": args.baseline_fa,
        "delta_iou": 0.0,
        "delta_pd": 0.0,
        "delta_fa": 0.0,
    }
    summary_rows.append(baseline)

    for name, ckpt in checkpoints.items():
        if ckpt is None:
            continue

        iou = ckpt.get("iou")
        pd = ckpt.get("pd")
        fa = ckpt.get("fa")

        if iou is None or pd is None or fa is None:
            continue

        summary_rows.append(
            {
                "name": name,
                "epoch": ckpt.get("epoch"),
                "iou": float(iou),
                "pd": float(pd),
                "fa": float(fa),
                "delta_iou": float(iou) - args.baseline_iou,
                "delta_pd": float(pd) - args.baseline_pd,
                "delta_fa": float(fa) - args.baseline_fa,
            }
        )

    report = {
        "run_dir": run_dir,
        "baseline": baseline,
        "checkpoints": checkpoints,
        "summary_rows": summary_rows,
        "num_epoch_metric_rows": len(epoch_metrics),
        "num_metric_log_rows": len(best_iou_log),
        "num_pd_fa_log_rows": len(pd_fa_log),
        "last_epoch_metric": epoch_metrics[-1] if epoch_metrics else None,
        "last_best_iou_log": best_iou_log[-1] if best_iou_log else None,
        "last_pd_fa_log": pd_fa_log[-1] if pd_fa_log else None,
    }

    out_json = args.out_json or os.path.join(run_dir, "dea_run_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    out_csv = args.out_csv or os.path.join(run_dir, "dea_run_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "epoch",
                "iou",
                "pd",
                "fa",
                "delta_iou",
                "delta_pd",
                "delta_fa",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    out_md = args.out_md or os.path.join(run_dir, "dea_run_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# DEA Run Summary\n\n")
        f.write(f"- Run dir: `{run_dir}`\n")
        f.write(
            f"- Paired baseline: IoU `{args.baseline_iou:.4f}`, "
            f"PD `{args.baseline_pd:.4f}`, FA `{args.baseline_fa:.4f}`\n\n"
        )

        f.write("| name | epoch | IoU | PD | FA | ΔIoU | ΔPD | ΔFA |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summary_rows:
            f.write(
                "| {name} | {epoch} | {iou} | {pd} | {fa} | {delta_iou} | {delta_pd} | {delta_fa} |\n".format(
                    name=row["name"],
                    epoch=row["epoch"],
                    iou=format_float(row["iou"]),
                    pd=format_float(row["pd"]),
                    fa=format_float(row["fa"]),
                    delta_iou=format_float(row["delta_iou"]),
                    delta_pd=format_float(row["delta_pd"]),
                    delta_fa=format_float(row["delta_fa"]),
                )
            )

        f.write("\n## Logs\n\n")
        f.write(f"- epoch_metric.log rows: `{len(epoch_metrics)}`\n")
        f.write(f"- metric.log rows: `{len(best_iou_log)}`\n")
        f.write(f"- metric_pd_fa_best.log rows: `{len(pd_fa_log)}`\n")

        if epoch_metrics:
            f.write(f"- Last epoch metric: `{epoch_metrics[-1]}`\n")
        if best_iou_log:
            f.write(f"- Last best-IoU log: `{best_iou_log[-1]}`\n")
        if pd_fa_log:
            f.write(f"- Last PD/FA-best log: `{pd_fa_log[-1]}`\n")

    print("[OK] wrote:", out_md)
    print("[OK] wrote:", out_json)
    print("[OK] wrote:", out_csv)


if __name__ == "__main__":
    main()
PY
```

编译检查：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile \
  tools/dea_run_report.py
```

运行：

```bash
cd /home/ly/DEA

export RUN_DIR=/home/ly/DEA/weight/MSHNet-YYYY-MM-DD-HH-MM-SS

/home/ly/BasicIRSTD/infrarenet/bin/python tools/dea_run_report.py \
  --run-dir "$RUN_DIR" \
  --baseline-iou 0.6705 \
  --baseline-pd 0.9150 \
  --baseline-fa 9.2616
```

它会生成：

```text
$RUN_DIR/dea_run_summary.md
$RUN_DIR/dea_run_summary.json
$RUN_DIR/dea_run_summary.csv
```

---

## 10. 再跑更保守版本：`lambda_single=0.005 + tau=0.6`

目的：

```text
验证更弱 single-scale anti-sufficiency 是否能进一步保护 IoU / PD；
如果 FA 降幅仍然存在，则 0.005 可能是更稳的主配置；
如果 FA 降幅太小，则 0.01 是当前 IRSTD-1K 主配置。
```

命令：

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

## 11. `0.005 + tau=0.6` 的判断标准

对照：

```text
paired baseline:
IoU = 0.6705
PD  = 0.9150
FA  = 9.2616
```

当前 `0.01` 结果：

```text
best-IoU:
IoU = 0.6705
PD  = 0.9116
FA  = 7.8951

PD/FA-best:
IoU = 0.6639
PD  = 0.9286
FA  = 7.8951
```

`0.005 + tau=0.6` 如果出现下面任意一种，就有价值：

### 情况 A：更稳的 best-IoU

```text
IoU >= 0.6700
PD  >= 0.9150
FA  <  9.2616
```

这比 0.01 best-IoU 更好，因为 0.01 best-IoU 的 PD 是 0.9116。

### 情况 B：更好的 PD/FA-best

```text
IoU >= 0.6605
PD  >= 0.9286
FA  <= 7.8951
```

这说明更保守的配置在不牺牲 FA 的情况下进一步保护 PD。

### 情况 C：FA 降幅太小

如果：

```text
FA 接近 9.2616
```

那 `0.005` 太弱，主配置仍用 `0.01`。

---

## 12. 后续如果 0.005 跑完

跑完后同样执行：

```bash
cd /home/ly/DEA

export RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -1)

bash tools/archive_dea_checkpoints.sh \
  "$RUN_DIR" \
  lambda_single_0p005_tau0p6

/home/ly/BasicIRSTD/infrarenet/bin/python tools/dea_run_report.py \
  --run-dir "$RUN_DIR" \
  --baseline-iou 0.6705 \
  --baseline-pd 0.9150 \
  --baseline-fa 9.2616
```

然后比较：

```text
0.01 best-IoU
0.01 PD/FA-best
0.005 best-IoU
0.005 PD/FA-best
paired DEA-off baseline
```

---

## 13. 当前可以写进实验记录的结论

建议写成：

```text
The paired DEA-off baseline reaches 0.6705 IoU, 0.9150 PD, and 9.2616 FA.
With lambda_single=0.01, DEA-lite preserves the best-IoU operating point
at 0.6705 IoU while reducing FA to 7.8951. At the PD/FA-best checkpoint,
DEA-lite improves PD to 0.9286 and reduces FA to 7.8951, with a small
IoU trade-off to 0.6639.
```

中文：

```text
在 paired 设置下，DEA-off baseline 的 best-IoU 为 0.6705，PD 为 0.9150，FA 为 9.2616。
当 lambda_single=0.01 时，DEA-lite 在 best-IoU 点保持 IoU=0.6705，同时将 FA 降到 7.8951；
在 PD/FA-best 点，DEA-lite 将 PD 提升到 0.9286，并将 FA 降到 7.8951，代价是 IoU 小幅下降到 0.6639。
```

这不是 mIoU SOTA 结论，而是：

```text
DEA-lite 改善了 false-alarm / detection-probability operating trade-off。
```

---

## 14. 当前版本的论文定位

当前证据支持：

```text
MSHNet 的一部分 false alarms 可以被 single-scale counterfactual evidence 触发；
对 z_only_max 的 conservative anti-sufficiency loss 可以降低 FA；
适当权重 lambda_single=0.01 可以得到比 paired baseline 更好的 FA/PD trade-off。
```

当前证据还不支持：

```text
完整 DEA 框架已经成立；
decidability map 已经有效；
inference-time evidence gate 已经可靠；
mIoU/SOTA 显著提升。
```

所以现在写作应该保持：

```text
DEA-lite / CERA-lite first-stage evidence:
training-time counterfactual single-scale anti-sufficiency improves the operating trade-off.
```

不要过度 claim。

---

## 15. 最终建议

当前执行优先级：

```text
1. 归档并复测 0.01 两个 checkpoint。
2. 新增 tools/archive_dea_checkpoints.sh。
3. 新增 tools/dea_run_report.py。
4. 生成 0.01 run summary。
5. 跑 0.005 + tau=0.6。
6. 比较 0.01 与 0.005。
7. 如果 0.005 没有更好，IRSTD-1K 主配置使用 0.01。
8. 暂时不要开 lambda_dec / lambda_empty / d gate。
```

当前 `0.01` 是有效结果，可以保留为 paired positive signal。下一步不是继续改 DEA 结构，而是补齐归档、复测、自动汇总和更保守强度验证。
