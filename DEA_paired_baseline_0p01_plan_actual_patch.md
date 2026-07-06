# DEA-lite paired baseline 后的下一步方案与实际代码修改

> 适配仓库：`https://github.com/Arialliy/DEA`  
> 当前阶段：`/home/ly/DEA` 已经是 DEA-lite fork，不是原始 MSHNet。  
> 当前目标：以新的 paired DEA-off baseline 为主对照，跑 `lambda_single=0.01`，并让 `PD/FA-best` checkpoint 保存逻辑匹配当前 baseline。

---

## 1. 当前结果判断

### 1.1 paired DEA-off baseline 已经跑完

当前 paired baseline：

```text
Best-IoU checkpoint: epoch 222
IoU 0.6705 / PD 0.9150 / FA 9.2616

Final epoch 399:
IoU 0.5976 / PD 0.9116 / FA 14.4996
```

保存文件语义正常：

```text
weight.pkl                  = best-IoU, epoch 222
checkpoint_best_iou.pkl     = best-IoU, epoch 222
checkpoint.pkl              = latest/final, epoch 399
```

没有生成：

```text
checkpoint_pd_fa_best.pkl
```

原因不是保存代码坏了，而是之前阈值太严：

```text
PD >= 0.93
IoU >= 0.655
```

paired baseline 的 best-IoU 点虽然 IoU 和 FA 很好，但 PD 只有：

```text
PD = 0.9150
```

所以它不满足 `PD >= 0.93`。

---

## 2. 新的主对照必须换成 paired baseline

旧 sanity baseline：

```text
IoU 0.6592 / PD 0.9354 / FA 18.9786
```

新的 paired baseline：

```text
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

后续 `lambda_single=0.01` 不能再拿旧 baseline 当主对照。主对照必须是：

```text
paired DEA-off baseline:
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

因此 `lambda_single=0.01` 的目标应改成：

```text
1. IoU 接近 0.6705，最好不低于 0.6605 或 0.6655；
2. PD 不低于约 0.915；
3. FA 低于 9.2616；
4. 如果 FA 接近 baseline，则 IoU / PD 至少要更好。
```

更实际的 checkpoint 保存约束建议：

```text
PD >= 0.915
IoU >= paired_baseline_iou - 0.01
FA <= paired_baseline_fa
```

也就是：

```text
PD >= 0.915
IoU >= 0.6605
FA <= 9.2616
```

如果想更严格，可用：

```text
IoU >= paired_baseline_iou - 0.005 = 0.6655
```

但第一轮 `0.01` 建议先用 `0.01` margin，避免一个 checkpoint 都存不到。

---

## 3. 当前 0.02 run 的定位

之前 `lambda_single=0.02` 的 best：

```text
IoU 0.6658 / PD 0.9218 / FA 9.4134
```

与新的 paired baseline 相比：

```text
baseline: IoU 0.6705 / PD 0.9150 / FA 9.2616
0.02 run: IoU 0.6658 / PD 0.9218 / FA 9.4134
```

它不是明显优于 paired baseline：

```text
PD 略高；
IoU 略低；
FA 略高。
```

所以当前 `0.02` 只能保留为诊断结果：

> single-scale anti-sufficiency 有机会改变 FA / PD trade-off，但 `0.02` 不是最终主结果。

下一步必须重新跑：

```text
lambda_single=0.01
```

---

## 4. 代码修改目标

当前 `main.py` 已经有：

```text
--paired-baseline-iou
--pd-fa-min-pd
--pd-fa-min-iou
--pd-fa-iou-margin
```

但现在还需要补：

```text
--paired-baseline-pd
--paired-baseline-fa
--pd-fa-require-fa-below-baseline
--pd-fa-baseline-fa-margin
```

并把 `PD/FA-best` 逻辑改成：

```text
1. min_pd 可以来自 --pd-fa-min-pd 或 paired baseline PD；
2. min_iou 可以来自 paired_baseline_iou - margin；
3. 如果设置 require-fa-below-baseline，则候选点必须 FA <= paired_baseline_fa - margin；
4. 在满足 PD / IoU / FA-baseline 约束后，再按最低 FA 保存 checkpoint_pd_fa_best.pkl。
```

---

## 5. `main.py` 实际代码修改

下面是实际代码块，不是伪代码。

---

### 5.1 修改 `parse_args()`

在 `parse_args()` 里找到已有的这几行：

```python
parser.add_argument('--pd-fa-min-pd', type=float, default=0.93)
parser.add_argument('--pd-fa-min-iou', type=float, default=0.655)
parser.add_argument('--paired-baseline-iou', type=float, default=0.0)
parser.add_argument('--pd-fa-iou-margin', type=float, default=0.005)
```

替换成下面这一组：

```python
parser.add_argument('--pd-fa-min-pd', type=float, default=0.0)
parser.add_argument('--pd-fa-min-iou', type=float, default=0.655)
parser.add_argument('--paired-baseline-iou', type=float, default=0.0)
parser.add_argument('--paired-baseline-pd', type=float, default=0.0)
parser.add_argument('--paired-baseline-fa', type=float, default=0.0)
parser.add_argument('--pd-fa-iou-margin', type=float, default=0.01)
parser.add_argument('--pd-fa-require-fa-below-baseline', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--pd-fa-baseline-fa-margin', type=float, default=0.0)
```

说明：

```text
pd-fa-min-pd 默认改成 0.0，是为了允许自动使用 paired-baseline-pd。
pd-fa-iou-margin 默认改成 0.01，是为了第一轮不要过严。
```

---

### 5.2 在 `Trainer` 类中新增 `get_pd_fa_thresholds()`

在 `Trainer` 类中，建议放在 `set_optimizer_lr()` 后面，新增完整函数：

```python
def get_pd_fa_thresholds(self):
    min_pd = self.args.pd_fa_min_pd
    if min_pd <= 0 and self.args.paired_baseline_pd > 0:
        min_pd = self.args.paired_baseline_pd

    min_iou = self.args.pd_fa_min_iou
    if self.args.paired_baseline_iou > 0:
        min_iou = max(
            min_iou,
            self.args.paired_baseline_iou - self.args.pd_fa_iou_margin,
        )

    max_fa = None
    if (
        self.args.pd_fa_require_fa_below_baseline
        and self.args.paired_baseline_fa > 0
    ):
        max_fa = self.args.paired_baseline_fa - self.args.pd_fa_baseline_fa_margin

    return min_pd, min_iou, max_fa
```

---

### 5.3 替换 `test()` 里的 PD/FA-best candidate 逻辑

在 `test()` 函数里，找到当前类似这一段的逻辑：

```python
if self.args.paired_baseline_iou > 0:
    pd_fa_iou_threshold = max(
        self.args.pd_fa_min_iou,
        self.args.paired_baseline_iou - self.args.pd_fa_iou_margin,
    )
else:
    pd_fa_iou_threshold = self.args.pd_fa_min_iou

is_pd_fa_candidate = (
    current_pd >= self.args.pd_fa_min_pd
    and mean_IoU >= pd_fa_iou_threshold
    and current_fa < self.best_pd_fa
)
```

替换成下面完整版本：

```python
pd_fa_min_pd, pd_fa_iou_threshold, pd_fa_max_fa = self.get_pd_fa_thresholds()

if pd_fa_max_fa is None:
    pass_fa_baseline = True
else:
    pass_fa_baseline = current_fa <= pd_fa_max_fa

pass_pd_iou_constraints = (
    current_pd >= pd_fa_min_pd
    and mean_IoU >= pd_fa_iou_threshold
    and pass_fa_baseline
)

fa_improves = current_fa < self.best_pd_fa - 1e-6
fa_ties_iou_improves = (
    abs(current_fa - self.best_pd_fa) <= 1e-6
    and mean_IoU > self.best_pd_fa_iou
)

is_pd_fa_candidate = pass_pd_iou_constraints and (
    fa_improves or fa_ties_iou_improves
)
```

这个逻辑的含义是：

```text
先满足 PD / IoU / baseline-FA 约束，
再在满足条件的 checkpoint 里面选择最低 FA。
如果 FA 几乎相同，则选择 IoU 更高的点。
```

---

### 5.4 修改 `latest_states`、`best_iou_states`、`pd_fa_states` 保存字段

找到 `latest_states = { ... }`，在字典里补下面字段：

```python
"pd_fa_min_pd": pd_fa_min_pd,
"pd_fa_min_iou": pd_fa_iou_threshold,
"pd_fa_max_fa": pd_fa_max_fa,
"paired_baseline_iou": self.args.paired_baseline_iou,
"paired_baseline_pd": self.args.paired_baseline_pd,
"paired_baseline_fa": self.args.paired_baseline_fa,
```

即 latest checkpoint 推荐保存成：

```python
latest_states = {
    "net": self.model.state_dict(),
    "optimizer": self.optimizer.state_dict(),
    "epoch": epoch,
    "iou": mean_IoU,
    "pd": current_pd,
    "fa": current_fa,
    "best_iou": self.best_iou,
    "best_pd_fa": self.best_pd_fa,
    "best_pd_fa_iou": self.best_pd_fa_iou,
    "best_pd_fa_pd": self.best_pd_fa_pd,
    "best_pd_fa_epoch": self.best_pd_fa_epoch,
    "pd_fa_min_pd": pd_fa_min_pd,
    "pd_fa_min_iou": pd_fa_iou_threshold,
    "pd_fa_max_fa": pd_fa_max_fa,
    "paired_baseline_iou": self.args.paired_baseline_iou,
    "paired_baseline_pd": self.args.paired_baseline_pd,
    "paired_baseline_fa": self.args.paired_baseline_fa,
}
```

同样，把下面字段也加到 `best_iou_states` 和 `pd_fa_states` 里：

```python
"pd_fa_min_pd": pd_fa_min_pd,
"pd_fa_min_iou": pd_fa_iou_threshold,
"pd_fa_max_fa": pd_fa_max_fa,
"paired_baseline_iou": self.args.paired_baseline_iou,
"paired_baseline_pd": self.args.paired_baseline_pd,
"paired_baseline_fa": self.args.paired_baseline_fa,
```

---

### 5.5 增强 `metric_pd_fa_best.log` 信息

找到保存 `metric_pd_fa_best.log` 的地方，建议替换为：

```python
with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
    f.write(
        '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\t '
        '- minPD {:.4f}\t - minIoU {:.4f}\t - maxFA {}\n'.format(
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
            epoch,
            mean_IoU,
            current_pd,
            current_fa,
            pd_fa_min_pd,
            pd_fa_iou_threshold,
            'None' if pd_fa_max_fa is None else '%.4f' % pd_fa_max_fa,
        )
    )
```

---

## 6. 不需要修改的文件

当前阶段不要改：

```text
model/MSHNet.py
model/loss.py
```

原因：

```text
1. 当前问题不是 DEA 结构本身，而是 paired baseline 变强后保存和对照逻辑要更新；
2. single-scale loss 已经是 conservative 版本；
3. 现在继续改 MSHNet.py / loss.py 会引入新变量，不利于比较 lambda_single=0.01。
```

---

## 7. 编译检查

改完 `main.py` 后执行：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile \
  main.py \
  model/MSHNet.py \
  model/loss.py
```

---

## 8. 本地检查命令

确认新增参数存在：

```bash
cd /home/ly/DEA

grep -n "paired-baseline-pd\|paired-baseline-fa\|pd-fa-require-fa-below-baseline\|pd-fa-baseline-fa-margin" main.py
```

确认 helper 存在：

```bash
grep -n "def get_pd_fa_thresholds" main.py
```

确认 test 里使用新逻辑：

```bash
grep -n "pass_pd_iou_constraints\|pass_fa_baseline\|fa_ties_iou_improves" main.py
```

---

## 9. 下一步训练命令：`lambda_single=0.01`

先跑这一版：

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
  --dea-lambda-single 0.01 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.5 \
  --dea-detach-evidence \
  --save-dea-debug \
  --paired-baseline-iou 0.6705 \
  --paired-baseline-pd 0.9150 \
  --paired-baseline-fa 9.2616 \
  --pd-fa-min-pd 0.915 \
  --pd-fa-min-iou 0.655 \
  --pd-fa-iou-margin 0.01 \
  --pd-fa-require-fa-below-baseline true \
  --pd-fa-baseline-fa-margin 0.0
```

这个命令对应的 PD/FA-best 保存约束是：

```text
PD >= 0.915
IoU >= max(0.655, 0.6705 - 0.01) = 0.6605
FA <= 9.2616
```

如果生成了：

```text
checkpoint_pd_fa_best.pkl
weight_pd_fa_best.pkl
```

说明 `lambda_single=0.01` 至少找到了一个：

```text
PD 不低于 paired baseline；
IoU 接近 paired baseline；
FA 不高于 paired baseline。
```

这样的 operating point。

---

## 10. 如果 `0.01` 没有生成 PD/FA-best

先看 `epoch_metric.log`，不要马上开 `lambda_dec`。

```bash
cd /home/ly/DEA

RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -1)
echo "$RUN_DIR"
tail -n 30 "$RUN_DIR/epoch_metric.log"
```

如果 `0.01` 的 best-IoU 很好，但没有满足 `FA <= 9.2616`，可以再跑更弱版：

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
  --paired-baseline-pd 0.9150 \
  --paired-baseline-fa 9.2616 \
  --pd-fa-min-pd 0.915 \
  --pd-fa-min-iou 0.655 \
  --pd-fa-iou-margin 0.01 \
  --pd-fa-require-fa-below-baseline true \
  --pd-fa-baseline-fa-margin 0.0
```

---

## 11. 如果仍然没有 checkpoint_pd_fa_best.pkl

这不一定说明 DEA 无效。它可能说明 paired baseline 本身已经非常强：

```text
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

下一步可以临时放宽保存约束，只用于观察，不作为正式主结果：

```bash
--pd-fa-require-fa-below-baseline false
```

或者把 IoU margin 放宽：

```bash
--pd-fa-iou-margin 0.015
```

但正式主对照仍应报告：

```text
paired baseline best-IoU
DEA best-IoU
DEA PD/FA-best under matched PD/IoU constraints
```

---

## 12. 当前不要做

继续不要开：

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
当前 paired baseline 已经很强。
现在最重要的是确认 single-scale anti-sufficiency 在弱权重下是否能带来 matched-PD/IoU 的 FA 改善。
如果此时加 decidability loss 或 d gate，会重新引入 Pd 下降风险。
```

---

## 13. 最终执行顺序

```text
1. 修改 main.py：增加 paired-baseline-pd/fa 和 FA-baseline 约束。
2. py_compile。
3. grep 检查新参数和新逻辑。
4. 跑 lambda_single=0.01。
5. 检查：weight.pkl、checkpoint_best_iou.pkl、checkpoint.pkl、checkpoint_pd_fa_best.pkl。
6. 如果 0.01 没有 matched-PD/IoU 的 FA 改善，再跑 0.005 + tau=0.6。
7. 继续不要开 lambda_dec / lambda_empty / d gate。
```

---

## 14. 当前阶段的判断标准

`lambda_single=0.01` 如果出现下面任一结果，就值得保留：

### 情况 A：best-IoU 超 baseline

```text
IoU > 0.6705
PD >= 0.915
FA <= 或接近 9.2616
```

### 情况 B：PD/FA-best 优于 baseline

```text
IoU >= 0.6605
PD >= 0.915
FA < 9.2616
```

### 情况 C：同 FA 下更高 PD / IoU

```text
FA ≈ 9.2616
PD > 0.915 或 IoU > 0.6705
```

如果 `0.01` 和 `0.005` 都无法超过 paired baseline，则说明：

```text
当前 IRSTD-1K paired baseline 已经很强，DEA-lite single-only 在这个配置下没有明显主表收益。
```

但 `0.02` 仍可作为诊断证据：它说明 single-scale anti-sufficiency 能改变 false-alarm behavior，只是需要更好的 candidate selection 或 decidability modeling 才能进一步提升主指标。
