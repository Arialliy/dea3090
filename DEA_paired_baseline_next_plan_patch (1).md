# DEA-lite paired baseline 后的下一步方案与代码修改

> 适配仓库：`https://github.com/Arialliy/DEA`  
> 当前本地路径：`/home/ly/DEA`  
> 当前阶段：DEA-lite single-scale anti-sufficiency 调参，不进入 full DEA。  
> 目标：以新的 paired DEA-off baseline 为主对照，跑 `lambda_single=0.01`，并修正 PD/FA-best checkpoint 保存逻辑。

---

## 1. 当前结果重新判断

### 1.1 paired DEA-off baseline 已经跑完

你的 paired DEA-off baseline：

```text
Best-IoU checkpoint: epoch 222
IoU 0.6705 / PD 0.9150 / FA 9.2616

Final epoch 399:
IoU 0.5976 / PD 0.9116 / FA 14.4996
```

文件语义现在是正确的：

```text
weight.pkl                  = best-IoU 权重，epoch 222
checkpoint_best_iou.pkl     = best-IoU checkpoint，epoch 222
checkpoint.pkl              = latest/final checkpoint，epoch 399
```

没有生成：

```text
checkpoint_pd_fa_best.pkl
```

原因是原来的约束太硬：

```text
PD >= 0.93
IoU >= 0.655
```

而 paired baseline best-IoU 点是：

```text
PD = 0.9150
```

所以 `PD >= 0.93` 会导致 baseline 里没有任何点满足 PD/FA-best 保存条件。

---

## 2. 新主对照必须切换到 paired baseline

旧 sanity baseline：

```text
IoU 0.6592 / PD 0.9354 / FA 18.9786
```

新的 paired DEA-off baseline：

```text
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

所以后续 `lambda_single=0.01` 不能再拿旧 sanity baseline 当主对照。

后续主对照必须是：

```text
paired baseline:
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

当前 `lambda_single=0.02` 的 best-IoU 结果是：

```text
IoU 0.6658 / PD 0.9218 / FA 9.4134
```

相对于新的 paired baseline：

```text
IoU: 0.6658 vs 0.6705    略低
PD : 0.9218 vs 0.9150    略高
FA : 9.4134 vs 9.2616    略高
```

所以 `lambda_single=0.02` 不能再说是主结果明显优于 baseline。它更适合作为诊断结果：

> single-scale anti-sufficiency 能改变 FA/PD operating point，但当前 0.02 没有稳定形成优于 paired DEA-off baseline 的 Pareto improvement。

---

## 3. 下一步目标

下一步跑 `lambda_single=0.01`。

目标不是盲目降低 FA，而是相对 paired baseline 建立更合理的 operating point：

```text
paired baseline:
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

`lambda_single=0.01` 需要尽量满足：

```text
IoU 接近 0.6705，最好 >= 0.6655 或 >= 0.6605；
PD 不低于约 0.915；
FA 低于 9.2616，或者在同等 FA 下 IoU/PD 更好。
```

建议保存 checkpoint 的约束从：

```text
PD >= 0.93
```

改成：

```text
PD >= 0.915
IoU >= paired_baseline_iou - margin
```

第一轮 margin 建议：

```text
pd_fa_iou_margin = 0.01
```

也就是：

```text
IoU >= 0.6705 - 0.01 = 0.6605
```

如果你想更严格，可以用：

```text
pd_fa_iou_margin = 0.005
IoU >= 0.6655
```

---

## 4. 需要修改的代码点

当前 `main.py` 已经有 DEA-lite 参数和训练逻辑，但 PD/FA-best 阈值需要与 paired baseline 绑定。

需要做 4 个修改：

```text
1. argparse 增加 paired baseline 参数；
2. Trainer 增加 PD/FA threshold 计算函数；
3. test() 里保存 PD/FA-best 时使用 paired baseline 约束；
4. checkpoint.pkl 继续保持 latest，checkpoint_best_iou.pkl 保持 best-IoU。
```

---

## 5. `main.py` 代码修改 1：增加 argparse 参数

在 `parse_args()` 里找到 DEA 参数区域：

```python
parser.add_argument('--dea-lambda-single', type=float, default=0.0)
parser.add_argument('--dea-lambda-dec', type=float, default=0.0)
parser.add_argument('--dea-lambda-empty', type=float, default=0.0)
parser.add_argument('--dea-tau', type=float, default=0.5)
parser.add_argument('--dea-ramp-epochs', type=int, default=0)
parser.add_argument('--save-dea-debug', action='store_true')
parser.add_argument('--dea-debug-interval', type=int, default=50)
parser.add_argument('--dea-debug-max-batches', type=int, default=1)
parser.add_argument('--dea-detach-evidence', action='store_true')
```

在后面直接追加：

```python
parser.add_argument('--paired-baseline-iou', type=float, default=0.0)
parser.add_argument('--paired-baseline-pd', type=float, default=0.0)
parser.add_argument('--paired-baseline-fa', type=float, default=0.0)

parser.add_argument('--pd-fa-min-pd', type=float, default=0.0)
parser.add_argument('--pd-fa-min-iou', type=float, default=-1.0)
parser.add_argument('--pd-fa-iou-margin', type=float, default=0.01)
parser.add_argument('--pd-fa-pd-margin', type=float, default=0.0)
parser.add_argument('--pd-fa-require-fa-below-baseline', type=str2bool, nargs='?', const=True, default=False)
```

含义：

```text
--paired-baseline-iou              paired DEA-off baseline 的 IoU
--paired-baseline-pd               paired DEA-off baseline 的 PD
--paired-baseline-fa               paired DEA-off baseline 的 FA
--pd-fa-min-pd                     PD/FA-best 的最小 PD
--pd-fa-min-iou                    显式指定最小 IoU；如果为 -1，则用 baseline_iou - margin
--pd-fa-iou-margin                 IoU 允许比 paired baseline 低多少
--pd-fa-pd-margin                  PD 允许比 paired baseline 低多少
--pd-fa-require-fa-below-baseline  是否要求 FA 必须低于 paired baseline FA 才保存
```

---

## 6. `main.py` 代码修改 2：初始化 best 变量

在 `Trainer.__init__` 里找到：

```python
self.best_iou = 0
self.warm_epoch = args.warm_epoch
```

替换成：

```python
self.best_iou = 0.0
self.best_iou_epoch = -1

self.best_pd_fa = float('inf')
self.best_pd_fa_iou = 0.0
self.best_pd_fa_pd = 0.0
self.best_pd_fa_epoch = -1

self.warm_epoch = args.warm_epoch
```

---

## 7. `main.py` 代码修改 3：resume 逻辑同步改

找到 checkpoint resume 里的这段：

```python
self.start_epoch = checkpoint['epoch']+1
self.best_iou = checkpoint['iou']
self.save_folder = check_folder
```

替换成：

```python
self.start_epoch = checkpoint.get('epoch', -1) + 1
self.best_iou = checkpoint.get('best_iou', checkpoint.get('iou', 0.0))
self.best_iou_epoch = checkpoint.get('best_iou_epoch', checkpoint.get('epoch', -1))

self.best_pd_fa = checkpoint.get('best_pd_fa', float('inf'))
self.best_pd_fa_iou = checkpoint.get('best_pd_fa_iou', 0.0)
self.best_pd_fa_pd = checkpoint.get('best_pd_fa_pd', 0.0)
self.best_pd_fa_epoch = checkpoint.get('best_pd_fa_epoch', -1)

self.save_folder = check_folder
```

原因：

```text
checkpoint.pkl 现在是 latest/resume checkpoint，不一定是 best-IoU checkpoint。
所以恢复时不能再把 checkpoint['iou'] 直接当 best_iou。
```

---

## 8. `main.py` 代码修改 4：增加 threshold 计算函数

在 `Trainer` 类里，建议放在 `use_dea()` 函数前后，加入：

```python
def get_pd_fa_thresholds(self):
    args = self.args

    if args.pd_fa_min_pd > 0:
        min_pd = args.pd_fa_min_pd
    elif args.paired_baseline_pd > 0:
        min_pd = max(0.0, args.paired_baseline_pd - args.pd_fa_pd_margin)
    else:
        min_pd = 0.0

    if args.pd_fa_min_iou >= 0:
        min_iou = args.pd_fa_min_iou
    elif args.paired_baseline_iou > 0:
        min_iou = max(0.0, args.paired_baseline_iou - args.pd_fa_iou_margin)
    else:
        min_iou = 0.0

    return min_pd, min_iou
```

---

## 9. `main.py` 代码修改 5：替换 `test()` 里的 train-mode 保存逻辑

在 `test()` 函数里找到：

```python
if self.mode == 'train':
    current_pd = PD[0]
    current_fa = FA[0] * 1000000
    metric_line = '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
        time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
        epoch, mean_IoU, current_pd, current_fa,
    )
    print(metric_line.strip())
    with open(osp.join(self.save_folder, 'epoch_metric.log'), 'a') as f:
        f.write(metric_line)

    if mean_IoU > self.best_iou:
        self.best_iou = mean_IoU
        torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
        with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
            f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), epoch, self.best_iou, current_pd, current_fa))
        all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
        torch.save(all_states, osp.join(self.save_folder, 'checkpoint.pkl'))
```

替换成下面完整代码：

```python
if self.mode == 'train':
    current_pd = float(PD[0])
    current_fa = float(FA[0] * 1000000)
    current_iou = float(mean_IoU)

    delta_iou = None
    delta_pd = None
    delta_fa = None
    if self.args.paired_baseline_iou > 0:
        delta_iou = current_iou - self.args.paired_baseline_iou
    if self.args.paired_baseline_pd > 0:
        delta_pd = current_pd - self.args.paired_baseline_pd
    if self.args.paired_baseline_fa > 0:
        delta_fa = current_fa - self.args.paired_baseline_fa

    metric_line = '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}'.format(
        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
        epoch,
        current_iou,
        current_pd,
        current_fa,
    )

    if delta_iou is not None:
        metric_line += '\t - dIoU {:+.4f}'.format(delta_iou)
    if delta_pd is not None:
        metric_line += '\t - dPD {:+.4f}'.format(delta_pd)
    if delta_fa is not None:
        metric_line += '\t - dFA {:+.4f}'.format(delta_fa)
    metric_line += '\n'

    print(metric_line.strip())
    with open(osp.join(self.save_folder, 'epoch_metric.log'), 'a') as f:
        f.write(metric_line)

    # ------------------------------------------------------------
    # 1. best-IoU checkpoint
    # ------------------------------------------------------------
    if current_iou > self.best_iou:
        self.best_iou = current_iou
        self.best_iou_epoch = epoch

        torch.save(
            self.model.state_dict(),
            osp.join(self.save_folder, 'weight.pkl')
        )

        best_iou_states = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "iou": current_iou,
            "pd": current_pd,
            "fa": current_fa,
            "best_iou": self.best_iou,
            "best_iou_epoch": self.best_iou_epoch,
            "best_pd_fa": self.best_pd_fa,
            "best_pd_fa_iou": self.best_pd_fa_iou,
            "best_pd_fa_pd": self.best_pd_fa_pd,
            "best_pd_fa_epoch": self.best_pd_fa_epoch,
            "paired_baseline_iou": self.args.paired_baseline_iou,
            "paired_baseline_pd": self.args.paired_baseline_pd,
            "paired_baseline_fa": self.args.paired_baseline_fa,
        }

        torch.save(
            best_iou_states,
            osp.join(self.save_folder, 'checkpoint_best_iou.pkl')
        )

        with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
            f.write(
                '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
                    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                    epoch,
                    self.best_iou,
                    current_pd,
                    current_fa,
                )
            )

    # ------------------------------------------------------------
    # 2. PD-constrained FA-best checkpoint
    # ------------------------------------------------------------
    min_pd, min_iou = self.get_pd_fa_thresholds()

    is_pd_fa_candidate = (
        current_pd + 1e-12 >= min_pd
        and current_iou + 1e-12 >= min_iou
        and current_fa < self.best_pd_fa
    )

    if (
        self.args.pd_fa_require_fa_below_baseline
        and self.args.paired_baseline_fa > 0
    ):
        is_pd_fa_candidate = (
            is_pd_fa_candidate
            and current_fa < self.args.paired_baseline_fa
        )

    if is_pd_fa_candidate:
        self.best_pd_fa = current_fa
        self.best_pd_fa_iou = current_iou
        self.best_pd_fa_pd = current_pd
        self.best_pd_fa_epoch = epoch

        torch.save(
            self.model.state_dict(),
            osp.join(self.save_folder, 'weight_pd_fa_best.pkl')
        )

        pd_fa_states = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "iou": current_iou,
            "pd": current_pd,
            "fa": current_fa,
            "best_iou": self.best_iou,
            "best_iou_epoch": self.best_iou_epoch,
            "best_pd_fa": self.best_pd_fa,
            "best_pd_fa_iou": self.best_pd_fa_iou,
            "best_pd_fa_pd": self.best_pd_fa_pd,
            "best_pd_fa_epoch": self.best_pd_fa_epoch,
            "pd_fa_min_pd": min_pd,
            "pd_fa_min_iou": min_iou,
            "paired_baseline_iou": self.args.paired_baseline_iou,
            "paired_baseline_pd": self.args.paired_baseline_pd,
            "paired_baseline_fa": self.args.paired_baseline_fa,
        }

        torch.save(
            pd_fa_states,
            osp.join(self.save_folder, 'checkpoint_pd_fa_best.pkl')
        )

        with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
            f.write(
                '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\t - minPD {:.4f}\t - minIoU {:.4f}\n'.format(
                    time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                    epoch,
                    current_iou,
                    current_pd,
                    current_fa,
                    min_pd,
                    min_iou,
                )
            )

    # ------------------------------------------------------------
    # 3. latest checkpoint for resume
    # ------------------------------------------------------------
    latest_states = {
        "net": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "epoch": epoch,
        "iou": current_iou,
        "pd": current_pd,
        "fa": current_fa,
        "best_iou": self.best_iou,
        "best_iou_epoch": self.best_iou_epoch,
        "best_pd_fa": self.best_pd_fa,
        "best_pd_fa_iou": self.best_pd_fa_iou,
        "best_pd_fa_pd": self.best_pd_fa_pd,
        "best_pd_fa_epoch": self.best_pd_fa_epoch,
        "paired_baseline_iou": self.args.paired_baseline_iou,
        "paired_baseline_pd": self.args.paired_baseline_pd,
        "paired_baseline_fa": self.args.paired_baseline_fa,
    }

    torch.save(
        latest_states,
        osp.join(self.save_folder, 'checkpoint.pkl')
    )
```

---

## 10. `model/loss.py` 暂时不用改

当前 `model/loss.py` 的 single-scale loss 已经是 conservative 版本：

```python
hard_bg = safe_bg * (torch.sigmoid(z_only_max.detach()) > tau).float()
```

现在不要改回使用：

```python
z_full + z_only_max union
```

也不要开：

```text
lambda_dec
lambda_empty
```

---

## 11. 编译检查

改完后执行：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile \
  main.py \
  model/MSHNet.py \
  model/loss.py
```

---

## 12. 不需要重跑 paired DEA-off baseline

你已经有 paired DEA-off baseline：

```text
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

如果这次代码修改只影响 checkpoint 保存逻辑和 threshold 逻辑，不影响训练 forward/loss，那么不需要重跑 baseline。

但后续 `lambda_single=0.01` 必须带上 paired baseline 参数。

---

## 13. 运行 `lambda_single=0.01`

第一版建议用较宽松的 IoU margin：

```text
pd_fa_iou_margin = 0.01
```

这样保存条件为：

```text
PD  >= 0.915
IoU >= 0.6605
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
  --pd-fa-iou-margin 0.01
```

如果你想只保存真正 FA 低于 paired baseline 的点，加：

```bash
  --pd-fa-require-fa-below-baseline true
```

但第一轮不建议加这个开关，因为它可能导致没有 `checkpoint_pd_fa_best.pkl`。建议先保存所有满足 PD/IoU 约束的最低 FA 点，再人工比较它是否优于 baseline。

---

## 14. 如果 0.01 有结果后怎么判定

主对照：

```text
paired baseline:
IoU 0.6705 / PD 0.9150 / FA 9.2616
```

`lambda_single=0.01` 如果出现以下情况，才算有增益：

### 情况 A：更低 FA，PD/IoU 基本不掉

```text
IoU >= 0.6605
PD  >= 0.915
FA  < 9.2616
```

这是最直接的正结果。

### 情况 B：IoU 更高，FA 接近

```text
IoU > 0.6705
PD  >= 0.915
FA  <= 9.2616 + 少量容忍
```

这可以作为另一个 operating point。

### 情况 C：PD 更高，FA 接近

```text
PD  > 0.915
IoU >= 0.6655
FA  接近 9.2616
```

这也可以作为 Pareto point，但不能说 FA 显著下降。

---

## 15. 如果 0.01 仍然没有优于 paired baseline

再跑更保守版本：

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
  --pd-fa-iou-margin 0.01
```

这版更保守：

```text
lambda_single: 0.01 -> 0.005
tau: 0.5 -> 0.6
```

它的目标是尽量不改变 PD/IoU，只看是否还能略降 FA。

---

## 16. 当前继续不要做

继续不要开：

```text
lambda_dec > 0
lambda_empty > 0
inference-time d gate
component selector
all-16 subset
positive necessity
candidate verifier
```

原因：现在 paired baseline 已经很强，`lambda_single=0.02` 相对 paired baseline 没有形成明显 Pareto improvement。此时不应该增加更多变量。

---

## 17. 结果记录建议

跑完 `0.01` 后记录三个 checkpoint：

```text
weight.pkl
checkpoint_best_iou.pkl
checkpoint_pd_fa_best.pkl
```

并整理：

```text
1. best-IoU result
2. PD/FA-best result
3. final epoch result
4. 对 paired baseline 的 delta
```

建议记录表：

```text
Method                         Epoch   IoU     PD      FA
DEA-off paired baseline         222    0.6705  0.9150  9.2616
DEA-lite lambda_single=0.02     340    0.6658  0.9218  9.4134
DEA-lite lambda_single=0.01     TBD    TBD     TBD     TBD
DEA-lite lambda_single=0.005    TBD    TBD     TBD     TBD
```

---

## 18. 最终执行顺序

```text
1. 修改 main.py：增加 paired baseline 参数。
2. 修改 main.py：增加 get_pd_fa_thresholds()。
3. 修改 main.py：替换 test() 里的 train-mode 保存逻辑。
4. py_compile。
5. 不重跑 paired DEA-off baseline。
6. 直接跑 lambda_single=0.01。
7. 查看 weight.pkl / checkpoint_best_iou.pkl / checkpoint_pd_fa_best.pkl。
8. 如果 0.01 没有优于 paired baseline，再跑 0.005 + tau=0.6。
```

