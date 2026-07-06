# DEA-lite after 400 epochs: 下一步方案与实际代码修改

> 适配仓库：`https://github.com/Arialliy/DEA`  
> 本文基于当前 `/home/ly/DEA` 的 DEA-lite fork，不再按“原始 MSHNet 还没改”的假设来写。  
> 当前阶段目标：保留 `lambda_single=0.02` 的正结果，同时跑更保守的 `lambda_single=0.01`，并修改代码保存 **best-IoU checkpoint** 与 **PD-constrained FA-best checkpoint** 两类 operating point。

---

## 1. 当前 run 结论

当前 run 已经完整跑完 400 epochs：

```text
final epoch 399:
IoU 0.6169 / PD 0.9116 / FA 29.2270

best epoch 340:
IoU 0.6658 / PD 0.9218 / FA 9.4134

epoch 346:
IoU 0.6572 / PD 0.9286 / FA 9.1097

epoch 358:
IoU 0.6585 / PD 0.9286 / FA 9.8689
```

clean sanity baseline best：

```text
IoU 0.6592 / PD 0.9354 / FA 18.9786
```

对比 best epoch 340：

```text
IoU: 0.6592 -> 0.6658   +0.0066
PD : 0.9354 -> 0.9218   -0.0136
FA : 18.9786 -> 9.4134  -9.5652
```

结论：

```text
1. 这次 run 有正结果。
2. lambda_single=0.02 明显降低 FA。
3. IoU 比 clean baseline 略升。
4. PD 有小幅下降。
5. final epoch 明显退化，不能用 final epoch。
6. 当前 run 应该只使用 best checkpoint。
```

因此，当前结果可以作为：

```text
DEA-lite single-only diagnostic result:
单尺度 anti-sufficiency 能压 FA，但 0.02 仍略保守，会牺牲部分 PD。
```

---

## 2. 当前 run 不要重跑，也不要用 final epoch

当前代码在 `mean_IoU > self.best_iou` 时保存：

```text
weight.pkl
checkpoint.pkl
metric.log
```

所以如果 `metric.log` 最后一条 best 是 epoch 340，则当前目录里的：

```text
weight.pkl
checkpoint.pkl
```

应该对应 epoch 340，而不是 final epoch 399。

先保留当前 run：

```bash
cd /home/ly/DEA
RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -n 1)

echo "Current run dir: $RUN_DIR"
tail -n 5 "$RUN_DIR/metric.log"
tail -n 10 "$RUN_DIR/epoch_metric.log"

cp "$RUN_DIR/weight.pkl" "$RUN_DIR/weight_lambda_single_0p02_best_iou_e340.pkl"
cp "$RUN_DIR/checkpoint.pkl" "$RUN_DIR/checkpoint_lambda_single_0p02_best_iou_e340.pkl"
cp "$RUN_DIR/metric.log" "$RUN_DIR/metric_lambda_single_0p02_best_iou.log"
```

如果 `RUN_DIR` 不是这次 `0.02` 的目录，就手动改成对应目录。

---

## 3. 现在最大的问题：当前代码只保存 best IoU，不保存 PD/FA 更优的点

这次 run 里：

```text
epoch 340: IoU 0.6658 / PD 0.9218 / FA 9.4134
epoch 346: IoU 0.6572 / PD 0.9286 / FA 9.1097
```

如果论文目标是 **FA 降低且 PD 尽量保持**，epoch 346 也是一个有价值的 operating point。  
但当前代码只在 `mean_IoU > best_iou` 时保存 checkpoint，所以 epoch 346 的权重一般不会被保存。

因此下一步需要加一个额外 checkpoint 逻辑：

```text
weight.pkl                     # 继续保存 best IoU
checkpoint.pkl                 # 继续保存 best IoU checkpoint
weight_pd_fa_best.pkl          # 新增：在 PD/IoU 约束下保存 FA 最低的模型
checkpoint_pd_fa_best.pkl      # 新增：对应 checkpoint
metric_pd_fa_best.log          # 新增：记录 PD-constrained FA-best 的刷新点
```

这不是改模型结构，而是改 checkpoint selection。它能避免下一次 run 中丢掉 epoch 346 这类更适合 FA/PD trade-off 的点。

---

## 4. `main.py` 实际代码修改 1：测试权重加载兼容 `weight.pkl` / `checkpoint.pkl`

当前训练保存：

```python
torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
```

但 test 模式原来常见写法是按官方 tar 格式读取：

```python
weight = load_torch_file(args.weight_path)
self.load_model_state(weight['state_dict'])
```

这会导致 `weight.pkl` 或 `checkpoint.pkl` 测试不方便。直接把 `main.py` 中 test 加载部分替换为下面这段。

### 替换位置

在 `Trainer.__init__` 里找到：

```python
if args.mode=='test':
    weight = load_torch_file(args.weight_path)
    self.load_model_state(weight['state_dict'])
```

### 替换成

```python
if args.mode=='test':
    weight = load_torch_file(args.weight_path)

    if isinstance(weight, dict) and 'state_dict' in weight:
        self.load_model_state(weight['state_dict'])
    elif isinstance(weight, dict) and 'net' in weight:
        self.load_model_state(weight['net'])
    else:
        self.load_model_state(weight)
```

这样下面三种文件都能直接测试：

```text
official weight tar:   contains state_dict
checkpoint.pkl:        contains net
weight.pkl:            raw state_dict
```

---

## 5. `main.py` 实际代码修改 2：增加 PD/FA-best 保存参数

在 `parse_args()` 里，找到 DEA 参数区域：

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

在后面追加：

```python
parser.add_argument('--save-pd-fa-best', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--pd-floor', type=float, default=0.92)
parser.add_argument('--iou-floor', type=float, default=0.655)
```

默认解释：

```text
--pd-floor 0.92
    只保存 PD >= 0.92 的 FA-best checkpoint。

--iou-floor 0.655
    只保存 IoU >= 0.655 的 FA-best checkpoint。

--save-pd-fa-best true
    默认开启，不影响原本的 best-IoU checkpoint。
```

这组默认值适合你当前结果：

```text
epoch 340: IoU 0.6658 / PD 0.9218 / FA 9.4134  合格
epoch 346: IoU 0.6572 / PD 0.9286 / FA 9.1097  合格，且 FA 更低
```

---

## 6. `main.py` 实际代码修改 3：初始化 PD/FA-best 状态

在 `Trainer.__init__` 里找到：

```python
self.best_iou = 0
self.warm_epoch = args.warm_epoch
```

替换成：

```python
self.best_iou = 0
self.best_pd_constrained_fa = float('inf')
self.warm_epoch = args.warm_epoch
```

如果有 resume checkpoint 逻辑，找到：

```python
self.best_iou = checkpoint['iou']
```

替换成：

```python
self.best_iou = checkpoint['iou']
self.best_pd_constrained_fa = checkpoint.get('pd_constrained_fa', float('inf'))
```

---

## 7. `main.py` 实际代码修改 4：保存 PD-constrained FA-best checkpoint

在 `test()` 里找到原始 best-IoU 保存逻辑：

```python
if mean_IoU > self.best_iou:
    self.best_iou = mean_IoU

    torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
        format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
        epoch, self.best_iou, current_pd, current_fa))

    all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}

    torch.save(all_states, osp.join(self.save_folder, 'checkpoint.pkl'))
```

替换成下面这一整段：

```python
if mean_IoU > self.best_iou:
    self.best_iou = mean_IoU

    torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
            epoch,
            self.best_iou,
            current_pd,
            current_fa,
        ))

    all_states = {
        "net": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "epoch": epoch,
        "iou": self.best_iou,
        "pd": current_pd,
        "fa": current_fa,
        "pd_constrained_fa": self.best_pd_constrained_fa,
    }

    torch.save(all_states, osp.join(self.save_folder, 'checkpoint.pkl'))

if self.args.save_pd_fa_best:
    meets_pd_fa_constraint = (
        current_pd >= self.args.pd_floor
        and mean_IoU >= self.args.iou_floor
    )

    if meets_pd_fa_constraint and current_fa < self.best_pd_constrained_fa:
        self.best_pd_constrained_fa = current_fa

        torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight_pd_fa_best.pkl'))

        with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
            f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\t - pd_floor {:.4f}\t - iou_floor {:.4f}\n'.format(
                time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                epoch,
                mean_IoU,
                current_pd,
                current_fa,
                self.args.pd_floor,
                self.args.iou_floor,
            ))

        pd_fa_states = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "iou": mean_IoU,
            "pd": current_pd,
            "fa": current_fa,
            "pd_constrained_fa": self.best_pd_constrained_fa,
        }

        torch.save(pd_fa_states, osp.join(self.save_folder, 'checkpoint_pd_fa_best.pkl'))
```

这个修改保留原有 `weight.pkl` / `checkpoint.pkl`，同时新增一套更适合 DEA-lite 目标的 checkpoint。

---

## 8. 当前 `model/loss.py` 不需要再改

重新核对后，当前 `model/loss.py` 已经是 conservative single-scale loss：

```python
hard_bg = safe_bg * (torch.sigmoid(z_only_max.detach()) > tau).float()
```

并且 `dea_lite_loss()` 里默认：

```python
lambda_single=0.0
lambda_dec=0.0
lambda_empty=0.0
tau=0.5
```

所以现在不要改 `loss.py`。  
下一步只调命令行参数。

---

## 9. 当前 `model/MSHNet.py` 不需要再改

重新核对后，当前 `model/MSHNet.py` 已经做了这些事：

```text
1. build_dea_lite_outputs(scale_logits, z_full, detach_evidence=False)
2. detach_evidence=True 时 detach scale_logits 和 z_full
3. 用 grouped conv 一次性算 z_only
4. z_only_max / z_only_var
5. decidability_head
6. forward(..., return_dea=False, dea_detach_evidence=False)
```

所以现在不要再改模型结构。

---

## 10. 代码检查命令

改完 `main.py` 后运行：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile main.py model/MSHNet.py model/loss.py

grep -R "save-pd-fa-best\|pd-floor\|iou-floor" -n main.py
grep -R "weight_pd_fa_best\|checkpoint_pd_fa_best\|metric_pd_fa_best" -n main.py
grep -R "state_dict'\|net'" -n main.py | head -n 20
```

预期能看到：

```text
--save-pd-fa-best
--pd-floor
--iou-floor
weight_pd_fa_best.pkl
checkpoint_pd_fa_best.pkl
metric_pd_fa_best.log
state_dict / net 兼容加载逻辑
```

---

## 11. 测试当前 0.02 best checkpoint

改完 test loader 兼容逻辑后，可以直接测试当前 run 的 best-IoU 权重。

```bash
cd /home/ly/DEA
RUN_DIR=$(ls -td /home/ly/DEA/weight/MSHNet-* | head -n 1)

echo "Testing best-IoU weight from: $RUN_DIR"

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/weight.pkl"
```

如果你想测 checkpoint 文件：

```bash
CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/checkpoint.pkl"
```

两个结果应该对应 best-IoU checkpoint。

---

## 12. 下一步主实验：跑 `lambda_single=0.01`

不要从 `0.02` 的 checkpoint resume。  
为了公平比较，应重新从头跑同配置。

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
  --dea-lambda-single 0.01 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.5 \
  --dea-detach-evidence \
  --save-dea-debug \
  --save-pd-fa-best true \
  --pd-floor 0.92 \
  --iou-floor 0.655
```

目标：

```text
IoU >= 0.6592 附近
PD  >= 0.93 更好
FA  < 18.9786，理想低于 12~14
```

如果 `0.01` 结果类似下面这种，就比 `0.02` 更适合作为主结果：

```text
IoU 0.660+
PD  0.930+
FA  10~14
```

---

## 13. 如果 `0.01` 仍然掉 PD

如果 `lambda_single=0.01` 的 best checkpoint 仍然满足：

```text
PD < 0.925
```

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
  --dea-lambda-single 0.005 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.6 \
  --dea-detach-evidence \
  --save-dea-debug \
  --save-pd-fa-best true \
  --pd-floor 0.92 \
  --iou-floor 0.655
```

这个版本的目标是最大程度保护 PD：

```text
PD 尽量接近 0.9354
FA 低于 18.9786 即可
IoU 不低于 baseline 太多
```

---

## 14. 如果 `0.01` 保住 PD 但 FA 降得不够

如果 `0.01` 出现：

```text
PD >= 0.93
IoU >= 0.659
FA 仍然 > 15
```

再跑中间强度：

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
  --dea-lambda-single 0.015 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.5 \
  --dea-detach-evidence \
  --save-dea-debug \
  --save-pd-fa-best true \
  --pd-floor 0.92 \
  --iou-floor 0.655
```

这个只在 `0.01` FA 降得不够时跑。

---

## 15. 当前继续不要开的东西

继续不要开：

```text
--dea-lambda-dec > 0
--dea-lambda-empty > 0
inference-time d gate
component selector
all-16 subset
positive necessity loss
candidate verifier
```

原因：

```text
1. 当前 single-only 已经有正结果，不需要马上加复杂项。
2. lambda_dec 之前 d_prob 基本全 1，负样本监督还不稳定。
3. lambda_empty 会影响 self.final.bias，当前先不需要。
4. d gate 容易进一步压低弱目标，伤 PD。
5. component selector / all-16 会让变量变多，影响判断。
```

---

## 16. 当前结果怎么写进实验记录

可以这样记录：

```text
Sanity baseline:
IoU 0.6592 / PD 0.9354 / FA 18.9786

DEA-lite single-only, lambda_single=0.02, tau=0.5, detach evidence:
Best-IoU checkpoint, epoch 340:
IoU 0.6658 / PD 0.9218 / FA 9.4134

Final epoch 399:
IoU 0.6169 / PD 0.9116 / FA 29.2270

Interpretation:
The final epoch is not used because the run overfits / degenerates after the best checkpoint.
The best checkpoint shows the expected trade-off: FA is reduced by about 50%, IoU slightly improves, but PD drops by 1.36 points.
Next, lambda_single=0.01 is evaluated to recover PD while preserving FA reduction.
```

---

## 17. 最终执行顺序

```text
Step 1
保留当前 0.02 run 的 best checkpoint。

Step 2
修改 main.py：
    1. test loader 兼容 weight.pkl / checkpoint.pkl / official tar；
    2. 增加 PD-constrained FA-best checkpoint 保存。

Step 3
py_compile 检查。

Step 4
测试当前 0.02 best checkpoint，确认 weight.pkl 可直接 test。

Step 5
跑 lambda_single=0.01。

Step 6
比较：
    best-IoU checkpoint
    PD-constrained FA-best checkpoint

Step 7
如果 0.01 仍掉 PD，跑 0.005 + tau 0.6。
如果 0.01 保住 PD 但 FA 不够低，跑 0.015。
```

---

## 18. 当前判断

这次 `lambda_single=0.02` 不是失败。

更准确的判断是：

```text
DEA-lite single-scale anti-sufficiency 的方向成立：FA 明显下降。
但是 0.02 这一版偏强：PD 有下降，后期也有退化。
```

下一步最重要的不是继续加 DEA 模块，而是找到更好的 operating point：

```text
lambda_single=0.01
lambda_dec=0
lambda_empty=0
tau=0.5
detach_evidence=True
```

并用新增的 `weight_pd_fa_best.pkl` 保存“PD/Iou 约束下 FA 最低”的 checkpoint。

