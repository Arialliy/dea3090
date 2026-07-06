# DEA-lite 0.02 跑完后的修正版下一步方案：checkpoint 保护、seed 补丁、PD/FA-best 保存

> 适配仓库：`/home/ly/DEA`，对应 `Arialliy/DEA`。  
> 当前阶段：DEA-lite debug / single-scale anti-sufficiency 调参。  
> 当前不要扩展到 full DEA、component selector、all-16 subset、inference-time d gate。

---

## 0. 当前 0.02 run 的结论

当前 run 已经 400 epochs 完成。

关键结果：

```text
final epoch 399:
    IoU 0.6169
    PD  0.9116
    FA  29.2270

best epoch 340:
    IoU 0.6658
    PD  0.9218
    FA  9.4134

epoch 346:
    IoU 0.6572
    PD  0.9286
    FA  9.1097

epoch 358:
    IoU 0.6585
    PD  0.9286
    FA  9.8689

clean sanity baseline best:
    IoU 0.6592
    PD  0.9354
    FA  18.9786
```

结论：

```text
lambda_single=0.02 有正信号：
    IoU: 0.6592 -> 0.6658
    FA : 18.9786 -> 9.4134
    PD : 0.9354 -> 0.9218

但后期退化明显：
    final epoch 399 不能用。
```

所以当前 run 的定位是：

```text
0.02 证明 single-scale anti-sufficiency 能显著降低 FA；
但它仍然略伤 PD，且后期存在 over-training / degradation。
```

下一步不是继续增强 DEA，而是：

```text
1. 正确保存当前 0.02 的 best-IoU 权重；
2. 修复 test loader 对 weight.pkl / checkpoint.pkl 的兼容；
3. 补回 seed；
4. 加 PD/FA-best 保存逻辑，用于后续 run；
5. 从零跑 lambda_single=0.01。
```

---

## 1. 重要修正：不要复制当前 checkpoint.pkl 当 epoch 340 best

你本地已经确认当前代码缩进和文件内容：

```text
weight.pkl:
    best-IoU 权重，对应 epoch 340。

checkpoint.pkl:
    每轮覆盖，当前里面是 epoch 399。
    checkpoint.pkl['iou'] 记录的是 best_iou = 0.665827...
    但 checkpoint.pkl['net'] 是 final epoch 399 的网络权重。
```

因此当前 0.02 run 只能可靠保留：

```bash
RUN_DIR=/home/ly/DEA/weight/<你的当前0.02-run目录>

cp "$RUN_DIR/weight.pkl" \
   "$RUN_DIR/weight_lambda_single_0p02_best_iou_e340.pkl"
```

不要执行：

```bash
cp "$RUN_DIR/checkpoint.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p02_best_iou_e340.pkl"
```

原因：

```text
当前 checkpoint.pkl 的 net 不是 epoch 340 权重。
它只能作为 latest/final resume checkpoint，不能当 best-IoU 权重使用。
```

建议顺手确认：

```bash
cd /home/ly/DEA

python - <<'PY'
import torch, os
run_dir = os.environ.get('RUN_DIR')
if not run_dir:
    raise SystemExit('Please export RUN_DIR=/home/ly/DEA/weight/<run_dir> first')

ckpt = torch.load(os.path.join(run_dir, 'checkpoint.pkl'), map_location='cpu')
print('checkpoint epoch:', ckpt.get('epoch'))
print('checkpoint iou:', ckpt.get('iou'))
print('checkpoint keys:', ckpt.keys())
print('weight exists:', os.path.exists(os.path.join(run_dir, 'weight.pkl')))
PY
```

---

## 2. 当前 0.02 run 只能追回 best-IoU，不能追回 epoch 346

PD/FA-best checkpoint 逻辑值得加，但只能用于后续 run。

当前 run 没有保存 epoch 346 的权重，因此不能追回：

```text
epoch 346:
    IoU 0.6572
    PD  0.9286
    FA  9.1097
```

除非你当时保存了每轮 checkpoint，否则只能保留当前的：

```text
weight.pkl = best-IoU epoch 340 权重
```

所以当前 0.02 的可靠保留文件只有：

```text
weight_lambda_single_0p02_best_iou_e340.pkl
```

后续 0.01 run 才加入：

```text
weight_pd_fa_best.pkl
checkpoint_pd_fa_best.pkl
```

---

## 3. 执行顺序修正版

现在建议执行顺序改成：

```text
Step 1. 保留当前 0.02 的 weight.pkl，不复制 checkpoint.pkl。
Step 2. 修改 main.py：test loader 兼容 weight.pkl / checkpoint.pkl / official tar。
Step 3. 修改 main.py：新增 PD/FA-best checkpoint 保存。
Step 4. 补 seed 和 dataloader worker seed。
Step 5. py_compile。
Step 6. 测试当前 0.02 的 best-IoU weight.pkl。
Step 7. 从零跑 lambda_single=0.01。
```

---

## 4. 修改 main.py：补 seed

### 4.1 增加 import

在 `main.py` 顶部 import 区域加入：

```python
import random
import numpy as np
```

保留原来的：

```python
import torch
import torch.utils.data as Data
```

---

### 4.2 增加 seed 函数

在 `load_torch_file` 后面或 `parse_args` 前面加入：

```python
def seed_everything(seed, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

---

### 4.3 parse_args 增加 seed 参数

在 `parse_args()` 里加入：

```python
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=True)
```

建议位置放在训练超参数附近，例如：

```python
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--warm-epoch', type=int, default=5)
```

---

### 4.4 main 入口调用 seed_everything

把文件底部：

```python
if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
```

改成：

```python
if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed, args.deterministic)
    trainer = Trainer(args)
```

---

### 4.5 DataLoader 加 generator 和 worker_init_fn

在 `Trainer.__init__` 里，创建 DataLoader 之前加入：

```python
self.data_generator = torch.Generator()
self.data_generator.manual_seed(args.seed)
```

然后把 train loader 改成：

```python
self.train_loader = Data.DataLoader(
    trainset,
    args.batch_size,
    shuffle=True,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=self.data_generator,
    **loader_kwargs,
)
```

把 val loader 改成：

```python
self.val_loader = Data.DataLoader(
    valset,
    1,
    shuffle=False,
    drop_last=False,
    worker_init_fn=seed_worker,
    generator=self.data_generator,
    **loader_kwargs,
)
```

注意：

```text
val loader 显式 shuffle=False。
```

---

## 5. 修改 main.py：test loader 兼容 weight.pkl / checkpoint.pkl / official tar

当前 test 模式如果只支持：

```python
weight['state_dict']
```

那会导致直接测试 `weight.pkl` 失败，因为 `weight.pkl` 通常就是一个 raw `state_dict`。

### 5.1 增加 extract_model_state

在 `load_torch_file` 后面加入：

```python
def extract_model_state(obj):
    if isinstance(obj, dict):
        if 'state_dict' in obj:
            return obj['state_dict']
        if 'net' in obj:
            return obj['net']
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj

    raise KeyError(
        'Unsupported weight format. Expected one of: '
        'raw state_dict, dict with state_dict, or dict with net.'
    )
```

---

### 5.2 替换 test 模式加载逻辑

把：

```python
if args.mode=='test':
    weight = load_torch_file(args.weight_path)
    self.load_model_state(weight['state_dict'])
```

改成：

```python
if args.mode == 'test':
    weight = load_torch_file(args.weight_path)
    state_dict = extract_model_state(weight)
    self.load_model_state(state_dict)
```

这样下面三种都能测：

```text
1. weight.pkl                      raw model.state_dict()
2. checkpoint.pkl                  {'net': state_dict, ...}
3. official IRSTD-1k_weight.tar    {'state_dict': state_dict, ...}
```

---

## 6. 修改 main.py：新增 PD/FA-best checkpoint 保存逻辑

这个逻辑只对后续 run 生效，不能追回当前 0.02 的 epoch 346。

### 6.1 parse_args 增加参数

在 DEA 参数附近加入：

```python
parser.add_argument('--save-pd-fa-best', action='store_true')
parser.add_argument('--pd-constraint', type=float, default=0.93)
parser.add_argument('--fa-tiebreak-eps', type=float, default=1e-6)
```

---

### 6.2 Trainer.__init__ 增加状态变量

在：

```python
self.best_iou = 0
```

后面加入：

```python
self.best_pd_fa = float('inf')
self.best_pd_fa_iou = 0.0
self.best_pd_fa_epoch = -1
```

---

### 6.3 保存 latest checkpoint，避免 checkpoint.pkl 语义混乱

建议把 `checkpoint.pkl` 明确作为 latest checkpoint，用于 resume。

在 `test()` 里每个 train epoch 结束后，计算完：

```python
current_pd = PD[0]
current_fa = FA[0] * 1000000
```

后面加入 latest 保存：

```python
latest_states = {
    "net": self.model.state_dict(),
    "optimizer": self.optimizer.state_dict(),
    "epoch": epoch,
    "iou": mean_IoU,
    "best_iou": self.best_iou,
    "pd": current_pd,
    "fa": current_fa,
}

torch.save(latest_states, osp.join(self.save_folder, 'checkpoint.pkl'))
```

这样以后明确：

```text
checkpoint.pkl = latest checkpoint，用于 resume，不代表 best。
```

---

### 6.4 best-IoU 保存逻辑改成同时保存 checkpoint_best_iou.pkl

保留原始：

```python
if mean_IoU > self.best_iou:
    self.best_iou = mean_IoU
    torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
```

然后在这个 if 里面加入：

```python
best_iou_states = {
    "net": self.model.state_dict(),
    "optimizer": self.optimizer.state_dict(),
    "epoch": epoch,
    "iou": mean_IoU,
    "pd": current_pd,
    "fa": current_fa,
}

torch.save(best_iou_states, osp.join(self.save_folder, 'checkpoint_best_iou.pkl'))
```

完整块建议改成：

```python
if mean_IoU > self.best_iou:
    self.best_iou = mean_IoU

    torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))

    best_iou_states = {
        "net": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "epoch": epoch,
        "iou": mean_IoU,
        "pd": current_pd,
        "fa": current_fa,
    }
    torch.save(best_iou_states, osp.join(self.save_folder, 'checkpoint_best_iou.pkl'))

    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
            time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
            epoch,
            self.best_iou,
            current_pd,
            current_fa,
        ))
```

---

### 6.5 新增 PD-constrained FA-best 保存

在 best-IoU 保存块后面加入：

```python
if self.args.save_pd_fa_best and current_pd >= self.args.pd_constraint:
    better_fa = current_fa < self.best_pd_fa - self.args.fa_tiebreak_eps
    tie_better_iou = (
        abs(current_fa - self.best_pd_fa) <= self.args.fa_tiebreak_eps
        and mean_IoU > self.best_pd_fa_iou
    )

    if better_fa or tie_better_iou:
        self.best_pd_fa = current_fa
        self.best_pd_fa_iou = mean_IoU
        self.best_pd_fa_epoch = epoch

        torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight_pd_fa_best.pkl'))

        pd_fa_states = {
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "iou": mean_IoU,
            "pd": current_pd,
            "fa": current_fa,
            "pd_constraint": self.args.pd_constraint,
        }
        torch.save(pd_fa_states, osp.join(self.save_folder, 'checkpoint_pd_fa_best.pkl'))

        with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
            f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\t - constraint_PD {:.4f}\n'.format(
                time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                epoch,
                mean_IoU,
                current_pd,
                current_fa,
                self.args.pd_constraint,
            ))
```

解释：

```text
weight.pkl:
    best IoU 权重。

checkpoint.pkl:
    latest checkpoint，用于 resume。

weight_pd_fa_best.pkl:
    在 PD >= constraint 的前提下，FA 最低的权重。

checkpoint_pd_fa_best.pkl:
    对应的 resume/debug checkpoint。
```

---

## 7. py_compile 检查

修改完后运行：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile \
  main.py \
  model/MSHNet.py \
  model/loss.py
```

如果通过，再测试当前 0.02 best-IoU weight。

---

## 8. 测试当前 0.02 best-IoU weight

先设置当前 run 目录：

```bash
export RUN_DIR=/home/ly/DEA/weight/<你的当前0.02-run目录>
```

复制并命名 best-IoU 权重：

```bash
cp "$RUN_DIR/weight.pkl" \
   "$RUN_DIR/weight_lambda_single_0p02_best_iou_e340.pkl"
```

测试：

```bash
cd /home/ly/DEA

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/weight_lambda_single_0p02_best_iou_e340.pkl"
```

期望接近：

```text
IoU ≈ 0.6658
PD  ≈ 0.9218
FA  ≈ 9.4134
```

如果不接近，优先检查：

```text
1. weight-path 是否指向 weight.pkl 的复制文件；
2. test loader 是否正确支持 raw state_dict；
3. dataset-dir / split 是否和 train 时一致；
4. mode test 是否没有启用 return_dea / d gate。
```

---

## 9. 从零跑 paired seed 的 DEA-off baseline

如果之前 sanity baseline 没有固定 seed，那么建议补一个 paired baseline。

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
  --dea-lambda-single 0 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0
```

这个 run 作为后续 `lambda_single=0.01` 的 paired baseline。

---

## 10. 从零跑 lambda_single=0.01

如果当前 baseline 已经固定 seed，直接跑 0.01：

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
  --save-pd-fa-best \
  --pd-constraint 0.93
```

目标：

```text
相比 paired DEA-off baseline：
    PD 尽量回到 0.93+
    FA 明显低于 baseline
    IoU 不低于或接近 baseline
```

如果 0.01 的 best-IoU 仍然：

```text
PD < 0.92
```

或者比 paired baseline 明显低：

```text
PD drop > 0.015 ~ 0.02
```

则继续降低强度。

---

## 11. 如果 0.01 仍然伤 PD，再跑 0.005 + tau 0.6

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
  --save-pd-fa-best \
  --pd-constraint 0.93
```

---

## 12. 当前仍然不要做的事

继续不要做：

```text
1. --dea-lambda-dec > 0
2. --dea-lambda-empty > 0
3. inference-time d gate
4. component selector
5. all-16 subset
6. positive necessity loss
7. candidate verifier
8. 复制当前 checkpoint.pkl 作为 epoch 340 best
```

原因：

```text
0.02 已经证明 single loss 有效果，但也会伤 PD。
现在要做的是找稳定 operating point，不是继续增加约束。
```

---

## 13. 最终修正结论

你的修正是对的。

这次计划需要改成：

```text
1. 当前 0.02 run：只保留 weight.pkl 作为 epoch 340 best-IoU 权重。
2. 不复制当前 checkpoint.pkl 当 best，因为 checkpoint.pkl 的 net 是 final epoch 399。
3. PD/FA-best checkpoint 只能用于后续 run，不能追回当前 epoch 346。
4. 在跑 0.01 前，必须补 seed 和 dataloader worker seed。
5. test loader 要兼容 weight.pkl / checkpoint.pkl / official tar。
6. 后续 run 同时保存：
       checkpoint.pkl              latest，用于 resume
       weight.pkl                  best IoU
       checkpoint_best_iou.pkl     best IoU checkpoint
       weight_pd_fa_best.pkl       PD constrained FA-best
       checkpoint_pd_fa_best.pkl   PD constrained FA-best checkpoint
7. 从零跑 paired seed 的 lambda_single=0.01。
```

一句话：

> **当前 0.02 run 已经有有效诊断结果，但只能相信 `weight.pkl` 的 best-IoU 权重；后续要补 seed、补 test 加载兼容、补 PD/FA-best 保存，再用 paired seed 跑更保守的 `lambda_single=0.01`。**
