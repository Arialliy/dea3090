# DEA-lite 400 epochs 后的修正版执行方案与代码修改

> 适配仓库：`https://github.com/Arialliy/DEA`  
> 本地路径：`/home/ly/DEA`  
> 当前阶段：DEA-lite debug / paired comparison  
> 当前目标：保留 `lambda_single=0.02` 的诊断正结果，补齐 checkpoint / seed / test loading 逻辑，然后跑 paired DEA-off baseline 与 `lambda_single=0.01`。

---

## 0. 当前结论

当前 `lambda_single=0.02` run 已跑完 400 epochs。

结果：

```text
final epoch 399:
IoU 0.6169 / PD 0.9116 / FA 29.2270

best epoch 340:
IoU 0.6658 / PD 0.9218 / FA 9.4134

epoch 346:
IoU 0.6572 / PD 0.9286 / FA 9.1097

epoch 358:
IoU 0.6585 / PD 0.9286 / FA 9.8689

clean sanity baseline best:
IoU 0.6592 / PD 0.9354 / FA 18.9786
```

判断：

```text
1. 这次 0.02 run 有正结果。
2. best-IoU epoch 340 相比 baseline：
   IoU: 0.6592 -> 0.6658
   FA:  18.9786 -> 9.4134
   PD:  0.9354 -> 0.9218
3. 说明 single-scale anti-sufficiency 确实能明显压 FA。
4. 但 PD 有轻微下降，而且 final epoch 明显退化。
5. 所以不能用 final epoch，只能用 best checkpoint / best weight。
6. 当前 0.02 run 适合作为诊断正结果，不适合作为最终 paired 主证据。
```

下一步目标：

```text
跑更保守的 lambda_single=0.01，目标是把 PD 拉回 0.93+，同时维持 FA 低于 baseline。
```

---

## 1. 重要修正：不要复制当前 checkpoint.pkl 当作 epoch 340 best

你已经确认当前代码缩进下：

```text
weight.pkl      = best-IoU 权重
checkpoint.pkl  = latest checkpoint，每轮覆盖
```

当前 `checkpoint.pkl` 内容是：

```text
epoch 399
iou 0.665827...
```

因此：

```text
checkpoint.pkl 里的 net 是 final/latest epoch 399 权重，不是 epoch 340 best 权重。
```

所以当前 `0.02` run 只能可靠保留：

```bash
cd /home/ly/DEA
export RUN_DIR=/home/ly/DEA/weight/MSHNet-2026-07-06-13-10-58

cp "$RUN_DIR/weight.pkl" \
   "$RUN_DIR/weight_lambda_single_0p02_best_iou_e340.pkl"
```

不要执行：

```bash
cp "$RUN_DIR/checkpoint.pkl" \
   "$RUN_DIR/checkpoint_lambda_single_0p02_best_iou_e340.pkl"
```

原因：这个 `checkpoint.pkl` 是 latest，不是 epoch 340 best。

---

## 2. 确认当前 checkpoint.pkl 内容

执行：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python - <<'PY'
import os
import torch

run_dir = "/home/ly/DEA/weight/MSHNet-2026-07-06-13-10-58"
ckpt_path = os.path.join(run_dir, "checkpoint.pkl")
weight_path = os.path.join(run_dir, "weight.pkl")
protected_path = os.path.join(run_dir, "weight_lambda_single_0p02_best_iou_e340.pkl")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("checkpoint path:", ckpt_path)
print("checkpoint epoch:", ckpt.get("epoch"))
print("checkpoint iou:", ckpt.get("iou"))
print("checkpoint best_iou:", ckpt.get("best_iou"))
print("checkpoint keys:", sorted(list(ckpt.keys())))
print("weight exists:", os.path.exists(weight_path))
print("protected best weight exists:", os.path.exists(protected_path))
PY
```

预期：

```text
checkpoint epoch: 399
protected best weight exists: True
```

---

## 3. 修改 main.py：补 seed 与 deterministic

### 3.1 在 import 区域加入

在 `main.py` 顶部 import 区域加入：

```python
import random
import numpy as np
```

如果已经有 `numpy` 或 `random`，不要重复加。

---

### 3.2 在 parse_args() 中加入参数

在 `parse_args()` 的 DEA 参数附近加入：

```python
parser.add_argument('--seed', type=int, default=20260706)
parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--pd-fa-min-pd', type=float, default=0.93)
parser.add_argument('--pd-fa-min-iou', type=float, default=0.655)
parser.add_argument('--paired-baseline-iou', type=float, default=0.0)
parser.add_argument('--pd-fa-iou-margin', type=float, default=0.005)
```

这些参数作用：

```text
--seed
    保证 paired DEA-off baseline 和 lambda_single=0.01 使用同一随机种子。

--deterministic true
    尽量提高可复现性。

--pd-fa-min-pd
    保存 PD/FA-best checkpoint 时要求 PD 至少达到该值。

--pd-fa-min-iou
    保存 PD/FA-best checkpoint 时要求 IoU 至少达到该值。

--paired-baseline-iou
    如果已知 paired baseline IoU，可以传入这个值。

--pd-fa-iou-margin
    使用 paired-baseline-iou 时，IoU 阈值可设为 paired baseline - margin。
```

---

### 3.3 在 main.py 中加入 seed 函数

把下面代码放在 `get_dea_ramp()` 附近即可：

```python
def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

---

### 3.4 修改 main entry

找到文件最后：

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

## 4. 修改 DataLoader：加入 generator 和 worker_init_fn

当前 `Trainer.__init__` 里会创建：

```python
self.train_loader = Data.DataLoader(...)
self.val_loader = Data.DataLoader(...)
```

在创建 DataLoader 前加入：

```python
data_generator = torch.Generator()
data_generator.manual_seed(args.seed)
```

然后把 `loader_kwargs` 改成：

```python
loader_kwargs = {
    "num_workers": args.num_workers,
    "pin_memory": args.pin_memory,
    "persistent_workers": args.num_workers > 0,
    "worker_init_fn": seed_worker,
    "generator": data_generator,
}

if args.num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
```

最终 DataLoader 保持原结构：

```python
self.train_loader = Data.DataLoader(
    trainset,
    args.batch_size,
    shuffle=True,
    drop_last=True,
    **loader_kwargs,
)

self.val_loader = Data.DataLoader(
    valset,
    1,
    drop_last=False,
    **loader_kwargs,
)
```

如果你的 PyTorch 版本不允许 `generator` 与 `persistent_workers=False` 同时出现，可以保留 `generator`，删除 `persistent_workers` 这一项；但优先按上面写。

---

## 5. 修改 cudnn benchmark 设置

当前 `Trainer.__init__` 里如果有：

```python
torch.backends.cudnn.benchmark = False
```

改成：

```python
torch.backends.cudnn.benchmark = not args.deterministic
```

这样命令行传：

```bash
--deterministic true
```

时，会关闭 benchmark。

---

## 6. 修改 test loader：兼容 weight.pkl / checkpoint.pkl / official tar

当前 test 模式可能有：

```python
weight = load_torch_file(args.weight_path)
self.load_model_state(weight['state_dict'])
```

这会导致 raw `weight.pkl` 加载失败，因为当前 `weight.pkl` 是 `self.model.state_dict()`，没有 `state_dict` key。

### 6.1 在 Trainer 类中加入 extract_state_dict()

把下面函数放到 `load_model_state()` 前面或后面都可以：

```python
def extract_state_dict(self, weight_obj):
    if isinstance(weight_obj, dict):
        if 'state_dict' in weight_obj:
            return weight_obj['state_dict']
        if 'net' in weight_obj:
            return weight_obj['net']

        looks_like_state_dict = all(
            torch.is_tensor(v) for v in weight_obj.values()
        )
        if looks_like_state_dict:
            return weight_obj

    raise RuntimeError(
        'Unsupported weight format. Expected raw state_dict, '
        'dict with state_dict, or dict with net.'
    )
```

### 6.2 修改 test 模式加载

把：

```python
weight = load_torch_file(args.weight_path)
self.load_model_state(weight['state_dict'])
```

改成：

```python
weight = load_torch_file(args.weight_path)
state_dict = self.extract_state_dict(weight)
self.load_model_state(state_dict)
```

这样可以兼容：

```text
1. raw model.state_dict() 格式的 weight.pkl
2. {'net': ..., 'optimizer': ..., ...} 格式的 checkpoint.pkl
3. {'state_dict': ...} 格式的官方 tar / weight 文件
```

---

## 7. 修改 resume 逻辑：checkpoint.pkl 以后是 latest

当前恢复时不要再用：

```python
self.best_iou = checkpoint['iou']
```

因为以后 `checkpoint.pkl` 是 latest，`checkpoint['iou']` 可能是 latest epoch 的 IoU，不一定是 best IoU。

### 7.1 初始化 best 指标

在 `Trainer.__init__` 里初始化位置加入：

```python
self.best_iou = 0.0
self.best_pd_fa = float('inf')
self.best_pd_fa_iou = 0.0
self.best_pd_fa_pd = 0.0
self.best_pd_fa_epoch = -1
```

如果原来已有：

```python
self.best_iou = 0
```

替换成上面这组。

---

### 7.2 修改 if_checkpoint 恢复逻辑

把恢复中的：

```python
self.start_epoch = checkpoint['epoch']+1
self.best_iou = checkpoint['iou']
```

改成：

```python
self.start_epoch = checkpoint.get('epoch', -1) + 1
self.best_iou = float(checkpoint.get('best_iou', checkpoint.get('iou', 0.0)))

self.best_pd_fa = float(checkpoint.get('best_pd_fa', float('inf')))
self.best_pd_fa_iou = float(checkpoint.get('best_pd_fa_iou', 0.0))
self.best_pd_fa_pd = float(checkpoint.get('best_pd_fa_pd', 0.0))
self.best_pd_fa_epoch = int(checkpoint.get('best_pd_fa_epoch', -1))
```

这样兼容旧 checkpoint，也适配新 checkpoint。

---

## 8. 修改 checkpoint 保存逻辑

当前 `test()` 训练模式中，应把保存逻辑改成三类：

```text
checkpoint.pkl                 latest / resume checkpoint，每个 epoch 覆盖
weight.pkl                     best-IoU raw state_dict
checkpoint_best_iou.pkl        best-IoU checkpoint
weight_pd_fa_best.pkl          PD/FA constrained best raw state_dict
checkpoint_pd_fa_best.pkl      PD/FA constrained best checkpoint
```

### 8.1 在 test() 中 current metrics 后加入阈值

在 `test()` 中已有：

```python
current_pd = PD[0]
current_fa = FA[0] * 1000000
```

后面加入：

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

含义：

```text
PD/FA-best 不只看低 FA；还要满足：
PD >= 0.93
IoU >= 0.655，或 IoU >= paired baseline - margin
```

这样避免选到 IoU 很差但 FA 很低的点。

---

### 8.2 替换原来的 best-IoU checkpoint 保存块

找到当前类似代码：

```python
if mean_IoU > self.best_iou:
    self.best_iou = mean_IoU
    torch.save(self.model.state_dict(), osp.join(self.save_folder, 'weight.pkl'))
    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
        f.write(...)
    all_states = {"net":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), "epoch": epoch, "iou":self.best_iou}
    torch.save(all_states, osp.join(self.save_folder, 'checkpoint.pkl'))
```

把它替换为下面完整块：

```python
# -----------------------------
# Best-IoU checkpoint
# -----------------------------
if mean_IoU > self.best_iou:
    self.best_iou = mean_IoU

    torch.save(
        self.model.state_dict(),
        osp.join(self.save_folder, 'weight.pkl')
    )

    best_iou_states = {
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

# -----------------------------
# PD-constrained FA-best checkpoint
# -----------------------------
if is_pd_fa_candidate:
    self.best_pd_fa = current_fa
    self.best_pd_fa_iou = mean_IoU
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
        "iou": mean_IoU,
        "pd": current_pd,
        "fa": current_fa,
        "best_iou": self.best_iou,
        "best_pd_fa": self.best_pd_fa,
        "best_pd_fa_iou": self.best_pd_fa_iou,
        "best_pd_fa_pd": self.best_pd_fa_pd,
        "best_pd_fa_epoch": self.best_pd_fa_epoch,
    }

    torch.save(
        pd_fa_states,
        osp.join(self.save_folder, 'checkpoint_pd_fa_best.pkl')
    )

    with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
        f.write(
            '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
                time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                epoch,
                mean_IoU,
                current_pd,
                current_fa,
            )
        )

# -----------------------------
# Latest checkpoint for resume
# -----------------------------
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
}

torch.save(
    latest_states,
    osp.join(self.save_folder, 'checkpoint.pkl')
)
```

注意：

```text
checkpoint.pkl 从现在开始明确是 latest。
resume 用 checkpoint.pkl。
best-IoU 权重用 weight.pkl。
PD/FA-best 权重用 weight_pd_fa_best.pkl。
```

---

## 9. 编译检查

改完后执行：

```bash
cd /home/ly/DEA

/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile \
  main.py \
  model/MSHNet.py \
  model/loss.py
```

如果这里报错，先不要跑训练。

---

## 10. 测试当前 0.02 best-IoU 权重

执行：

```bash
cd /home/ly/DEA
export RUN_DIR=/home/ly/DEA/weight/MSHNet-2026-07-06-13-10-58

CUDA_VISIBLE_DEVICES=0 /home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/DEA/datasets/IRSTD-1K \
  --batch-size 4 \
  --num-workers 4 \
  --pin-memory false \
  --mode test \
  --weight-path "$RUN_DIR/weight_lambda_single_0p02_best_iou_e340.pkl"
```

预期接近：

```text
IoU ≈ 0.6658
PD  ≈ 0.9218
FA  ≈ 9.4134
```

如果不接近，优先检查：

```text
1. test loader 是否正确读取 raw state_dict；
2. weight 文件是否确实来自当前 run 的 weight.pkl；
3. dataset-dir 是否一致；
4. val/test split 是否一致；
5. base_size 是否一致。
```

---

## 11. 跑 paired DEA-off baseline

改完 seed 后，先从零跑同 seed 的 DEA-off baseline：

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
  --dea-lambda-empty 0 \
  --pd-fa-min-pd 0.93 \
  --pd-fa-min-iou 0.655
```

这个 baseline 的作用：

```text
给 lambda_single=0.01 提供 paired seed 对照。
```

不要用当前已有的非 paired baseline 替代它。

---

## 12. 跑 lambda_single=0.01

如果 paired DEA-off baseline 正常，再跑：

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
  --pd-fa-min-pd 0.93 \
  --pd-fa-min-iou 0.655 \
  --paired-baseline-iou <paired_baseline_best_iou>
```

这里的 `<paired_baseline_best_iou>` 必须替换成第 11 节 paired DEA-off baseline 的 best IoU。
例如 paired baseline best IoU 是 `0.6592` 时，实际约束为 `max(0.655, 0.6592 - 0.005)`。

目标：

```text
相比 paired DEA-off baseline：
1. IoU 持平或略升；
2. PD 拉回 0.93+，或至少接近 paired baseline；
3. FA 仍明显低于 baseline；
4. weight_pd_fa_best.pkl 能保存一个更合理的 operating point。
```

---

## 13. 如果 0.01 仍然掉 PD

如果 `lambda_single=0.01` 仍然出现：

```text
PD 低于 0.92
或者 IoU 明显低于 paired baseline
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
  --seed 20260706 \
  --deterministic true \
  --dea-lambda-single 0.005 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.6 \
  --dea-detach-evidence \
  --save-dea-debug \
  --pd-fa-min-pd 0.93 \
  --pd-fa-min-iou 0.655
```

---

## 14. 当前不要做

继续不要做：

```text
1. lambda_dec > 0
2. lambda_empty > 0
3. inference-time d gate
4. component selector
5. all-16 subset
6. positive necessity
7. candidate verifier
8. 把当前 0.02 当最终 paired 主证据
```

原因：

```text
1. 0.02 已经显示 single loss 会压低 PD；
2. lambda_dec 当前没有可靠负样本，d_prob 容易全 1；
3. lambda_empty 可能压 final bias，进一步伤 PD；
4. d gate 会直接压预测，风险更大；
5. component selector / all-16 会增加变量，不利于当前 debug。
```

---

## 15. 最终执行顺序

按这个顺序执行：

```text
Step 1
只复制当前 0.02 run 的 weight.pkl：
weight_lambda_single_0p02_best_iou_e340.pkl
不要复制 checkpoint.pkl。

Step 2
确认 checkpoint.pkl 是 latest/final，不是 best。

Step 3
修改 main.py：
- seed_everything
- seed_worker
- DataLoader generator
- test loader 兼容 raw state_dict / state_dict / net
- resume 使用 checkpoint.get('best_iou', checkpoint.get('iou', 0))
- latest checkpoint 每 epoch 保存
- best-IoU checkpoint 单独保存
- PD/FA-best checkpoint 加 PD 和 IoU 双约束

Step 4
py_compile。

Step 5
测试当前 0.02 best weight。

Step 6
从零跑 paired DEA-off baseline。

Step 7
从零跑 lambda_single=0.01。

Step 8
如果 0.01 仍掉 PD，再跑 0.005 + tau=0.6。
```

---

## 16. 当前 0.02 run 的论文/实验定位

当前 `lambda_single=0.02` 的定位：

```text
诊断正结果，而非最终主结果。
```

可以这样解释：

```text
single-scale anti-sufficiency 能显著降低 false alarms，说明 MSHNet 的一部分 hard-clutter false alarms 确实具有 single-scale sufficient evidence 特征；但 0.02 的约束强度仍会牺牲一部分 PD，因此需要进一步调节到更保守的 operating point。
```

这支持 DEA-lite 的基本 hypothesis，但最终主证据应该来自补 seed 后的 paired comparison。
