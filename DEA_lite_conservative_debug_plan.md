# DEA-lite on MSHNet：低 IoU / 低 Pd 后的保守排查与代码修改方案

> 适用项目：`Arialliy/DEA` 当前 DEA-lite / MSHNet 修改版。  
> 目标：先恢复原始 MSHNet 主路径，再用最小、保守的 single-scale anti-sufficiency loss 验证 DEA-lite 是否有效。  
> 当前问题：loss 下降太慢，IoU 约 0.6，Pd 很低。  
> 核心原则：不要继续堆 DEA-full 机制；先做纯净 sanity baseline，再只开最小 DEA 项。

---

## 0. 当前结论

你现在的判断是对的。接下来应该做这 5 件事：

```text
1. 加纯净开关：DEA loss 全关时，必须走原始 MSHNet forward。
2. DEA 默认权重全部设成 0，默认训练不启用 DEA。
3. 暂时关掉 decidability_loss，因为 d_prob 基本全 1。
4. single-scale loss 改成 conservative 版本，只用 z_only_max 选择 hard background。
5. 先启用 detach evidence，让 DEA auxiliary loss 不直接冲击 encoder / decoder。
```

当前不要做：

```text
lambda_dec > 0
inference-time d gate
component selector
all-16 subset
positive necessity loss
candidate verifier
继续用当前低 IoU run 当正式结果
```

---

## 1. 先停掉当前 run

如果在前台终端：

```bash
Ctrl-C
```

如果是后台进程，当前 PID 是 `743370`：

```bash
kill -INT 743370
```

不要优先使用：

```bash
kill -9 743370
```

只有 `kill -INT` 没反应时，再用 `kill -9`。

---

## 2. 为什么必须先跑纯净 sanity baseline

现在低 IoU / 低 Pd 可能来自两类原因：

```text
A. DEA-lite auxiliary loss 太强，把模型压坏了；
B. 代码修改后，原始 MSHNet 路径本身已经不等价了。
```

所以第一步不是继续调 DEA，而是确认：

```text
在当前已修改代码里，当 DEA loss 全关时，模型是否仍然等价于原始 MSHNet。
```

因此，当：

```bash
--dea-lambda-single 0 \
--dea-lambda-dec 0 \
--dea-lambda-empty 0
```

训练必须走：

```python
masks, pred = self.model(data, tag)
```

不要走：

```python
masks, pred, dea_out = self.model(data, tag, return_dea=True)
```

即使 DEA loss 权重为 0，也不要在 sanity baseline 中 `return_dea=True`。这个 baseline 必须干净。

---

## 3. `main.py`：参数默认值改成 0

在 `argparse` 中确认或新增下面参数。重点是默认 DEA 权重全部为 0。

```python
parser.add_argument('--dea-lambda-single', type=float, default=0.0,
                    help='Weight for DEA single-scale anti-sufficiency loss.')
parser.add_argument('--dea-lambda-dec', type=float, default=0.0,
                    help='Weight for DEA decidability loss.')
parser.add_argument('--dea-lambda-empty', type=float, default=0.0,
                    help='Weight for DEA empty-evidence loss.')
parser.add_argument('--dea-ramp-epochs', type=int, default=0,
                    help='Ramp epochs for DEA losses after MSHNet warm-up.')
parser.add_argument('--dea-tau', type=float, default=0.5,
                    help='Threshold for conservative hard background mining.')
parser.add_argument('--dea-debug-interval', type=int, default=50,
                    help='Print DEA debug info every N iterations.')
parser.add_argument('--save-dea-debug', action='store_true',
                    help='Enable DEA debug logging.')
parser.add_argument('--dea-detach-evidence', action='store_true',
                    help='Detach scale logits in counterfactual DEA paths.')
```

不要把默认值设成：

```python
lambda_single = 0.10
lambda_dec = 0.05
lambda_empty = 0.01
```

这些对当前 debug 阶段太强。

---

## 4. `main.py`：加入纯净 DEA 开关

在训练循环里，先根据当前 epoch 和 DEA loss 权重决定是否启用 DEA。

推荐写法：

```python
tag = epoch > self.warm_epoch

use_dea = (
    tag
    and (
        args.dea_lambda_single > 0
        or args.dea_lambda_dec > 0
        or args.dea_lambda_empty > 0
    )
)

if use_dea:
    masks, pred, dea_out = self.model(
        data,
        tag,
        return_dea=True,
        dea_detach_evidence=args.dea_detach_evidence,
    )
else:
    masks, pred = self.model(data, tag)
    dea_out = None
```

这个开关的作用：

```text
DEA 全关时：完全原始 MSHNet 路径。
DEA 开启时：只在 warm-up 后进入 DEA-lite forward。
```

---

## 5. `model/MSHNet.py`：forward 签名修改

把原来的 forward 签名：

```python
def forward(self, x, warm_flag, return_dea=False):
```

改成：

```python
def forward(self, x, warm_flag, return_dea=False, dea_detach_evidence=False):
```

如果你现在还是原始版本：

```python
def forward(self, x, warm_flag):
```

则直接改成完整版本：

```python
def forward(self, x, warm_flag, return_dea=False, dea_detach_evidence=False):
```

`return_dea=False` 保证测试阶段和 sanity baseline 不受影响。

---

## 6. `model/MSHNet.py`：保留原始 warm_flag=False 分支

`warm_flag=False` 阶段不要启用 DEA-lite。

保持：

```python
else:
    output = self.output_0(x_d0)
    return [], output
```

原因：

```text
warm_flag=False 是原始 MSHNet 的 warm-up / single-output 阶段；
这个阶段没有四尺度 fusion，不适合做 z_only_i / z_empty。
```

---

## 7. `model/MSHNet.py`：构造 scale_logits 和 z_full

在 `warm_flag=True` 分支中，保留原始四个输出头：

```python
mask0 = self.output_0(x_d0)
mask1 = self.output_1(x_d1)
mask2 = self.output_2(x_d2)
mask3 = self.output_3(x_d3)
```

然后显式保存上采样后的四个尺度 logit：

```python
s0 = mask0
s1 = self.up(mask1)
s2 = self.up_4(mask2)
s3 = self.up_8(mask3)

scale_logits = torch.cat([s0, s1, s2, s3], dim=1)
z_full = self.final(scale_logits)
```

然后：

```python
if return_dea:
    dea_out = self.build_dea_lite_outputs(
        scale_logits,
        z_full,
        detach_evidence=dea_detach_evidence,
    )
    return [mask0, mask1, mask2, mask3], z_full, dea_out

return [mask0, mask1, mask2, mask3], z_full
```

---

## 8. `model/MSHNet.py`：替换 `build_dea_lite_outputs`

用下面这个版本替换当前 `build_dea_lite_outputs`。

关键变化：

```text
1. 支持 detach_evidence；
2. z_only_i 仍然经过同一个 self.final；
3. detach_evidence=True 时，DEA counterfactual loss 不直接回传到 encoder / decoder / scale heads；
4. decidability head 可以 forward 做 debug，但当前 lambda_dec=0，不参与优化。
```

```python
def build_dea_lite_outputs(self, scale_logits, z_full, detach_evidence=False):
    """
    Build DEA-lite counterfactual outputs.

    Args:
        scale_logits: Tensor, [B, 4, H, W]
        z_full:       Tensor, [B, 1, H, W]
        detach_evidence: bool. If True, counterfactual paths do not
                         back-propagate to encoder/decoder/scale heads.

    Returns:
        dict with scale_logits, z_empty, z_only, z_only_max,
        z_only_var, decidability_logit.
    """
    if detach_evidence:
        cf_scale_logits = scale_logits.detach()
        dec_scale_logits = scale_logits.detach()
        dec_z_full = z_full.detach()
    else:
        cf_scale_logits = scale_logits
        dec_scale_logits = scale_logits
        dec_z_full = z_full

    neutral = torch.zeros_like(cf_scale_logits)

    # Empty evidence path: final([0, 0, 0, 0])
    z_empty = self.final(neutral)

    # Single-scale evidence paths
    z_only_list = []
    for i in range(4):
        e_only = neutral.clone()
        e_only[:, i:i + 1] = cf_scale_logits[:, i:i + 1]
        z_only_i = self.final(e_only)
        z_only_list.append(z_only_i)

    # [B, 4, H, W]
    z_only = torch.cat(z_only_list, dim=1)

    z_only_max = z_only.max(dim=1, keepdim=True)[0]
    z_only_var = z_only.var(dim=1, keepdim=True, unbiased=False)

    # Decidability input: 1 + 1 + 1 + 4 = 7 channels
    d_input = torch.cat([
        dec_z_full,
        z_only_max.detach() if detach_evidence else z_only_max,
        z_only_var.detach() if detach_evidence else z_only_var,
        dec_scale_logits,
    ], dim=1)

    d_logit = self.decidability_head(d_input)

    return {
        'scale_logits': scale_logits,
        'z_empty': z_empty,
        'z_only': z_only,
        'z_only_max': z_only_max,
        'z_only_var': z_only_var,
        'decidability_logit': d_logit,
    }
```

---

## 9. `model/MSHNet.py`：确认 decidability head 输入通道是 7

`__init__` 中应有：

```python
self.decidability_head = nn.Sequential(
    nn.Conv2d(7, 8, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=1),
)
```

为什么是 7：

```text
z_full      : 1 channel
z_only_max  : 1 channel
z_only_var  : 1 channel
scale_logits: 4 channels
-----------------------------
total       : 7 channels
```

不是 `Conv2d(6, 8, ...)`。

---

## 10. `model/loss.py`：safe background mask

确认有这个函数：

```python
import torch
import torch.nn.functional as F


def build_safe_bg(gt, kernel_size=15):
    """
    Args:
        gt: Tensor, [B, 1, H, W], binary mask
    Returns:
        safe_bg: Tensor, [B, 1, H, W]
    """
    pad = kernel_size // 2
    gt_dilate = F.max_pool2d(
        gt.float(),
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    safe_bg = (gt_dilate < 0.5).float()
    return safe_bg
```

---

## 11. `model/loss.py`：保守版 single-scale loss

用下面版本替换之前的 single-scale loss。

关键变化：

```text
hard background 只来自 sigmoid(z_only_max) > tau；
不要一开始用 z_full + z_only_max union。
```

```python
def single_scale_anti_sufficiency_loss(z_only_max, gt, tau=0.5):
    """
    Conservative version.

    Penalize single-scale evidence that independently produces high response
    on safe-background regions.

    Args:
        z_only_max: Tensor, [B, 1, H, W]
        gt:         Tensor, [B, 1, H, W]
        tau:        threshold for z_only_max hard-background mining
    """
    safe_bg = build_safe_bg(gt)

    with torch.no_grad():
        hard_bg = safe_bg * (torch.sigmoid(z_only_max.detach()) > tau).float()

    loss_map = F.binary_cross_entropy_with_logits(
        z_only_max,
        torch.zeros_like(z_only_max),
        reduction='none',
    )

    loss = (loss_map * hard_bg).sum() / (hard_bg.sum() + 1e-6)

    log_vars = {
        'hard_bg_ratio': hard_bg.mean().detach(),
        'z_only_prob_mean': torch.sigmoid(z_only_max.detach()).mean(),
        'z_only_prob_max': torch.sigmoid(z_only_max.detach()).max(),
    }

    return loss, log_vars
```

---

## 12. `model/loss.py`：empty loss 当前保留，但默认不用

保留函数：

```python
def empty_evidence_loss(z_empty):
    return F.binary_cross_entropy_with_logits(
        z_empty,
        torch.zeros_like(z_empty),
    )
```

但第一轮建议：

```bash
--dea-lambda-empty 0
```

原因：

```text
z_empty = self.final(0) 的 loss 会压 self.final.bias；
当前 Pd 已经低，先不要额外压 final bias。
```

等 single-scale loss 稳定后，再尝试：

```bash
--dea-lambda-empty 0.0005
```

---

## 13. `model/loss.py`：decidability loss 当前保留，但默认不用

保留函数即可：

```python
def decidability_loss(d_logit, z_full, gt, tau=0.5):
    safe_bg = build_safe_bg(gt)

    pos = gt.float()

    with torch.no_grad():
        hard_bg = safe_bg * (torch.sigmoid(z_full.detach()) > tau).float()

    valid = torch.clamp(pos + hard_bg, max=1.0)
    label = pos

    loss_map = F.binary_cross_entropy_with_logits(
        d_logit,
        label,
        reduction='none',
    )

    loss = (loss_map * valid).sum() / (valid.sum() + 1e-6)

    log_vars = {
        'd_pos_ratio': pos.mean().detach(),
        'd_hard_bg_ratio': hard_bg.mean().detach(),
        'd_prob_mean': torch.sigmoid(d_logit.detach()).mean(),
        'd_prob_max': torch.sigmoid(d_logit.detach()).max(),
    }

    return loss, log_vars
```

但当前命令固定：

```bash
--dea-lambda-dec 0
```

原因：

```text
d_prob 基本全 1，说明这个分支还没学到有效背景负样本；
现在开启 lambda_dec 只会增加不稳定性。
```

---

## 14. `model/loss.py`：DEA-lite 总 loss

替换成这个更保守的版本。

```python
def dea_lite_loss(
    dea_out,
    z_full,
    gt,
    lambda_single=0.0,
    lambda_dec=0.0,
    lambda_empty=0.0,
    tau=0.5,
):
    device = z_full.device
    total_loss = torch.tensor(0.0, device=device)
    log_vars = {}

    if lambda_single > 0:
        loss_single, single_log = single_scale_anti_sufficiency_loss(
            dea_out['z_only_max'],
            gt,
            tau=tau,
        )
        total_loss = total_loss + lambda_single * loss_single
        log_vars['loss_single_raw'] = loss_single.detach()
        log_vars['loss_single_weighted'] = (lambda_single * loss_single).detach()
        log_vars.update(single_log)

    if lambda_empty > 0:
        loss_empty = empty_evidence_loss(dea_out['z_empty'])
        total_loss = total_loss + lambda_empty * loss_empty
        log_vars['loss_empty_raw'] = loss_empty.detach()
        log_vars['loss_empty_weighted'] = (lambda_empty * loss_empty).detach()

    if lambda_dec > 0:
        loss_dec, dec_log = decidability_loss(
            dea_out['decidability_logit'],
            z_full,
            gt,
            tau=tau,
        )
        total_loss = total_loss + lambda_dec * loss_dec
        log_vars['loss_dec_raw'] = loss_dec.detach()
        log_vars['loss_dec_weighted'] = (lambda_dec * loss_dec).detach()
        log_vars.update(dec_log)
    else:
        if 'decidability_logit' in dea_out:
            log_vars['d_prob_mean'] = torch.sigmoid(
                dea_out['decidability_logit'].detach()
            ).mean()
            log_vars['d_prob_max'] = torch.sigmoid(
                dea_out['decidability_logit'].detach()
            ).max()

    return total_loss, log_vars
```

---

## 15. `main.py`：ramp 函数

新增：

```python
def get_dea_ramp(epoch, warm_epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    if epoch <= warm_epoch:
        return 0.0
    return min(1.0, float(epoch - warm_epoch) / float(ramp_epochs))
```

训练时：

```python
ramp = get_dea_ramp(epoch, self.warm_epoch, args.dea_ramp_epochs)

cur_lambda_single = args.dea_lambda_single * ramp
cur_lambda_dec = args.dea_lambda_dec * ramp
cur_lambda_empty = args.dea_lambda_empty * ramp
```

---

## 16. `main.py`：加入 DEA loss

原始 MSHNet loss 保持不变，但注意不要污染 full-resolution label。

```python
loss = 0
loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)

labels_for_scale = labels
for j in range(len(masks)):
    if j > 0:
        labels_for_scale = self.down(labels_for_scale)
    loss = loss + self.loss_fun(masks[j], labels_for_scale, self.warm_epoch, epoch)

loss = loss / (len(masks) + 1)
loss_seg_for_debug = loss.detach()
```

然后只在 `use_dea=True` 时加：

```python
if use_dea:
    ramp = get_dea_ramp(epoch, self.warm_epoch, args.dea_ramp_epochs)

    cur_lambda_single = args.dea_lambda_single * ramp
    cur_lambda_dec = args.dea_lambda_dec * ramp
    cur_lambda_empty = args.dea_lambda_empty * ramp

    loss_dea, dea_log = dea_lite_loss(
        dea_out=dea_out,
        z_full=pred,
        gt=labels,
        lambda_single=cur_lambda_single,
        lambda_dec=cur_lambda_dec,
        lambda_empty=cur_lambda_empty,
        tau=args.dea_tau,
    )

    loss = loss + loss_dea
```

---

## 17. `main.py`：debug 打印

如果你有 `--save-dea-debug`，建议加：

```python
if use_dea and args.save_dea_debug and i % args.dea_debug_interval == 0:
    dea_ratio = (loss_dea.detach() / (loss_seg_for_debug + 1e-6)).item()

    msg = [
        f'dea_ratio={dea_ratio:.4f}',
        f'lambda_single={cur_lambda_single:.6f}',
        f'lambda_empty={cur_lambda_empty:.6f}',
        f'lambda_dec={cur_lambda_dec:.6f}',
    ]

    for k, v in dea_log.items():
        try:
            msg.append(f'{k}={float(v):.6f}')
        except Exception:
            pass

    print('[DEA DEBUG] ' + ' | '.join(msg))
```

重点观察：

```text
dea_ratio
hard_bg_ratio
z_only_prob_mean
z_only_prob_max
loss_single_weighted
loss_single_raw
d_prob_mean / d_prob_max
```

理想初期：

```text
weighted DEA loss / segmentation loss ≈ 1% ~ 5%
最多不要超过 10%
```

---

## 18. 第一轮运行：纯净 sanity baseline

命令：

```bash
cd /home/ly/MSHNet

/home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/MSHNet/datasets/IRSTD-1K \
  --batch-size 16 \
  --num-workers 0 \
  --pin-memory false \
  --epochs 140 \
  --lr 0.01 \
  --mode train \
  --if-checkpoint \
  --checkpoint-dir /home/ly/MSHNet/weight/MSHNet-2026-07-05-16-24-09 \
  --multi-gpus true \
  --gpu-ids 0,1,2,3 \
  --dea-lambda-single 0 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0
```

注意：这一步要求代码实际走：

```python
masks, pred = self.model(data, tag)
```

而不是：

```python
return_dea=True
```

判断：

```text
如果 sanity baseline 仍然 IoU / Pd 很低：
    问题在代码路径、数据、checkpoint 或训练配置。
    先不要继续 DEA。

如果 sanity baseline 正常：
    进入 Step 2，只开 conservative single-scale loss。
```

---

## 19. 第二轮运行：只开 conservative single-scale loss

第一轮建议比之前更保守：

```bash
/home/ly/BasicIRSTD/infrarenet/bin/python -u main.py \
  --dataset-dir /home/ly/MSHNet/datasets/IRSTD-1K \
  --batch-size 16 \
  --num-workers 0 \
  --pin-memory false \
  --epochs 180 \
  --lr 0.01 \
  --mode train \
  --if-checkpoint \
  --checkpoint-dir /home/ly/MSHNet/weight/MSHNet-2026-07-05-16-24-09 \
  --multi-gpus true \
  --gpu-ids 0,1,2,3 \
  --dea-lambda-single 0.02 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.5 \
  --dea-detach-evidence \
  --save-dea-debug
```

这里特意先设：

```bash
--dea-lambda-empty 0
```

不是：

```bash
--dea-lambda-empty 0.001
```

原因：当前 Pd 已经低，先不要让 empty loss 压 `self.final.bias`。

---

## 20. 如果第二轮还是掉 IoU / Pd

按下面顺序调，不要直接加复杂模块。

### 20.1 降低 single loss

```bash
--dea-lambda-single 0.01
```

如果仍然掉：

```bash
--dea-lambda-single 0.005
```

### 20.2 提高 hard background 阈值

```bash
--dea-tau 0.6
```

### 20.3 继续保持

```bash
--dea-detach-evidence
--dea-lambda-dec 0
--dea-lambda-empty 0
```

### 20.4 检查 DEA loss 比例

如果：

```text
dea_ratio > 0.1
```

说明 DEA auxiliary loss 对当前阶段太强。

---

## 21. 如果 sanity baseline 都低，该查什么

如果 DEA 全关后，IoU / Pd 仍然低，优先检查：

```text
1. use_dea 是否真的 False；
2. return_dea=False 时 forward 是否完全返回原始格式；
3. warm_flag=False 分支是否保持原样；
4. test 阶段是否仍然 _, pred = self.model(data, tag)；
5. labels 是否被多尺度 loss 里的 downsample 污染；
6. checkpoint 是否是 clean checkpoint，而不是失败 run 保存的权重；
7. --checkpoint-dir 是否同时被用作 resume 和 save，导致覆盖；
8. pred shape 是否仍为 [B, 1, H, W]；
9. 多卡 DataParallel 下新增参数是否正常加载；
10. 新增 decidability_head 是否导致 checkpoint strict load 异常。
```

尤其注意：

```python
labels_for_scale = labels
```

不要直接写：

```python
labels = self.down(labels)
```

否则后面的 DEA loss 和 full-resolution loss 可能拿到错误尺寸的 label。

---

## 22. 如果 baseline 正常，但 single loss 后掉 Pd

说明 DEA loss 或 hard background 选择仍然不稳。优先试：

```text
1. lambda_single: 0.02 -> 0.01 -> 0.005
2. tau: 0.5 -> 0.6
3. lambda_empty 保持 0
4. lambda_dec 保持 0
5. 保持 dea-detach-evidence
```

如果 detach evidence 后正常，不 detach 后掉 Pd，说明：

```text
DEA loss 直接冲击 encoder / decoder / scale heads 过强。
```

这时第一版论文/实验可以先采用 detach 版本，因为它更符合当前目标：

```text
先训练 final fusion 不要依赖单尺度虚假 evidence，暂时不改动 backbone evidence 生成。
```

---

## 23. 什么时候重新打开 empty loss

只有当下面条件满足后再试：

```text
1. sanity baseline 正常；
2. lambda_single=0.02 或 0.01 时 IoU / Pd 基本不掉；
3. FA / FP 有下降趋势；
4. hard_bg_ratio 合理；
5. dea_ratio 不超过 0.1。
```

再试：

```bash
--dea-lambda-empty 0.0005
```

如果 Pd 掉，继续关掉 empty loss。

---

## 24. 什么时候重新打开 decidability loss

现在不要打开。

只有当 single-scale loss 稳定后，再重新设计 `d_label` 和 hard negative 采样。

可以从很小开始：

```bash
--dea-lambda-dec 0.001
```

或者：

```bash
--dea-lambda-dec 0.005
```

但当前阶段固定：

```bash
--dea-lambda-dec 0
```

原因：

```text
d_prob 基本全 1，说明 d 分支还没有有效负样本。
```

---

## 25. 当前阶段成功标准

当前不是要证明 DEA-full，而是验证最小问题：

```text
只开 conservative single-scale anti-sufficiency 后，
FA / FP 是否下降，
Pd / IoU 是否不明显掉。
```

成功信号：

```text
FA 下降
FP components 下降
Precision 上升
Pd 基本不下降
IoU 持平或轻微变化
```

失败信号：

```text
IoU 明显下降
Pd 明显下降
loss 下降明显变慢
hard_bg_ratio 过大
dea_ratio 过大
z_only_prob 被快速整体压死
```

---

## 26. 最终执行顺序

```text
Step 0
停掉当前 run。

Step 1
加纯净开关：DEA 全关时 return_dea=False。

Step 2
确认 DEA 默认权重全部为 0。

Step 3
改 conservative single-scale loss：hard_bg 只来自 z_only_max。

Step 4
加 dea_detach_evidence。

Step 5
跑 sanity baseline：lambda_single=0, lambda_dec=0, lambda_empty=0。

Step 6
如果 baseline 正常，跑 single-only：
lambda_single=0.02, lambda_dec=0, lambda_empty=0, tau=0.5, detach_evidence=True。

Step 7
如果 Pd / IoU 掉：
lambda_single 降到 0.01 或 0.005；tau 提到 0.6；保持 dec/empty 关闭。

Step 8
只有 single-only 稳定后，再考虑 empty loss 和 decidability loss。
```

---

## 27. 最终结论

这个修改方向是对的。

当前最推荐版本是：

```text
DEA-lite conservative debug version
```

配置：

```text
return_dea only when DEA loss > 0
lambda_single default = 0
lambda_dec default = 0
lambda_empty default = 0
hard_bg = safe_bg * (sigmoid(z_only_max) > tau)
tau = 0.5
detach_evidence = True
lambda_dec = 0
lambda_empty = 0
```

第一轮不要追求完整 DEA。现在最关键是先证明：

> MSHNet 的一部分 false alarms 是否确实具有 single-scale sufficiency，并且能否用保守的 single-scale anti-sufficiency loss 降低 FA / FP，同时不明显损伤 Pd / IoU。

如果这个问题验证成功，DEA-lite 才有继续升级到 component-compatible selector、decidability loss、attribution alignment 和 inference-time gate 的价值。
