# DEA-lite 当前训练不稳定后的下一步方案与代码修改

> 适用仓库：`https://github.com/Arialliy/DEA` / 本地路径 `/home/ly/DEA`  
> 当前现象：`lambda_single=0.02` 时 FA 明显下降，但 IoU / Pd 也下降。  
> 当前目标：不要继续扩展 DEA-full；先找到一个不明显伤 Pd / IoU 的 single-scale anti-sufficiency operating point。  
> 本文只给下一步执行方案和可直接粘贴的代码修改块，不写伪代码。

---

## 0. 当前结果判断

你当前运行命令：

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
  --dea-lambda-single 0.02 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0 \
  --dea-ramp-epochs 80 \
  --dea-tau 0.5 \
  --dea-detach-evidence \
  --save-dea-debug
```

当前结果：

```text
sanity baseline best:
IoU 0.6592 / PD 0.9354 / FA 18.9786

lambda_single=0.02 best:
IoU 0.6423 / PD 0.8844 / FA 5.6177
```

结论：

```text
FA 下降明显：18.9786 -> 5.6177
PD 下降明显：0.9354 -> 0.8844
IoU 也下降：0.6592 -> 0.6423
```

所以当前 `lambda_single=0.02` 版本说明了一个有价值的诊断现象：

> single-scale anti-sufficiency loss 能压制一部分 false alarm，但当前强度过大，已经明显伤 Pd。

这版不适合作为主结果。下一步应该降强度，而不是开 `lambda_dec`、`lambda_empty`、component selector 或 inference-time gate。

---

## 1. 先停掉当前 run

如果在前台终端：

```bash
Ctrl-C
```

如果是后台进程，先找 PID：

```bash
ps -ef | grep main.py | grep -v grep
```

然后温和停止：

```bash
kill -INT <PID>
```

如果你确认当前 PID 是 `833386`，用：

```bash
kill -INT 833386
```

不要先用：

```bash
kill -9 833386
```

只有 `kill -INT` 没反应时，再用 `kill -9`。

---

## 2. 下一步总原则

现在不要继续增加机制。下一步只做三件事：

```text
1. 确认 DEA-off baseline 是否与当前 run 是同配置。
2. 如果 baseline 同配置正常，跑 lambda_single=0.01。
3. 如果 0.01 仍然伤 Pd，再跑 lambda_single=0.005 + tau=0.6。
```

当前继续保持关闭：

```text
--dea-lambda-dec 0
--dea-lambda-empty 0
inference-time d gate
component selector
all-16 subset
positive necessity
candidate verifier
```

原因：当前 `0.02` single loss 已经伤 Pd。再加 `decidability_loss` 或 `empty_loss` 会进一步增加压制风险。

---

## 3. 代码修改 1：确认 DEA 默认权重全部为 0

文件：`main.py`

在 argparse 中确认以下参数。如果已经存在，就把默认值改成下面这样。

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

不要默认启用：

```python
# 不要这样设默认值
# --dea-lambda-single 0.10
# --dea-lambda-dec 0.05
# --dea-lambda-empty 0.01
```

---

## 4. 代码修改 2：加入纯净 DEA 开关

文件：`main.py`

在训练循环中，调用模型之前加入下面代码。核心要求是：

> 当 `dea-lambda-single=0`、`dea-lambda-dec=0`、`dea-lambda-empty=0` 时，必须走原始 MSHNet forward，不要 `return_dea=True`。

把训练循环中原来的：

```python
masks, pred = self.model(data, tag)
```

替换为：

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

如果你的 `Trainer` 里使用的是 `self.args`，就在训练函数开头加：

```python
args = self.args
```

---

## 5. 代码修改 3：加入 DEA ramp

文件：`main.py`

在类外或文件顶部加入：

```python
def get_dea_ramp(epoch, warm_epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    if epoch <= warm_epoch:
        return 0.0
    return min(1.0, float(epoch - warm_epoch) / float(ramp_epochs))
```

在计算 DEA loss 前加入：

```python
ramp = get_dea_ramp(epoch, self.warm_epoch, args.dea_ramp_epochs)

cur_lambda_single = args.dea_lambda_single * ramp
cur_lambda_dec = args.dea_lambda_dec * ramp
cur_lambda_empty = args.dea_lambda_empty * ramp
```

---

## 6. 代码修改 4：保证原始 segmentation loss 不污染 labels

文件：`main.py`

把原始 MSHNet loss 写成下面这个版本。注意：不要在多尺度 loss 里直接修改 `labels`。

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

不要写成：

```python
# 不要这样写
for j in range(len(masks)):
    if j > 0:
        labels = self.down(labels)
    loss = loss + self.loss_fun(masks[j], labels, self.warm_epoch, epoch)
```

因为后面 DEA loss 需要 full-resolution `labels`。如果 `labels` 被下采样污染，DEA loss 会拿到错误尺寸或错误监督。

---

## 7. 代码修改 5：只在 use_dea=True 时加 DEA loss

文件：`main.py`

在原始 segmentation loss 之后加入：

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
else:
    loss_dea = torch.tensor(0.0, device=pred.device)
    dea_log = {}
```

确认 `main.py` 顶部已经导入：

```python
import torch
from model.loss import dea_lite_loss
```

如果你的 `dea_lite_loss` 已经在同一个 loss module 中导入过，就不要重复导入。

---

## 8. 代码修改 6：debug 打印

文件：`main.py`

在 `loss.backward()` 之前或 `optimizer.step()` 之后加入下面 debug 打印。

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

重点看：

```text
dea_ratio
hard_bg_ratio
z_only_prob_mean
z_only_prob_max
loss_single_raw
loss_single_weighted
d_prob_mean / d_prob_max
```

判断线：

```text
weighted DEA loss / segmentation loss 最好在 1% ~ 5%；
如果 dea_ratio 经常 > 0.1，说明 DEA auxiliary loss 仍然偏强。
```

---

## 9. 代码修改 7：MSHNet forward 支持 detach evidence

文件：`model/MSHNet.py`

把 forward 签名改成：

```python
def forward(self, x, warm_flag, return_dea=False, dea_detach_evidence=False):
```

在 `warm_flag=True` 分支中，确保结构是下面这样：

```python
if warm_flag:
    mask0 = self.output_0(x_d0)
    mask1 = self.output_1(x_d1)
    mask2 = self.output_2(x_d2)
    mask3 = self.output_3(x_d3)

    s0 = mask0
    s1 = self.up(mask1)
    s2 = self.up_4(mask2)
    s3 = self.up_8(mask3)

    scale_logits = torch.cat([s0, s1, s2, s3], dim=1)
    z_full = self.final(scale_logits)

    if return_dea:
        dea_out = self.build_dea_lite_outputs(
            scale_logits,
            z_full,
            detach_evidence=dea_detach_evidence,
        )
        return [mask0, mask1, mask2, mask3], z_full, dea_out

    return [mask0, mask1, mask2, mask3], z_full
else:
    output = self.output_0(x_d0)
    return [], output
```

`warm_flag=False` 分支不要启用 DEA。

---

## 10. 代码修改 8：替换 `build_dea_lite_outputs`

文件：`model/MSHNet.py`

用下面完整函数替换当前 `build_dea_lite_outputs`。

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

    z_empty = self.final(neutral)

    z_only_list = []
    for scale_idx in range(4):
        e_only = neutral.clone()
        e_only[:, scale_idx:scale_idx + 1] = cf_scale_logits[:, scale_idx:scale_idx + 1]
        z_only_i = self.final(e_only)
        z_only_list.append(z_only_i)

    z_only = torch.cat(z_only_list, dim=1)
    z_only_max = z_only.max(dim=1, keepdim=True)[0]
    z_only_var = z_only.var(dim=1, keepdim=True, unbiased=False)

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

确认 `__init__` 中有：

```python
self.decidability_head = nn.Sequential(
    nn.Conv2d(7, 8, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, kernel_size=1),
)
```

这里输入通道必须是 7：

```text
z_full       1
z_only_max   1
z_only_var   1
scale_logits 4
----------------
total        7
```

---

## 11. 代码修改 9：替换 conservative single-scale loss

文件：`model/loss.py`

确认文件顶部有：

```python
import torch
import torch.nn.functional as F
```

加入或替换 `build_safe_bg`：

```python
def build_safe_bg(gt, kernel_size=15):
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    gt = gt.float()
    pad = kernel_size // 2
    gt_dilate = F.max_pool2d(
        gt,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    safe_bg = (gt_dilate < 0.5).float()
    return safe_bg
```

用下面函数替换旧的 `single_scale_anti_sufficiency_loss`：

```python
def single_scale_anti_sufficiency_loss(z_only_max, gt, tau=0.5):
    """
    Conservative single-scale anti-sufficiency loss.

    Only mine hard background from z_only_max, not from z_full.
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

重要：旧版本如果有下面逻辑，必须删掉或不用：

```python
hard_bg_from_full = (torch.sigmoid(z_full) > tau).float()
hard_bg_from_only = (torch.sigmoid(z_only_max) > tau).float()
hard_bg = safe_bg * torch.clamp(hard_bg_from_full + hard_bg_from_only, max=1.0)
```

当前阶段不要用 `z_full + z_only_max` union 选 hard background。

---

## 12. 代码修改 10：保留 empty / decidability loss，但默认不用

文件：`model/loss.py`

保留：

```python
def empty_evidence_loss(z_empty):
    return F.binary_cross_entropy_with_logits(
        z_empty,
        torch.zeros_like(z_empty),
    )
```

保留：

```python
def decidability_loss(d_logit, z_full, gt, tau=0.5):
    safe_bg = build_safe_bg(gt)
    pos = gt.float()

    if pos.dim() == 3:
        pos = pos.unsqueeze(1)

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

但当前运行固定：

```bash
--dea-lambda-empty 0
--dea-lambda-dec 0
```

原因：

```text
1. empty loss 会压 self.final.bias，当前 Pd 已经偏低，先不要加；
2. d_prob 基本全 1，decidability 分支还没有有效负样本，先不要训练它。
```

---

## 13. 代码修改 11：替换 `dea_lite_loss`

文件：`model/loss.py`

用下面函数替换当前 `dea_lite_loss`。

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

## 14. 本地确认命令

修改后，先检查旧逻辑是否还存在：

```bash
cd /home/ly/DEA

grep -R "hard_bg_from_full" -n model main.py
grep -R "torch.clamp(hard_bg_from_full" -n model main.py
grep -R "return_dea=True" -n main.py model
grep -R "dea_detach_evidence" -n main.py model
grep -R "dea-lambda-single" -n main.py
```

期望：

```text
hard_bg_from_full: 不应再出现在当前 single loss 中
return_dea=True: 只应在 use_dea=True 分支出现
dea_detach_evidence: main.py 和 model/MSHNet.py 都应出现
dea-lambda-single: 默认值应为 0.0
```

做语法检查：

```bash
cd /home/ly/DEA
/home/ly/BasicIRSTD/infrarenet/bin/python -m py_compile main.py model/MSHNet.py model/loss.py
```

---

## 15. 如果 baseline 不是同配置，先重跑 DEA-off baseline

当前 DEA run 是：

```text
/home/ly/DEA
single GPU
batch-size 4
lr 0.05
epochs 400
no checkpoint
dataset-dir /home/ly/DEA/datasets/IRSTD-1K
```

如果你引用的 baseline 不是完全同配置，必须先跑同配置 DEA-off：

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
  --dea-lambda-single 0 \
  --dea-lambda-dec 0 \
  --dea-lambda-empty 0
```

这个 baseline 必须走：

```python
masks, pred = self.model(data, tag)
```

不能走：

```python
return_dea=True
```

如果 DEA-off baseline 都低，先不要跑 DEA。优先查：

```text
1. 数据路径和 split；
2. warm_flag 是否正常；
3. return_dea=False 是否真的纯净；
4. labels 是否被 downsample 污染；
5. optimizer / lr 是否和 baseline 一致；
6. checkpoint 是否来自失败 run。
```

---

## 16. 下一轮主命令：lambda_single=0.01

如果 baseline 同配置正常，下一轮跑：

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
  --save-dea-debug
```

这一轮目标：

```text
不是把 FA 压到最低；
而是让 Pd / IoU 尽量接近 baseline，同时 FA 明显低于 baseline。
```

参考目标：

```text
IoU 接近 0.659
PD 接近 0.935
FA 明显低于 18.98
```

如果结果类似下面这样，就比 `0.02` 更有价值：

```text
IoU 0.65+
PD 0.92+
FA 10~15
```

---

## 17. 如果 0.01 仍然伤 Pd：lambda_single=0.005 + tau=0.6

如果 `0.01` 版本仍然出现：

```text
PD < baseline - 0.03
或 IoU < baseline - 0.01
```

就跑更保守版本：

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
  --save-dea-debug
```

`tau=0.6` 的作用：

```text
只惩罚更高置信的 z_only_max hard background；
减少误压弱目标和目标附近模糊区域的风险。
```

---

## 18. 当前不要打开这些开关

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
1. 当前 0.02 single loss 已经明显伤 Pd；
2. d_prob 基本全 1，decidability 分支没有学到有效负样本；
3. empty loss 会压 self.final.bias；
4. inference-time d gate 很可能进一步压掉弱小真目标；
5. component selector / all-16 subset 会引入新变量，当前阶段不利于定位问题。
```

---

## 19. 什么时候可以重新打开 empty loss

只有满足下面条件后再试：

```text
1. lambda_single=0.01 或 0.005 时 Pd / IoU 基本不掉；
2. FA 有下降；
3. dea_ratio 不经常超过 0.1；
4. hard_bg_ratio 不异常大。
```

再尝试：

```bash
--dea-lambda-empty 0.0005
```

如果 Pd 下降，继续关掉 empty loss。

---

## 20. 什么时候可以重新打开 decidability loss

现在不要打开。

等 single loss 稳定后，再从极小值开始：

```bash
--dea-lambda-dec 0.001
```

或：

```bash
--dea-lambda-dec 0.005
```

但要先解决 `d_prob 基本全 1` 的问题。否则 `decidability_loss` 只会增加不稳定性。

---

## 21. 当前阶段的成功标准

当前阶段不是证明 DEA-full，而是验证最小问题：

> 只开 conservative single-scale anti-sufficiency 后，FA / FP 是否下降，同时 Pd / IoU 是否不明显掉。

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
dea_ratio 经常 > 0.1
hard_bg_ratio 异常大
z_only_prob 被整体压死
```

---

## 22. 最终执行顺序

```text
Step 1
停掉当前 lambda_single=0.02 run。

Step 2
确认代码已经是 conservative 版本：
- DEA 默认权重为 0；
- DEA 全关时 return_dea=False；
- hard_bg 只来自 z_only_max；
- 支持 --dea-detach-evidence。

Step 3
如果 baseline 不是同配置，跑同配置 DEA-off baseline。

Step 4
baseline 正常后，跑 lambda_single=0.01, tau=0.5。

Step 5
如果 0.01 仍然伤 Pd，跑 lambda_single=0.005, tau=0.6。

Step 6
只有 single-only 稳定后，再考虑 empty loss、decidability loss 和完整 DEA。
```

---

## 23. 最终结论

当前 `lambda_single=0.02` 不是代码完全坏了，而是压制强度过大：

```text
FA 降得很明显，但 Pd 掉得太多。
```

下一步不应继续跑 400，也不应打开更多 DEA 机制。正确路线是：

```text
先停当前 run；
确认代码 conservative；
跑同配置 DEA-off baseline；
然后跑 lambda_single=0.01；
若仍伤 Pd，再跑 lambda_single=0.005 + tau=0.6。
```

当前最重要的目标是找到一个稳定平衡点：

> FA 明显低于 baseline，但 Pd / IoU 基本接近 baseline。
