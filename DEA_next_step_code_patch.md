# DEA-lite 当前低 IoU / 低 Pd 后的下一步方案与代码修改

> 适用对象：当前 `Arialliy/DEA` 项目里的 DEA-lite / MSHNet 修改版。  
> 当前问题：loss 下降太慢，IoU 约 0.6，Pd 很低。  
> 当前目标：先恢复原始 MSHNet 主路径，再验证 `single-scale anti-sufficiency` 是否能降低 FA / FP，同时不明显损伤 IoU / Pd。  
> 注意：我没有在当前环境直接拉取到 `Arialliy/DEA` 仓库源码。因此下面按你已有 DEA-lite 结构和 `Lliu666/MSHNet` 风格给出可直接落地的补丁式修改。文件名、类名如与你仓库略有不同，按同名逻辑位置替换。

---

## 0. 当前判断

当前低 IoU / 低 Pd 不要继续硬训。现在最重要的是排查：

```text
是 DEA-lite auxiliary loss 把模型压坏了？
还是改过代码后，原始 MSHNet baseline 路径本身已经不等价了？
```

因此下一步顺序固定为：

```text
Step 1: 停掉当前 run
Step 2: 代码加保护开关，保证 DEA loss 全关时 return_dea=False
Step 3: 跑纯净 sanity baseline
Step 4: baseline 正常后，只开 single-scale anti-sufficiency
Step 5: 暂时不开 decidability loss，不做 inference-time gate
```

---

## 1. 先停掉当前 run

如果在前台终端：

```bash
Ctrl-C
```

如果后台 PID 是 `743370`：

```bash
kill -INT 743370
```

不要优先用：

```bash
kill -9 743370
```

只有 `kill -INT` 不响应时，再用 `kill -9`。

---

## 2. 必须先做的代码保护

### 2.1 目标

当：

```bash
--dea-lambda-single 0 \
--dea-lambda-dec 0 \
--dea-lambda-empty 0
```

时，训练必须完全走原始 MSHNet 路径：

```python
masks, pred = self.model(data, tag)
```

而不是：

```python
masks, pred, dea_out = self.model(data, tag, return_dea=True)
```

原因：即使 DEA loss 权重是 0，如果仍然 `return_dea=True`，虽然理论上不影响梯度，但 sanity baseline 不够干净，也容易引入返回格式、显存、debug 逻辑等额外风险。

---

## 3. `main.py` 增加参数

在 `argparse` 位置加入下面参数。如果你的仓库已经有其中一部分，只补缺失项。

```python
parser.add_argument('--dea-lambda-single', type=float, default=0.0,
                    help='Weight for DEA single-scale anti-sufficiency loss.')
parser.add_argument('--dea-lambda-dec', type=float, default=0.0,
                    help='Weight for DEA decidability loss.')
parser.add_argument('--dea-lambda-empty', type=float, default=0.0,
                    help='Weight for DEA empty-evidence loss.')
parser.add_argument('--dea-ramp-epochs', type=int, default=0,
                    help='Ramp epochs for DEA losses after warm-up.')
parser.add_argument('--dea-tau', type=float, default=0.5,
                    help='Threshold for hard safe-background mining.')
parser.add_argument('--dea-debug-interval', type=int, default=50,
                    help='Print DEA debug info every N iterations.')
parser.add_argument('--save-dea-debug', action='store_true',
                    help='Enable DEA debug logging.')
parser.add_argument('--dea-detach-evidence', action='store_true',
                    help='Detach scale logits in counterfactual DEA paths for stability.')
```

建议默认值设成：

```text
lambda_single = 0
lambda_dec    = 0
lambda_empty  = 0
tau            = 0.5
```

这样默认训练就是原始 MSHNet。

---

## 4. `model/MSHNet.py` 修改

### 4.1 修改 forward 函数签名

把：

```python
def forward(self, x, warm_flag, return_dea=False):
```

改成：

```python
def forward(self, x, warm_flag, return_dea=False, dea_detach_evidence=False):
```

如果你现在还是：

```python
def forward(self, x, warm_flag):
```

则直接改成上面的完整签名。

---

### 4.2 修改 `build_dea_lite_outputs`

用下面这个版本替换当前 `build_dea_lite_outputs`。

这个版本做了两个关键保护：

```text
1. 支持 dea_detach_evidence=True；
2. d_input 也使用稳定输入，避免 decidability 分支早期干扰主干。
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

### 4.3 forward 里调用 DEA 输出

在 `warm_flag=True` 分支里，确保是下面这种结构。

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

`warm_flag=False` 分支不要启用 DEA-lite。

---

## 5. `model/loss.py` 修改

### 5.1 增加 safe background

```python
import torch
import torch.nn.functional as F


def build_safe_bg(gt, kernel_size=15):
    """
    Args:
        gt: Tensor, [B, 1, H, W], binary mask.
    Returns:
        safe_bg: Tensor, [B, 1, H, W].
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

### 5.2 替换 single-scale anti-sufficiency loss

当前 IoU / Pd 低，建议先用更保守的 hard-bg 选择：

```text
hard_bg 只来自 z_only_max，而不是 z_full + z_only_max 的 union。
```

这样可以避免 early stage 把一些 full prediction 的弱目标附近响应当成背景压掉。

```python
def single_scale_anti_sufficiency_loss(z_only_max, gt, tau=0.5, kernel_size=15):
    """
    Penalize single-scale evidence that independently produces high response
    on hard safe-background regions.

    Args:
        z_only_max: Tensor, [B, 1, H, W]
        gt:         Tensor, [B, 1, H, W]
        tau:        hard background threshold
    """
    safe_bg = build_safe_bg(gt, kernel_size=kernel_size)

    with torch.no_grad():
        hard_bg = safe_bg * (torch.sigmoid(z_only_max.detach()) > tau).float()

    if hard_bg.sum() < 1:
        loss = z_only_max.sum() * 0.0
    else:
        loss_map = F.binary_cross_entropy_with_logits(
            z_only_max,
            torch.zeros_like(z_only_max),
            reduction='none',
        )
        loss = (loss_map * hard_bg).sum() / (hard_bg.sum() + 1e-6)

    stats = {
        'hard_bg_ratio': hard_bg.mean().detach(),
        'z_only_prob_mean': torch.sigmoid(z_only_max.detach()).mean(),
        'z_only_prob_max': torch.sigmoid(z_only_max.detach()).amax(),
    }
    return loss, stats
```

---

### 5.3 empty evidence loss

当前 Pd 低时，`lambda_empty` 建议先设为 0。函数保留即可。

```python
def empty_evidence_loss(z_empty):
    return F.binary_cross_entropy_with_logits(
        z_empty,
        torch.zeros_like(z_empty),
    )
```

---

### 5.4 decidability loss 暂时保留但不开

`d_prob` 基本全 1 时，`lambda_dec` 暂时必须为 0。函数可以保留，后续再用。

```python
def decidability_loss(d_logit, z_full, gt, tau=0.5, kernel_size=15):
    """
    Positive:
        GT target regions -> decidability = 1
    Negative:
        hard safe-background predicted regions -> decidability = 0
    Ignore:
        easy background
    """
    safe_bg = build_safe_bg(gt, kernel_size=kernel_size)

    pos = gt.float()

    with torch.no_grad():
        hard_bg = safe_bg * (torch.sigmoid(z_full.detach()) > tau).float()

    valid = torch.clamp(pos + hard_bg, max=1.0)
    label = pos

    if valid.sum() < 1:
        loss = d_logit.sum() * 0.0
    else:
        loss_map = F.binary_cross_entropy_with_logits(
            d_logit,
            label,
            reduction='none',
        )
        loss = (loss_map * valid).sum() / (valid.sum() + 1e-6)

    stats = {
        'd_prob_mean': torch.sigmoid(d_logit.detach()).mean(),
        'd_prob_max': torch.sigmoid(d_logit.detach()).amax(),
        'dec_valid_ratio': valid.mean().detach(),
    }
    return loss, stats
```

---

### 5.5 替换 DEA-lite 总 loss

这个版本有三个改动：

```text
1. lambda=0 时不计算对应分支；
2. single loss 默认 tau=0.5；
3. 返回 debug stats，方便判断 DEA loss 是否过强。
```

```python
def dea_lite_loss(
    dea_out,
    z_full,
    gt,
    lambda_single=0.0,
    lambda_dec=0.0,
    lambda_empty=0.0,
    tau=0.5,
    kernel_size=15,
):
    total = z_full.sum() * 0.0

    log_vars = {
        'loss_single': z_full.new_tensor(0.0),
        'loss_dec': z_full.new_tensor(0.0),
        'loss_empty': z_full.new_tensor(0.0),
        'hard_bg_ratio': z_full.new_tensor(0.0),
        'z_only_prob_mean': z_full.new_tensor(0.0),
        'z_only_prob_max': z_full.new_tensor(0.0),
        'd_prob_mean': z_full.new_tensor(0.0),
        'd_prob_max': z_full.new_tensor(0.0),
    }

    if lambda_single > 0:
        loss_single, stats_single = single_scale_anti_sufficiency_loss(
            dea_out['z_only_max'],
            gt,
            tau=tau,
            kernel_size=kernel_size,
        )
        total = total + lambda_single * loss_single
        log_vars['loss_single'] = loss_single.detach()
        log_vars.update({k: v.detach() for k, v in stats_single.items()})

    if lambda_dec > 0:
        loss_dec, stats_dec = decidability_loss(
            dea_out['decidability_logit'],
            z_full,
            gt,
            tau=tau,
            kernel_size=kernel_size,
        )
        total = total + lambda_dec * loss_dec
        log_vars['loss_dec'] = loss_dec.detach()
        log_vars.update({k: v.detach() for k, v in stats_dec.items()})

    if lambda_empty > 0:
        loss_empty = empty_evidence_loss(dea_out['z_empty'])
        total = total + lambda_empty * loss_empty
        log_vars['loss_empty'] = loss_empty.detach()

    return total, log_vars
```

---

## 6. `main.py` 训练逻辑修改

### 6.1 增加 ramp 函数

可以放在 `Trainer` 类里，也可以放在文件顶部。

```python
def get_ramped_lambda(base_lambda, epoch, warm_epoch, ramp_epochs):
    if base_lambda <= 0:
        return 0.0
    if ramp_epochs <= 0:
        return float(base_lambda)

    progress = (epoch - warm_epoch) / float(ramp_epochs)
    progress = max(0.0, min(1.0, progress))
    return float(base_lambda) * progress
```

---

### 6.2 训练 forward 改成纯净开关

把训练 loop 里的 forward 部分改成下面结构。

重点：

```text
只有 warm-up 之后，且至少一个 DEA lambda > 0 时，才 return_dea=True。
```

```python
tag = epoch > self.warm_epoch

base_dea_enabled = (
    self.args.dea_lambda_single > 0
    or self.args.dea_lambda_dec > 0
    or self.args.dea_lambda_empty > 0
)

use_dea = tag and base_dea_enabled

if use_dea:
    masks, pred, dea_out = self.model(
        data,
        tag,
        return_dea=True,
        dea_detach_evidence=self.args.dea_detach_evidence,
    )
else:
    masks, pred = self.model(data, tag)
    dea_out = None
```

如果你的 `args` 不是 `self.args`，而是 `args` 或其他名字，把 `self.args` 替换成你项目中的实际变量名。

---

### 6.3 原始 MSHNet loss 保持不变，但不要污染 labels

必须避免：

```python
labels = self.down(labels)
```

因为后面 DEA loss 需要 full-resolution GT。

正确写法：

```python
loss = 0
loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)

labels_for_scale = labels
for j in range(len(masks)):
    if j > 0:
        labels_for_scale = self.down(labels_for_scale)
    loss = loss + self.loss_fun(masks[j], labels_for_scale, self.warm_epoch, epoch)

loss = loss / (len(masks) + 1)
```

---

### 6.4 加 DEA loss

把 DEA loss 加在原始 loss 后面。

```python
if use_dea:
    lambda_single = get_ramped_lambda(
        self.args.dea_lambda_single,
        epoch,
        self.warm_epoch,
        self.args.dea_ramp_epochs,
    )
    lambda_dec = get_ramped_lambda(
        self.args.dea_lambda_dec,
        epoch,
        self.warm_epoch,
        self.args.dea_ramp_epochs,
    )
    lambda_empty = get_ramped_lambda(
        self.args.dea_lambda_empty,
        epoch,
        self.warm_epoch,
        self.args.dea_ramp_epochs,
    )

    loss_dea, dea_log = dea_lite_loss(
        dea_out,
        pred,
        labels,
        lambda_single=lambda_single,
        lambda_dec=lambda_dec,
        lambda_empty=lambda_empty,
        tau=self.args.dea_tau,
    )

    loss = loss + loss_dea

    if self.args.save_dea_debug and i % self.args.dea_debug_interval == 0:
        seg_value = float(loss.detach().item() - loss_dea.detach().item())
        dea_value = float(loss_dea.detach().item())
        ratio = dea_value / (abs(seg_value) + 1e-6)

        print(
            '[DEA] '
            f'epoch={epoch} iter={i} '
            f'lambda_single={lambda_single:.6f} '
            f'lambda_dec={lambda_dec:.6f} '
            f'lambda_empty={lambda_empty:.6f} '
            f'loss_dea={dea_value:.6f} '
            f'dea/seg={ratio:.4f} '
            f'loss_single={float(dea_log["loss_single"]):.6f} '
            f'loss_dec={float(dea_log["loss_dec"]):.6f} '
            f'loss_empty={float(dea_log["loss_empty"]):.6f} '
            f'hard_bg={float(dea_log["hard_bg_ratio"]):.6f} '
            f'z_only_mean={float(dea_log["z_only_prob_mean"]):.6f} '
            f'z_only_max={float(dea_log["z_only_prob_max"]):.6f} '
            f'd_mean={float(dea_log["d_prob_mean"]):.6f} '
            f'd_max={float(dea_log["d_prob_max"]):.6f}'
        )
```

---

## 7. 接下来怎么跑

### 7.1 Run A：纯净 sanity baseline

目的：确认改过 DEA 代码后，原始 MSHNet 主路径仍然正常。

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

重要：

```text
如果 /home/ly/MSHNet/weight/MSHNet-2026-07-05-16-24-09 是当前失败 run 覆盖出来的坏 checkpoint，不要从它继续。
```

这种情况下，换一个 clean checkpoint，或者从头跑 sanity baseline。

---

### 7.2 Run A 的判断

如果 sanity baseline 仍然：

```text
IoU 约 0.6
Pd 很低
loss 下降慢
```

先不要跑 DEA。优先检查：

```text
1. return_dea=False 时 forward 是否完全等价原始 MSHNet；
2. warm_flag=False 分支是否保持不变；
3. test 阶段是否仍然只调用 self.model(data, tag)，没有 return_dea=True；
4. labels 是否被 downsample 污染；
5. checkpoint 是否来自坏 run；
6. checkpoint load 是否因为新增 decidability_head 出现 strict 或 missing key 问题；
7. pred shape 是否仍是 [B, 1, H, W]。
```

如果 sanity baseline 正常，再进入 Run B。

---

### 7.3 Run B：只开 single-scale anti-sufficiency

目的：验证最小问题：

```text
只压制单尺度 sufficient background response，能否降低 FA / FP，同时不明显损伤 IoU / Pd？
```

建议先用更保守配置：

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

这里故意设置：

```text
lambda_dec = 0
lambda_empty = 0
lambda_single = 0.02
tau = 0.5
detach evidence = on
```

原因：当前已经出现 Pd 低，先把所有可能压低 target response 的因素都关掉，只验证 single-scale loss。

---

## 8. Run B 后如何判断

### 情况 1：IoU / Pd 基本恢复，FA / FP 下降

说明 DEA-lite 的核心假设成立：

```text
一部分 false alarms 具有 single-scale sufficient response，
压制 z_only_max 的 hard background 可以降低虚警。
```

下一步可以尝试：

```bash
--dea-lambda-empty 0.0005
```

或者把：

```bash
--dea-detach-evidence
```

去掉，观察能否进一步改善。

---

### 情况 2：IoU / Pd 仍然掉

按顺序降强度：

```bash
--dea-lambda-single 0.01 \
--dea-lambda-empty 0 \
--dea-lambda-dec 0 \
--dea-ramp-epochs 100 \
--dea-tau 0.6 \
--dea-detach-evidence
```

如果还掉：

```bash
--dea-lambda-single 0.005 \
--dea-lambda-empty 0 \
--dea-lambda-dec 0 \
--dea-ramp-epochs 100 \
--dea-tau 0.6 \
--dea-detach-evidence
```

如果仍然掉，说明不是权重问题，优先查 hard-bg mask 是否选错。

---

### 情况 3：IoU / Pd 正常，但 FA / FP 不降

说明 single-scale loss 可能太弱，或者 false alarms 不是 single-scale sufficient 类型。

下一步可以尝试：

```bash
--dea-lambda-single 0.03 \
--dea-tau 0.4 \
--dea-detach-evidence
```

或者把 hard-bg 选择从：

```python
hard_bg = safe_bg * (sigmoid(z_only_max) > tau)
```

改成更强的 union：

```python
hard_bg_from_full = (torch.sigmoid(z_full.detach()) > tau).float()
hard_bg_from_only = (torch.sigmoid(z_only_max.detach()) > tau).float()
hard_bg = safe_bg * torch.clamp(hard_bg_from_full + hard_bg_from_only, max=1.0)
```

但这个 union 版本不要作为当前第一选择。当前 Pd 低，先用 only-based hard_bg。

---

## 9. 当前不要做的事情

现在先不要做：

```text
1. lambda_dec > 0
2. inference-time d gate
3. component-compatible scale selector
4. all-16 subset
5. positive necessity loss
6. candidate MLP verifier
7. 用当前低 IoU / 低 Pd run 当正式结果
```

尤其不要现在打开：

```bash
--dea-lambda-dec > 0
```

因为你已经观察到：

```text
d_prob 基本全 1
```

这说明 decidability head 还没有学到有效的 hard-background negative。现在开 dec loss 只会增加不稳定性。

---

## 10. Debug 指标必须看

打开 `--save-dea-debug` 后，重点看：

```text
hard_bg_ratio
z_only_prob_mean
z_only_prob_max
loss_single
weighted DEA loss / segmentation loss
d_prob_mean
d_prob_max
```

理想情况：

```text
weighted DEA loss / segmentation loss <= 0.05
```

如果超过：

```text
0.10 ~ 0.20
```

第一版基本太强，容易压低 Pd / IoU。

---

## 11. 现阶段最终执行顺序

```text
1. 停掉当前 run。
2. 加上本 Markdown 的代码保护：
   - return_dea 纯净开关
   - dea_detach_evidence
   - conservative single loss
   - ramped lambda
   - debug logging
3. 跑 DEA 全关 sanity baseline。
4. baseline 正常后，只开：
   - lambda_single = 0.02
   - lambda_dec = 0
   - lambda_empty = 0
   - tau = 0.5
   - ramp = 80
   - detach evidence = on
5. 观察 IoU / Pd 是否保持，FA / FP 是否下降。
6. 如果 Pd 还掉：
   - lambda_single 降到 0.01 或 0.005
   - tau 提到 0.6
   - 保持 empty=0, dec=0
7. 只有 single-scale loss 稳定后，再考虑：
   - lambda_empty = 0.0005
   - lambda_dec = 0.005
   - component-compatible sufficiency
   - inference-time decidability gate
```

---

## 12. 当前一句话策略

> 现在不要继续完整 DEA，也不要开 decidability loss。先把代码改成“DEA 全关时完全等价 MSHNet”，确认 sanity baseline 正常；然后只开一个非常弱的 `single-scale anti-sufficiency loss`，并用 `detach evidence + tau=0.5 + ramp=80` 保护 Pd。这个阶段只验证最小假设：压制 `z_only_max` 的 hard safe-background response 是否能降低 false alarms，而不明显损伤 IoU / Pd。
