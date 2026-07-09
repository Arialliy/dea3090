# DEA / DEA-lite 下一步修改建议：面向 SOTA 与问题解决的结构重构方案

当前目标不是“最小改造赶投稿”，而是：

> **把 DEA 真正做成有效结构：要么冲 SOTA，要么至少清楚解决它声称要解决的 false alarm / evidence ambiguity 问题。**

核心判断：

> **当前问题最大的不是 DEA-lite 的 λ 没调好，而是 Full DEA prototype 的结构方向不够对。它现在更像在 `x_d0` 后面接了一个二值 attention head，而不是在 MSHNet 的多尺度 evidence fusion 处做“目标证据 / 杂波证据 / 反事实抑制”。**

下面按以下顺序分析：

1. 哪个结构有问题
2. 应该改成什么
3. 代码怎么改
4. 怎么验证
5. 怎么进一步冲 SOTA

---

# 1. 当前最有问题的结构是哪一个？

## 1.1 问题一：Full DEA 现在插在 `x_d0` 后面，绕开了 MSHNet 真正的问题发生点

原始 MSHNet 的 final prediction 是由四个 scale mask 上采样后 concat，再经过：

```python
self.final = nn.Conv2d(4, 1, 3, 1, 1)
```

融合得到的。

也就是说，原始 MSHNet 的最终预测依赖：

```text
output_0
output_1
output_2
output_3
```

四个尺度的 logits。

DEA-lite 的前验证正好说明：问题发生在 **四尺度 evidence fusion** 这里。

DEA-lite 构造：

```python
scale_logits = torch.cat([s0, s1, s2, s3], dim=1)
z_full = self.final(scale_logits)
z_only
z_empty
```

本质是在诊断：

```text
某个 scale 是否单独足以触发背景误检？
```

但是当前 Full DEA prototype 是把 `FullDEAHead` 插在最终 decoder feature `x_d0` 后面，而不是插在多尺度 evidence fusion 处。

这就是第一个核心结构问题：

```text
DEA-lite 发现的问题：多尺度 fusion 里的 evidence ambiguity
当前 Full DEA 作用的位置：x_d0 单尺度 decoder feature
结构不匹配。
```

如果目标是 SOTA 或真正解决问题，Full DEA 必须改成：

```text
输入 = 多尺度 feature + 四尺度 logits + baseline fused logit
输出 = target evidence、clutter evidence、clutter suppression、final calibrated logit
```

而不是只对 `x_d0` 做一个二分类 head。

---

## 1.2 问题二：当前 counterfactual branch 被重新混回 final，逻辑上容易把 clutter 又加回来

当前 `FullDEAHead` 的逻辑大致是：

```python
cf_gate = sigmoid(clutter_evidence_logit - target_evidence_logit)
counterfactual_feature = decoder_feature * cf_gate
y_cf = counterfactual_head(counterfactual_feature)

evidence_gate = sigmoid(gate_head(...))
y_final = evidence_gate * y_real + (1.0 - evidence_gate) * y_cf
```

也就是说：

```text
y_cf 是 counterfactual / clutter path
但 final prediction 又把 y_cf 混回了最终输出
```

这在语义上有问题。

如果 `y_cf` 表示：

```text
杂波条件下会产生的响应
```

或者：

```text
反事实 clutter-only prediction
```

那么它不应该作为 positive segmentation logit 被加回 final。

它应该作为 **负证据 / suppression term** 去扣除 final logit。

更合理的形式是：

```python
z_final = z_target - alpha * suppression_gate * softplus(z_clutter)
```

其中：

```text
z_target             : 目标分支预测
z_clutter            : 杂波分支预测
suppression_gate     : 哪里应该执行 clutter suppression
softplus(z_clutter)  : 保证被扣除项非负
```

这个结构和 DEA 要解决的问题更一致：

> **不是在 real prediction 和 counterfactual prediction 之间二选一，而是用 counterfactual clutter evidence 去校准 / 抑制 real target prediction。**

---

## 1.3 问题三：target evidence / clutter evidence 没有真正监督，容易 collapse

当前 `full_dea_loss.py` 的主要逻辑包括：

```python
seg_loss = SoftIoULoss(y_final, target)
cf_loss = BCE(y_cf, 0) on target
bg_loss = BCE(y_final, 0) on safe background
sep_loss = mean(target_evidence * clutter_evidence)
```

但它没有明确告诉模型：

```text
GT target 区域：
    target evidence 应该高
    clutter evidence 应该低

hard false-alarm background 区域：
    clutter evidence 应该高
    target evidence 应该低
```

当前只用了：

```python
sep_loss = target_evidence * clutter_evidence
```

这会允许一个很糟糕的退化解：

```text
target_evidence ≈ 0
clutter_evidence ≈ 0
sep_loss 很小
但 evidence 没有语义
```

也就是说，当前 Full DEA 代码更像 structural smoke test，不是真正可训练的 evidence decomposition。

---

## 1.4 问题四：当前 FullDEAMSHNet 的 scale mask 返回契约不对

当前 `_build_scale_logits()` 返回的是：

```python
[
    mask0,
    self.up(mask1),
    self.up_4(mask2),
    self.up_8(mask3),
]
```

也就是全部被上采样到了 full resolution。

但原始 `main.py` 的多尺度辅助监督逻辑是：

```python
for j in range(len(masks)):
    if j > 0:
        labels_for_scale = self.down(labels_for_scale)
    loss = loss + self.loss_fun(masks[j], labels_for_scale, ...)
```

也就是说，`masks[j]` 应该是原尺度：

```python
[mask0, mask1, mask2, mask3]
```

然后 label 逐级 downsample 去匹配。

如果 FullDEAMSHNet 直接把 full-resolution scale logits 当作 `masks` 返回，一旦接入原训练流程，就会破坏多尺度监督。

这个不是 SOTA 问题，而是基本训练 contract 问题，必须修。

---

# 2. 下一步不应该继续调 DEA-lite，而是重构成 Full DEA v2

建议下一版结构命名为：

```text
Full DEA v2: Counterfactual Clutter-Suppressed Evidence Aggregation
```

核心思想：

```text
MSHNet baseline fused logit 作为 identity prior；
DEA 不从零重新预测 mask；
DEA 学一个 target residual 和一个 clutter suppression；
初始状态几乎等价于 baseline；
训练后只在 hard false alarm 区域做有监督抑制。
```

这样比“重新接一个 head 预测 y_final”稳定得多，也更有希望超过 baseline。

---

# 3. 推荐的新结构

新结构应该如下：

```text
输入：
    x_d0, x_d1, x_d2, x_d3          # 多尺度 decoder features
    mask0, mask1, mask2, mask3      # 原尺度 scale logits
    scale_logits_full               # 上采样后的四尺度 logits
    z_base                          # 原 MSHNet final(scale_logits_full)

Full DEA Head：
    1. 多尺度 feature fusion
    2. target evidence / clutter evidence decomposition
    3. target residual prediction
    4. clutter prediction
    5. clutter suppression gate
    6. final calibrated prediction

输出：
    z_base
    z_target = z_base + target_delta
    z_clutter
    suppression_gate
    z_final = z_target - alpha * suppression_gate * softplus(z_clutter)
```

这个结构有几个优点：

1. `z_base` 是原 MSHNet 输出，DEA 初始时可以几乎等价于 baseline，不容易一上来掉性能。
2. DEA 作用在多尺度 fusion 之后，不再绕开 MSHNet 的问题发生点。
3. counterfactual branch 不再被混回 final，而是作为负证据扣除。
4. target evidence / clutter evidence 可以用 GT target 和 hard false-alarm pseudo-label 显式监督。

---

# 4. 代码修改方案

建议直接重写三个文件：

```text
model/full_dea_head.py
model/full_dea_mshnet.py
model/full_dea_loss.py
```

---

## 4.1 替换 `model/full_dea_head.py`

这个版本不是 minimal head，而是 SOTA-oriented 的稳定结构：

```text
baseline-preserving residual + clutter subtractive calibration
```

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FullDEAHeadV2(nn.Module):
    """
    Full DEA v2.

    Design:
        z_base   : original MSHNet fused logit
        z_target : baseline-preserving target logit
        z_clutter: clutter evidence logit
        z_final  : target logit minus clutter suppression

    This is intentionally initialized close to the original MSHNet:
        target_delta ~= 0
        clutter_logit << 0
        suppression_gate ~= 0
        z_final ~= z_base
    """

    def __init__(
        self,
        hidden_channels: int = 32,
        scale_channels: int = 4,
    ):
        super().__init__()

        h = hidden_channels

        # Project decoder features to a common channel count.
        self.proj0 = ConvBNAct(16, h, kernel_size=1)
        self.proj1 = ConvBNAct(32, h, kernel_size=1)
        self.proj2 = ConvBNAct(64, h, kernel_size=1)
        self.proj3 = ConvBNAct(128, h, kernel_size=1)

        self.feature_fuse = nn.Sequential(
            ConvBNAct(h * 4, h * 2, kernel_size=3),
            ConvBNAct(h * 2, h, kernel_size=3),
        )

        # scale_logits: 4 channels
        # scale stats: mean, max, min, var = 4 channels
        scale_stat_channels = scale_channels + 4

        self.scale_fuse = nn.Sequential(
            ConvBNAct(scale_stat_channels, h // 2, kernel_size=3),
            ConvBNAct(h // 2, h // 2, kernel_size=3),
        )

        evidence_in_channels = h + h // 2
        self.evidence_head = nn.Sequential(
            ConvBNAct(evidence_in_channels, h, kernel_size=3),
            nn.Conv2d(h, 2, kernel_size=1),
        )

        # Target branch predicts only a residual on top of z_base.
        target_in_channels = h + scale_channels + 3
        self.target_delta_head = nn.Sequential(
            ConvBNAct(target_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        # Clutter branch predicts positive clutter evidence.
        clutter_in_channels = h + scale_channels + 3
        self.clutter_head = nn.Sequential(
            ConvBNAct(clutter_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        # Suppression gate decides where clutter evidence should be subtracted.
        gate_in_channels = h + 10

        self.suppression_head = nn.Sequential(
            ConvBNAct(gate_in_channels, h, kernel_size=3),
            ConvBNAct(h, h, kernel_size=3),
            nn.Conv2d(h, 1, kernel_size=1),
        )

        # Learnable positive suppression strength.
        self.log_alpha = nn.Parameter(torch.tensor(-1.0))

        self._init_close_to_baseline()

    def _init_close_to_baseline(self) -> None:
        """
        Important for stability:
        before DEA learns useful clutter evidence, z_final should be close to z_base.
        """
        final_delta = self.target_delta_head[-1]
        nn.init.zeros_(final_delta.weight)
        nn.init.zeros_(final_delta.bias)

        final_clutter = self.clutter_head[-1]
        nn.init.zeros_(final_clutter.weight)
        nn.init.constant_(final_clutter.bias, -4.0)

        final_suppress = self.suppression_head[-1]
        nn.init.zeros_(final_suppress.weight)
        nn.init.constant_(final_suppress.bias, -4.0)

    @staticmethod
    def _scale_stats(scale_logits_full: torch.Tensor) -> torch.Tensor:
        scale_mean = scale_logits_full.mean(dim=1, keepdim=True)
        scale_max = scale_logits_full.max(dim=1, keepdim=True)[0]
        scale_min = scale_logits_full.min(dim=1, keepdim=True)[0]
        scale_var = scale_logits_full.var(dim=1, keepdim=True, unbiased=False)

        return torch.cat(
            [
                scale_logits_full,
                scale_mean,
                scale_max,
                scale_min,
                scale_var,
            ],
            dim=1,
        )

    def forward(
        self,
        x_d0: torch.Tensor,
        x_d1: torch.Tensor,
        x_d2: torch.Tensor,
        x_d3: torch.Tensor,
        scale_logits_full: torch.Tensor,
        z_base: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        size = x_d0.shape[-2:]

        f0 = self.proj0(x_d0)
        f1 = F.interpolate(self.proj1(x_d1), size=size, mode="bilinear", align_corners=True)
        f2 = F.interpolate(self.proj2(x_d2), size=size, mode="bilinear", align_corners=True)
        f3 = F.interpolate(self.proj3(x_d3), size=size, mode="bilinear", align_corners=True)

        fused_feature = self.feature_fuse(torch.cat([f0, f1, f2, f3], dim=1))

        scale_stats = self._scale_stats(scale_logits_full)
        scale_feature = self.scale_fuse(scale_stats)

        evidence_logits = self.evidence_head(torch.cat([fused_feature, scale_feature], dim=1))
        target_evidence_logit, clutter_evidence_logit = torch.chunk(evidence_logits, chunks=2, dim=1)

        target_evidence = torch.sigmoid(target_evidence_logit)
        clutter_evidence = torch.sigmoid(clutter_evidence_logit)

        target_gate = torch.sigmoid(target_evidence_logit - clutter_evidence_logit)
        clutter_gate = torch.sigmoid(clutter_evidence_logit - target_evidence_logit)

        # Residual target prediction. z_target starts as z_base.
        target_feature = fused_feature * (1.0 + target_gate)
        target_input = torch.cat(
            [
                target_feature,
                scale_logits_full,
                z_base,
                target_evidence_logit,
                clutter_evidence_logit,
            ],
            dim=1,
        )
        target_delta = self.target_delta_head(target_input)
        z_target = z_base + target_delta

        # Clutter prediction. This should become high on hard false-alarm background.
        clutter_feature = fused_feature * (1.0 + clutter_gate)
        clutter_input = torch.cat(
            [
                clutter_feature,
                scale_logits_full,
                z_base,
                target_evidence_logit,
                clutter_evidence_logit,
            ],
            dim=1,
        )
        z_clutter = self.clutter_head(clutter_input)

        scale_aux_stats = scale_stats[:, 4:, :, :]
        gate_input = torch.cat(
            [
                fused_feature,
                target_evidence_logit,
                clutter_evidence_logit,
                z_target,
                z_clutter,
                target_gate,
                clutter_gate,
                scale_aux_stats,
            ],
            dim=1,
        )
        suppression_logit = self.suppression_head(gate_input)
        suppression_gate = torch.sigmoid(suppression_logit)

        alpha = F.softplus(self.log_alpha) + 1e-6

        # Key change:
        # counterfactual clutter evidence is subtracted, not mixed back as positive prediction.
        z_final = z_target - alpha * suppression_gate * F.softplus(z_clutter)

        return {
            "z_base": z_base,
            "scale_logits_full": scale_logits_full,
            "fused_feature": fused_feature,

            "target_evidence_logit": target_evidence_logit,
            "clutter_evidence_logit": clutter_evidence_logit,
            "target_evidence": target_evidence,
            "clutter_evidence": clutter_evidence,

            "target_gate": target_gate,
            "clutter_gate": clutter_gate,

            "target_delta": target_delta,
            "z_target": z_target,
            "z_clutter": z_clutter,

            "suppression_logit": suppression_logit,
            "suppression_gate": suppression_gate,
            "alpha": alpha.detach(),

            "y_real": z_target,
            "y_cf": z_clutter,
            "y_final": z_final,
            "z_final": z_final,
        }
```

这个 head 的关键点是：

```text
初始 z_final ≈ z_base
训练稳定性远高于从零学习 y_final
counterfactual clutter branch 只负责减法抑制
```

---

## 4.2 替换 `model/full_dea_mshnet.py`

重点修两个地方：

1. `masks` 返回原尺度 `[mask0, mask1, mask2, mask3]`
2. Full DEA 使用 full-resolution scale logits 和 `z_base`

```python
from __future__ import annotations

import torch

from model.MSHNet import MSHNet, ResNet
from model.full_dea_head import FullDEAHeadV2


class FullDEAMSHNet(MSHNet):
    """
    MSHNet + Full DEA v2.

    This class preserves the original MSHNet multi-scale supervision contract:
        masks = [mask0, mask1, mask2, mask3] at their native resolutions.

    Full DEA operates on:
        full-resolution scale logits,
        original MSHNet fused logit z_base,
        multi-scale decoder features.
    """

    def __init__(self, input_channels: int, block=ResNet):
        super().__init__(input_channels, block=block)
        self.full_dea_head = FullDEAHeadV2(hidden_channels=32)

    def _forward_features(self, x: torch.Tensor):
        x_e0 = self.encoder_0(self.conv_init(x))
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        x_m = self.middle_layer(self.pool(x_e3))

        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], dim=1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], dim=1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], dim=1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], dim=1))

        return x_d0, x_d1, x_d2, x_d3

    def _build_masks_and_fullres_scale_logits(self, x_d0, x_d1, x_d2, x_d3):
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)

        masks = [mask0, mask1, mask2, mask3]

        s0 = mask0
        s1 = self.up(mask1)
        s2 = self.up_4(mask2)
        s3 = self.up_8(mask3)

        scale_logits_full = torch.cat([s0, s1, s2, s3], dim=1)
        return masks, scale_logits_full

    def forward(
        self,
        x: torch.Tensor,
        warm_flag: bool,
        return_full_dea: bool = False,
        return_dict: bool = False,
    ):
        x_d0, x_d1, x_d2, x_d3 = self._forward_features(x)

        if not warm_flag:
            pred = self.output_0(x_d0)
            if return_dict:
                return {
                    "masks": [],
                    "pred": pred,
                    "z_base": pred,
                    "full_dea": None,
                }
            return [], pred

        masks, scale_logits_full = self._build_masks_and_fullres_scale_logits(
            x_d0,
            x_d1,
            x_d2,
            x_d3,
        )

        z_base = self.final(scale_logits_full)

        full_dea_out = self.full_dea_head(
            x_d0=x_d0,
            x_d1=x_d1,
            x_d2=x_d2,
            x_d3=x_d3,
            scale_logits_full=scale_logits_full,
            z_base=z_base,
        )

        pred = full_dea_out["z_final"]

        if return_dict:
            return {
                "masks": masks,
                "pred": pred,
                "z_base": z_base,
                "scale_logits_full": scale_logits_full,
                "full_dea": full_dea_out,
            }

        if return_full_dea:
            return masks, pred, full_dea_out

        return masks, pred
```

这里的关键是：

```text
masks 给原多尺度 loss 用；
scale_logits_full 给 DEA 用；
z_base 是原 MSHNet final output；
z_final 是 DEA-calibrated output。
```

---

## 4.3 替换 / 新增 `model/full_dea_loss.py`

这个 loss 的核心不是简单压 safe background，而是构造：

```text
hard false-alarm pseudo clutter label
```

因为 IRSTD 没有 clutter 标注，不能直接监督 clutter evidence。

所以要用在线 hard negative：

```text
hard_bg = safe background 中当前模型 / baseline / scale branch 高响应的位置
```

这才是 DEA 应该处理的“杂波证据”。

```python
from __future__ import annotations

import torch
import torch.nn.functional as F

from model.loss import SoftIoULoss, build_safe_bg


def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (loss_map * mask).sum() / (mask.sum() + eps)


def _topk_safe_mask(
    score: torch.Tensor,
    safe_bg: torch.Tensor,
    ratio: float = 0.002,
) -> torch.Tensor:
    """
    Select top-k safe-background pixels per image as hard pseudo clutter.

    score:   [B,1,H,W]
    safe_bg: [B,1,H,W]
    """
    b, _, h, w = score.shape
    score = score.detach() * safe_bg
    flat_score = score.view(b, -1)
    flat_safe = safe_bg.view(b, -1)

    k = max(1, int(h * w * ratio))

    masks = []
    for i in range(b):
        valid_idx = torch.nonzero(flat_safe[i] > 0.5, as_tuple=False).view(-1)

        if valid_idx.numel() == 0:
            masks.append(torch.zeros_like(flat_safe[i]))
            continue

        local_score = flat_score[i, valid_idx]
        local_k = min(k, valid_idx.numel())
        top_idx_local = torch.topk(local_score, k=local_k, largest=True).indices
        top_idx = valid_idx[top_idx_local]

        m = torch.zeros_like(flat_safe[i])
        m[top_idx] = 1.0
        masks.append(m)

    return torch.stack(masks, dim=0).view(b, 1, h, w)


def build_hard_clutter_label(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    safe_kernel: int = 15,
    topk_ratio: float = 0.002,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct pseudo clutter labels from online false-alarm-like responses.

    hard_bg is not "all background".
    It is safe background with suspiciously high model / scale response.
    """
    target = target.float()
    safe_bg = build_safe_bg(target, kernel_size=safe_kernel)

    z_base = full_dea_out["z_base"]
    z_target = full_dea_out["z_target"]
    scale_logits_full = full_dea_out["scale_logits_full"]

    with torch.no_grad():
        p_base = torch.sigmoid(z_base.detach())
        p_target = torch.sigmoid(z_target.detach())
        p_scale = torch.sigmoid(scale_logits_full.detach()).max(dim=1, keepdim=True)[0]

        hard_by_threshold = safe_bg * (
            (p_base > tau_base).float()
            + (p_target > tau_target).float()
            + (p_scale > tau_scale).float()
        )
        hard_by_threshold = (hard_by_threshold > 0).float()

        hard_score = torch.maximum(torch.maximum(p_base, p_target), p_scale)
        hard_by_topk = _topk_safe_mask(hard_score, safe_bg, ratio=topk_ratio)

        hard_bg = torch.maximum(hard_by_threshold, hard_by_topk)
        hard_bg = hard_bg * safe_bg

    return hard_bg, safe_bg


def full_dea_aux_loss_v2(
    full_dea_out: dict[str, torch.Tensor],
    target: torch.Tensor,
    epoch: int,
    warm_epoch: int,
    seg_criterion=None,
    tau_base: float = 0.45,
    tau_target: float = 0.45,
    tau_scale: float = 0.45,
    margin: float = 1.0,
    lambda_target_aux: float = 0.30,
    lambda_ev_target: float = 0.20,
    lambda_ev_clutter: float = 0.20,
    lambda_clutter_pred: float = 0.20,
    lambda_suppress_gate: float = 0.10,
    lambda_margin: float = 0.05,
    lambda_hard_bg_final: float = 0.10,
    lambda_suppress_order: float = 0.05,
    safe_kernel: int = 15,
    topk_ratio: float = 0.002,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    target = target.float()
    device = target.device

    z_base = full_dea_out["z_base"]
    z_target = full_dea_out["z_target"]
    z_final = full_dea_out["z_final"]
    z_clutter = full_dea_out["z_clutter"]

    e_t_logit = full_dea_out["target_evidence_logit"]
    e_c_logit = full_dea_out["clutter_evidence_logit"]
    suppression_logit = full_dea_out["suppression_logit"]

    hard_bg, safe_bg = build_hard_clutter_label(
        full_dea_out=full_dea_out,
        target=target,
        tau_base=tau_base,
        tau_target=tau_target,
        tau_scale=tau_scale,
        safe_kernel=safe_kernel,
        topk_ratio=topk_ratio,
    )

    # Valid pixels for evidence supervision:
    #   target pixels and hard clutter pixels.
    # Do not force all background to be clutter=0 too strongly,
    # otherwise small target recall can suffer.
    valid_ev = torch.clamp(target + hard_bg, 0.0, 1.0)

    # Optional target auxiliary segmentation.
    if seg_criterion is None:
        loss_target_aux = SoftIoULoss(z_target, target)
    else:
        loss_target_aux = seg_criterion(z_target, target, warm_epoch, epoch)

    # Target evidence:
    #   target -> 1
    #   hard clutter -> 0
    loss_ev_t_map = F.binary_cross_entropy_with_logits(
        e_t_logit,
        target,
        reduction="none",
    )
    loss_ev_t = _masked_mean(loss_ev_t_map, valid_ev)

    # Clutter evidence:
    #   hard clutter -> 1
    #   target -> 0
    loss_ev_c_map = F.binary_cross_entropy_with_logits(
        e_c_logit,
        hard_bg,
        reduction="none",
    )
    loss_ev_c = _masked_mean(loss_ev_c_map, valid_ev)

    # Clutter branch prediction:
    #   hard clutter -> high
    #   target -> low
    loss_clutter_map = F.binary_cross_entropy_with_logits(
        z_clutter,
        hard_bg,
        reduction="none",
    )
    loss_clutter_pred = _masked_mean(loss_clutter_map, valid_ev)

    # Suppression gate:
    #   hard clutter -> suppress
    #   target -> do not suppress
    loss_sup_map = F.binary_cross_entropy_with_logits(
        suppression_logit,
        hard_bg,
        reduction="none",
    )
    loss_suppress_gate = _masked_mean(loss_sup_map, valid_ev)

    # Evidence margin:
    #   target: target evidence should exceed clutter evidence
    #   hard clutter: clutter evidence should exceed target evidence
    margin_target = F.relu(margin - (e_t_logit - e_c_logit))
    margin_clutter = F.relu(margin - (e_c_logit - e_t_logit))

    loss_margin_target = _masked_mean(margin_target, target)
    loss_margin_clutter = _masked_mean(margin_clutter, hard_bg)
    loss_margin = loss_margin_target + loss_margin_clutter

    # Final prediction should be low on hard clutter.
    loss_hard_bg_map = F.binary_cross_entropy_with_logits(
        z_final,
        torch.zeros_like(z_final),
        reduction="none",
    )
    loss_hard_bg_final = _masked_mean(loss_hard_bg_map, hard_bg)

    # Ordering constraint:
    # on hard clutter, DEA-calibrated output should not exceed baseline output.
    loss_suppress_order = _masked_mean(
        F.relu(z_final - z_base.detach()),
        hard_bg,
    )

    total = torch.tensor(0.0, device=device)
    total = total + lambda_target_aux * loss_target_aux
    total = total + lambda_ev_target * loss_ev_t
    total = total + lambda_ev_clutter * loss_ev_c
    total = total + lambda_clutter_pred * loss_clutter_pred
    total = total + lambda_suppress_gate * loss_suppress_gate
    total = total + lambda_margin * loss_margin
    total = total + lambda_hard_bg_final * loss_hard_bg_final
    total = total + lambda_suppress_order * loss_suppress_order

    log_vars = {
        "full_dea_loss_target_aux": loss_target_aux.detach(),
        "full_dea_loss_ev_t": loss_ev_t.detach(),
        "full_dea_loss_ev_c": loss_ev_c.detach(),
        "full_dea_loss_clutter_pred": loss_clutter_pred.detach(),
        "full_dea_loss_suppress_gate": loss_suppress_gate.detach(),
        "full_dea_loss_margin": loss_margin.detach(),
        "full_dea_loss_hard_bg_final": loss_hard_bg_final.detach(),
        "full_dea_loss_suppress_order": loss_suppress_order.detach(),

        "hard_bg_ratio": hard_bg.detach().mean(),
        "safe_bg_ratio": safe_bg.detach().mean(),

        "target_evidence_on_gt": (torch.sigmoid(e_t_logit).detach() * target).sum()
        / (target.sum() + 1e-6),
        "clutter_evidence_on_gt": (torch.sigmoid(e_c_logit).detach() * target).sum()
        / (target.sum() + 1e-6),
        "target_evidence_on_hard_bg": (torch.sigmoid(e_t_logit).detach() * hard_bg).sum()
        / (hard_bg.sum() + 1e-6),
        "clutter_evidence_on_hard_bg": (torch.sigmoid(e_c_logit).detach() * hard_bg).sum()
        / (hard_bg.sum() + 1e-6),

        "suppression_on_gt": (torch.sigmoid(suppression_logit).detach() * target).sum()
        / (target.sum() + 1e-6),
        "suppression_on_hard_bg": (torch.sigmoid(suppression_logit).detach() * hard_bg).sum()
        / (hard_bg.sum() + 1e-6),
    }

    return total, log_vars
```

这份 loss 的关键变化是：

```text
不是把所有 safe background 都当 clutter；
而是只把 safe background 中高响应区域当 hard clutter。
```

这样更符合 DEA 的问题定义：

> **要抑制的是 false-alarm-like background evidence，而不是所有背景。**

---

# 5. `main.py` 应该怎么接

当前 `main.py` 只构造：

```python
model = MSHNet(3)
```

训练时只接了 DEA-lite loss。

应该新增一个模式，不要和 DEA-lite 混在一起。

---

## 5.1 import

在 `main.py` 顶部加入：

```python
from model.full_dea_mshnet import FullDEAMSHNet
from model.full_dea_loss import full_dea_aux_loss_v2
```

---

## 5.2 argparse 增加参数

```python
parser.add_argument(
    "--model-type",
    type=str,
    default="mshnet",
    choices=["mshnet", "full_dea"],
)

parser.add_argument("--full-dea-lambda", type=float, default=1.0)
parser.add_argument("--full-dea-ramp-epochs", type=int, default=30)

parser.add_argument("--full-dea-tau-base", type=float, default=0.45)
parser.add_argument("--full-dea-tau-target", type=float, default=0.45)
parser.add_argument("--full-dea-tau-scale", type=float, default=0.45)
parser.add_argument("--full-dea-topk-ratio", type=float, default=0.002)
parser.add_argument("--full-dea-safe-kernel", type=int, default=15)
```

---

## 5.3 模型构造处修改

把：

```python
model = MSHNet(3)
```

改成：

```python
if args.model_type == "full_dea":
    model = FullDEAMSHNet(3)
else:
    model = MSHNet(3)
```

---

## 5.4 train loop 修改

当前训练流程大概是：

```python
if use_dea:
    masks, pred, dea_out = self.model(... return_dea=True ...)
else:
    masks, pred = self.model(data, tag)
```

建议改成：

```python
if self.args.model_type == "full_dea":
    out = self.model(
        data,
        tag,
        return_dict=True,
    )
    masks = out["masks"]
    pred = out["pred"]
    full_dea_out = out["full_dea"]
    dea_out = None
else:
    use_dea = self.use_dea(epoch)
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
    full_dea_out = None
```

原始 segmentation loss 保持：

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

然后加 Full DEA auxiliary loss：

```python
if self.args.model_type == "full_dea" and tag and full_dea_out is not None:
    ramp = get_dea_ramp(epoch, self.warm_epoch, self.args.full_dea_ramp_epochs)
    loss_full_dea, full_dea_log = full_dea_aux_loss_v2(
        full_dea_out=full_dea_out,
        target=labels,
        epoch=epoch,
        warm_epoch=self.warm_epoch,
        seg_criterion=self.loss_fun,
        tau_base=self.args.full_dea_tau_base,
        tau_target=self.args.full_dea_tau_target,
        tau_scale=self.args.full_dea_tau_scale,
        safe_kernel=self.args.full_dea_safe_kernel,
        topk_ratio=self.args.full_dea_topk_ratio,
    )
    loss = loss + self.args.full_dea_lambda * ramp * loss_full_dea
```

DEA-lite 分支可以保留，但不要和 Full DEA 同时开：

```python
if self.args.model_type == "full_dea":
    assert self.args.dea_lambda_single == 0
    assert self.args.dea_lambda_dec == 0
    assert self.args.dea_lambda_empty == 0
```

---

# 6. 推荐训练策略

不要从零直接训 Full DEA。

建议使用：

```text
baseline-preserving two-stage training
```

---

## 6.1 Stage 1：训练 / 加载 MSHNet baseline

先得到一个强 baseline：

```bash
python main.py \
  --model-type mshnet \
  --dataset-dir datasets/NUAA-SIRST \
  --epochs 400 \
  --lr 0.05 \
  --batch-size 4 \
  --mode train
```

保存 best IoU 权重。

---

## 6.2 Stage 2：初始化 Full DEA，并加载 baseline 权重

FullDEAMSHNet 继承 MSHNet，大部分参数名一致。

加载 baseline 时允许 missing keys：

```python
state = torch.load(baseline_path, map_location="cpu")
state_dict = state["net"] if "net" in state else state

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
```

预期 missing keys 应该主要是：

```text
full_dea_head.*
```

这一步非常关键。

因为新 head 初始化为：

```text
target_delta ≈ 0
clutter ≈ negative
suppression_gate ≈ 0
z_final ≈ z_base
```

所以一开始性能应该接近 baseline。

然后 DEA 只需要学习：

```text
在哪里扣掉 clutter
```

而不是重新学习整个 segmentation。

---

## 6.3 Stage 3：先冻结 backbone，只训 DEA head

建议前 20～40 epoch：

```python
for name, p in model.named_parameters():
    if not name.startswith("full_dea_head"):
        p.requires_grad = False
```

学习率可以稍大：

```text
lr = 0.01 或 0.005
```

目标是让 evidence / clutter / suppression gate 先学会基本语义。

---

## 6.4 Stage 4：解冻全网，小学习率 finetune

后面再解冻：

```python
for p in model.parameters():
    p.requires_grad = True
```

学习率降低：

```text
lr = 0.002 ~ 0.005
```

Full DEA loss ramp 继续打开。

---

# 7. 最重要的 ablation 顺序

不要一上来只看最终 IoU。

你需要知道是哪一部分真的有用。

推荐 ablation：

```text
A0: MSHNet baseline

A1: FullDEAHeadV2, but no suppression
    z_final = z_target
    目的：验证多尺度 feature + residual head 是否只是增加容量。

A2: A1 + evidence supervision
    目的：看 target_evidence / clutter_evidence 是否可分。

A3: A2 + clutter branch
    训练 z_clutter，但 final 不 subtract
    目的：看 clutter branch 是否学到 false alarm 区域。

A4: A3 + subtractive suppression
    z_final = z_target - alpha * gate * softplus(z_clutter)
    目的：真正验证 DEA 是否降低 FA。

A5: A4 去掉多尺度 decoder feature，只用 x_d0
    目的：证明 current prototype 的 x_d0-only 结构不够。
```

判断方式：

```text
如果 A4 比 A2/A3 明显降 FA 且 PD 不掉：
    DEA 机制成立。

如果 A1 就大幅提升：
    提升主要来自 capacity，不是 DEA。

如果 A5 接近 A4：
    多尺度 evidence 不是关键。

如果 A5 明显差：
    多尺度 evidence claim 更稳。
```

---

# 8. 必须增加诊断指标

只看 IoU / PD / FA 不够。

每个 epoch 至少打印这些：

```text
target_evidence_on_gt
clutter_evidence_on_gt
target_evidence_on_hard_bg
clutter_evidence_on_hard_bg
suppression_on_gt
suppression_on_hard_bg
hard_bg_ratio
alpha
```

理想现象：

```text
target_evidence_on_gt          高
clutter_evidence_on_gt         低
target_evidence_on_hard_bg     低
clutter_evidence_on_hard_bg    高
suppression_on_gt              低
suppression_on_hard_bg         高
```

如果不是这样，说明 Full DEA 没有学到：

```text
目标证据 / 杂波证据分解
```

而只是一个普通 attention module。

---

# 9. 进一步冲 SOTA：建议加 offline pseudo clutter

online hard_bg 有一个缺点：

```text
早期模型不稳定，hard background 会抖动
```

更强的做法是：

```text
先用强 MSHNet baseline 在训练集上跑 prediction；
把 GT dilated 区域去掉；
剩下的高响应 connected components 作为 pseudo clutter mask；
训练 Full DEA 时同时使用 offline clutter mask + online hard_bg。
```

伪代码：

```python
pseudo_clutter = safe_bg * (sigmoid(z_baseline) > 0.4)
pseudo_clutter = remove_small_noise_or_keep_top_components(pseudo_clutter)
hard_bg = max(pseudo_clutter, online_hard_bg)
```

这一步很可能比继续调 `lambda_single` 更有价值。

---

# 10. 对“哪个结构有问题”的最终判断

按严重程度排序如下。

---

## 10.1 第一严重：`y_final = gate * y_real + (1-gate) * y_cf`

这是最应该改的。

如果 `y_cf` 是 counterfactual clutter branch，它不应该被混回 final。

应该改成：

```python
z_final = z_target - alpha * suppression_gate * F.softplus(z_clutter)
```

---

## 10.2 第二严重：Full DEA 只插在 `x_d0`，没有真正使用多尺度 evidence fusion

DEA-lite 证明的问题来自：

```text
scale_logits -> final
```

Full DEA 应该使用：

```text
x_d0, x_d1, x_d2, x_d3
scale_logits_full
z_base
```

而不是只吃 `x_d0`。

---

## 10.3 第三严重：clutter evidence 没有监督

不能只靠：

```python
sep_loss = target_evidence * clutter_evidence
```

需要 hard false-alarm pseudo-label：

```text
GT target:
    target evidence = 1
    clutter evidence = 0

hard false alarm background:
    target evidence = 0
    clutter evidence = 1
```

---

## 10.4 第四严重：FullDEAMSHNet 返回的 scale masks 破坏原多尺度 loss contract

`masks` 必须是原尺度：

```python
[mask0, mask1, mask2, mask3]
```

不能是全部 upsample 后的 full-resolution masks。

---

# 11. 推荐下一次实验

先选 NUAA，因为 DEA-lite 在 NUAA 失败。

如果新 DEA 能在 NUAA 上恢复甚至超过 baseline，才说明它真的解决了 DEA-lite 的失败模式。

基础命令：

```bash
python main.py \
  --model-type full_dea \
  --dataset-dir datasets/NUAA-SIRST \
  --mode train \
  --epochs 500 \
  --warm-epoch 50 \
  --batch-size 4 \
  --lr 0.005 \
  --full-dea-lambda 1.0 \
  --full-dea-ramp-epochs 30 \
  --full-dea-tau-base 0.45 \
  --full-dea-tau-target 0.45 \
  --full-dea-tau-scale 0.45 \
  --full-dea-topk-ratio 0.002
```

但更推荐：

```text
先加载 NUAA baseline best checkpoint；
前 20~40 epoch 冻结 backbone，只训 full_dea_head；
再解冻全网 finetune。
```

---

# 12. DEA 真正应该解决什么？

不要把 DEA 写成：

```text
提高 IoU 的新 head
```

而应该写成：

```text
MSHNet 这类多尺度小目标检测器中，某些背景结构会在单尺度或多尺度 fusion 中形成伪目标证据。

DEA 显式分解 target evidence 和 clutter evidence，并通过 counterfactual clutter suppression 在推理时扣除杂波证据，从而降低 FA，同时保持 target response。
```

所以代码上必须满足三个条件：

```text
1. evidence 分解发生在多尺度 fusion 处；
2. clutter evidence 有 hard false-alarm 监督；
3. counterfactual branch 作为 suppression 使用，而不是作为 alternative prediction 混回 final。
```

当前 Full DEA prototype 主要缺：

```text
第 1 点：evidence 分解没有发生在多尺度 fusion 处
第 3 点：counterfactual branch 被混回 final
```

loss 主要缺：

```text
第 2 点：clutter evidence 没有 hard false-alarm 监督
```

因此，下一步优先按 Full DEA v2 改，而不是继续围绕 DEA-lite 调参。

---

# 13. 最终建议

短期内不要再把主要精力放在 DEA-lite 上。

DEA-lite 的价值是：

```text
前验证
诊断工具
ablation evidence
说明 single-scale false alarm suppression 在部分数据集有效
暴露 loss-only regularization 的局限
```

真正下一步应该是：

```text
Full DEA v2
= baseline-preserving residual target branch
+ explicit target / clutter evidence decomposition
+ hard false-alarm pseudo clutter supervision
+ subtractive counterfactual clutter suppression
+ 多尺度 evidence fusion 处介入
```

如果这个版本在 NUAA 上能恢复 baseline，并在 NUDT / IRSTD-1K 上进一步降低 FA 或提升 IoU，那么 DEA 的问题定义和结构设计才真正成立。
