# JIP-MSHNet：面向 AAAI-27 的 MSHNet 单原理结构改造方案

> **状态更正（2026-07-13）**：本文是仓库历史 JIP 候选设计，不再是当前冻结主模型，也没有形成可投稿的稳定增益证据。当前 TRACE 方向先执行状态空间/数据合同硬门槛；审计已经得到 `NO-GO`，详见 [`TRACE_MSHNet_STAGE0_audit.md`](TRACE_MSHNet_STAGE0_audit.md)。在该门槛解除前，JIP 与 TRACE 都不得写成“最终模型”或“已超过 MSHNet”。保留本文是为了审计研究决策，不代表继续堆叠或默认 JIP 正确。
>
> **论文主张候选**：A Peak Should Not Vote for Itself: Jackknife Influence Pooling for Infrared Small Target Detection
> **中文主张**：峰值不应给自己投票——用 Jackknife 自影响消除纠正红外小目标下采样中的“自证偏差”
> **仓库重新抓取版本**：`Arialliy/dea3090@43c8c8367c21b64cae9e719868aaccda5cc6d329`
> **基线**：参数等价的 deterministic MSHNet，保持 canonical forward、SLS loss、多尺度侧输出与 `3×3 Conv(4→1)` 融合不变
> **当前状态**：`HISTORICAL CANDIDATE`；`CODE-LEVEL TESTS ONLY`；`NO EMPIRICAL DESIGN PASS`
> **禁止误写**：截至本文生成时，JIP-MSHNet 尚未完成数据集训练，不能写成“已经超过 MSHNet”
> **AAAI-27 截止**：摘要 2026-07-21 23:59 UTC-12；全文 2026-07-28 23:59 UTC-12。换算为台北时间分别是 2026-07-22 19:59 与 2026-07-29 19:59，但项目内部仍按 7 月 21 日前完成摘要冻结。

---

## 0. 历史 JIP 决策（不再是当前最终决策）

这份历史方案当时决定不再从仓库中已经失败的 SDRR、OSO、DSF、DCDF、CCFD、SPT0 或尚未正式接入的 SIED 中改名拼装，也不再新增注意力、频域分支、边缘分支、动态卷积、门控器、辅助头或新损失。

**当时的唯一主模型候选为：**


**JIP-MSHNet（Jackknife Influence Pooling MSHNet）**。

它只做一件事：把 MSHNet 编码器中的原生 `2×2 MaxPool`，从输入端开始按连续前缀替换为同一个、零参数的 JIP 下采样方程。最小完整模型是只替换第一处池化的 `JIP-1`；只有当前一前缀通过预注册实验门槛后，才把完全相同的方程延伸到下一处池化。已经通过的前段不改公式、不调系数、不换模块。

### 0.1 研究目标应当这样写

> **研究目标**：在不改变 MSHNet 的编码卷积、残差注意力块、解码器、多尺度预测头、最终融合层和 SLS 训练目标，且不增加任何可训练参数的前提下，纠正 `2×2` 最大池化中“单通道峰值参与构造跨通道支持、再反过来验证自身”的自确认偏差；仅从最大值到次大值之间的独占证据中，扣除该峰值对其自身支持所造成的精确 Jackknife 影响，使跨通道一致的小目标峰值保持原样，而孤立热杂波的自证幅值受到通道数自适应、严格有界的抑制。

### 0.2 核心研究问题

> **一个局部峰值在跨越分辨率边界时，究竟是被其他特征通道共同支持，还是仅仅因为它自己参与了“共识”统计而完成了自证？**

这个问题直接落在 MSHNet 的原生下采样结构上，而不是在网络后面再叠加补救模块。

### 0.3 完整模型的确定规则

设四个原生池化边界为 `0,1,2,3`：

- `JIP-1`：只替换 boundary 0；这是最小完整投稿模型。
- `JIP-2`：在 `JIP-1` 已通过后，保持 boundary 0 完全不动，再将同一方程用于 boundary 1。
- `JIP-3`：在 `JIP-2` 已通过后，继续扩展至 boundary 2。
- `JIP-4`：继续扩展至 boundary 3。

最终模型是**从输入端开始、连续通过验收的最深前缀**。后一边界失败，只撤销新加入的后一边界；不回头调前段，不添加救援模块。这样“逐段设计”不会退化为模块堆叠。

---

## 1. 重新抓取仓库后的真实状态

### 1.1 最新权威状态与上一版方案不同

最新提交 `43c8c83` 把仓库明确定位为 MSHNet 的结构反事实研究工作台，并将历史候选保留为可审计负结果。README 的权威状态写明：

- SDRR、OSO、DSF、DCDF、CCFD 是机制研究或被拒绝的比较器；
- SPT0 已完成 400 epoch 配对实验，虽然降低 FA，但 IoU 与 PD 同时下降，状态为 `DESIGN FAIL`；
- 当前仓库没有任何结构方法达到 `DESIGN PASS`。

现有正式结果为：

| 结构 | best epoch | IoU | PD | FA/Mpixel | 结论 |
|---|---:|---:|---:|---:|---|
| Deterministic MSHNet | 379 | **0.728026** | **0.946768** | 22.1039 | canonical baseline |
| CCFD | 379 | 0.725052 | 0.950570 | 18.1822 | IoU gate 失败 |
| SPT0 | 399 | 0.716904 | 0.931559 | **11.9076** | IoU、PD 均失败 |

因此，上一版把 SDRR/SDRR refinement 写成“最终冻结模型”是不成立的。本方案从最新提交重新起算。

### 1.2 canonical MSHNet 的前向结构

MSHNet 的核心结构非常清晰：

1. `1×1 conv_init` 将输入映射到 16 通道；
2. 四个编码阶段，通道为 `16→32→64→128→256`，阶段之间使用四次 `2×2 MaxPool`；
3. 每个残差块包含 identity path、Channel Attention 和 Spatial Attention；
4. U-Net 解码器依次拼接对应 encoder skip；
5. `decoder0` 重新接收 full-resolution `e0`；
6. 四个 side heads 分别产生不同尺度的单通道 logit；
7. side logits 上采样到原分辨率后，由 `3×3 Conv(4→1)` 融合；
8. 训练保持原始 SLSIoU + location term。

确定性基线只把 Channel Attention 中的 adaptive max reduction backward 改为 `torch.amax`，参数、state dict 和 forward 数值语义保持一致。

### 1.3 仓库已有方向：全部与新模型物理隔离

| 已有方向 | 主要位置/思想 | 当前处置 | JIP 与它的区别 |
|---|---|---|---|
| SDR / SDRR | final linear fusion 的尺度删除责任 | 不作为提案 | JIP 在 encoder 下采样前端，不使用 final-scale deletion loss |
| OSO | 尺度所有权正交化 | 比较器 | JIP 不分配 side-head ownership |
| DSF / DCDF | 删除稳定或决策条件融合 | 比较器 | JIP 不修改最终融合 |
| CCFD | 冲突条件扩散式融合 | `DESIGN FAIL` | JIP 保持 final `3×3 Conv(4→1)` 原样 |
| SPT0 | 第一池化边界的软支持持续性 | `DESIGN FAIL` | JIP 删除精确自影响，删除上界仅 `1/C`，不是最高 50% |
| Soft LOCI helper | softmax ownership 的 leave-one-channel influence | 只作 ablation | JIP 使用无温度、无 epsilon、并列感知的硬秩 ownership，并形成正式前缀架构 |
| Counterfactual self-support helper | 自支持比例式删除 | 只作过强对照 | JIP 使用绝对有限样本 Jackknife 影响，天然受 `1/C` 约束 |
| SIED prototype | 同一 decoder 的四次 anchored coalition / Möbius 分解 | 未接入正式 variant | JIP 不改 decoder、不增加四倍 decoder 调用 |
| 各类 deep-supervision / loss 变体 | side loss、gradient projection、coalition supervision 等 | 不进入主模型 | JIP 保持 `legacy_exact` 和 canonical SLS loss |

主论文中不得把上述任何方向与 JIP 组合成 “JIP + CCFD + 新 loss”。这样会立刻变成模块堆叠，也无法归因。

---

## 2. MSHNet 中最值得先改的结构问题

### 2.1 逐层组件轨迹把问题压缩到两个断点

仓库在 baseline epoch 379 checkpoint 上，对 matched prediction、false component、matched target 和 missed target 回投到各层，得到：

| stage | matched prediction > false AUC | matched target > missed AUC |
|---|---:|---:|
| stem | 0.512 | 0.680 |
| encoder0 | 0.581 | 0.638 |
| encoder1 | 0.700 | 0.759 |
| encoder2 | **0.737** | 0.818 |
| encoder3 | 0.702 | 0.818 |
| middle | **0.562** | 0.762 |
| decoder3 | 0.678 | 0.795 |
| decoder2 | 0.804 | 0.843 |
| decoder1 | **0.842** | **0.870** |
| decoder0 | 0.710 | 0.846 |

这说明：

- encoder2 已形成较强的目标/杂波区分，但在后续 pooling 与 middle 中明显丢失；
- decoder1 已恢复最佳区分，但 decoder0 拼接 full-resolution `e0` 后又下降。

按“由前往后”的规则，必须先处理前一个断点：**下采样边界如何传输局部峰值证据**。不能跳过它直接在 decoder0 再加一个新门控器。

### 2.2 多尺度头不是当前首要瓶颈

冻结 checkpoint 的 16 个原生 side-scale subset 审计显示，只使用 `s0+s1` 得到：

- IoU：`0.729992`，高于全四尺度 `0.728026`；
- PD：相同；
- FA：`21.6760`，低于 `22.1039`。

这说明“再加更多尺度”不是答案；并且 final fusion 方向已经被仓库大量实验覆盖。当前更合理的问题是：**进入深层之前，证据是怎样在分辨率边界上被保留或污染的。**

### 2.3 固定反事实删除给出最关键的定量线索

对第一处 `2×2 MaxPool`，记最大值为 `m1`，删除该最大位点后的反事实最大值为 `m2`，冻结 baseline 权重并使用

\[
P_\alpha=(1-\alpha)m_1+\alpha m_2
\]

得到：

| α | IoU | PD | FA/Mpixel | false components |
|---:|---:|---:|---:|---:|
| 0 | 0.728026 | 0.946768 | 22.1039 | 15 |
| 0.02 | 0.728441 | 0.946768 | 21.7473 | 15 |
| 0.05 | **0.728509** | 0.946768 | 21.5334 | 15 |
| 0.10 | 0.727311 | 0.946768 | 20.6065 | 15 |
| 0.40 | 0.726167 | 0.946768 | 18.5387 | **12** |
| 1.00 | 0.709180 | 0.935361 | 16.1857 | 16 |

可复现结论不是“删得越多越好”，而是：

1. 最大位点独占证据中确实混入可删除杂波；
2. 有利区间很小，约为 2%–5%；
3. 10% 以上开始伤害 IoU；
4. 强删除虽然可降像素 FA，却可能降低 PD 或增加组件数；
5. 因而需要一个**无超参数、局部自适应、天然落在小删除区间**的机制。

### 2.4 SPT0 为什么失败

SPT0 使用软空间 ownership 与包含自身的跨通道共识，并令

\[
g=\frac{1+\kappa}{2}\in[0.5,1],\qquad
P=m_2+g(m_1-m_2).
\]

当支持最弱时，它会删除最大值与次大值之间 **50%** 的独占证据。这个幅度远高于冻结审计中约 2%–5% 的有利区间。其正式训练结果确实表现为 FA 大幅降低，但 IoU 和 PD 同时损失。

更根本的问题是：用于验证某通道峰值的共识 `\bar q` 包含该通道自己的 vote。一个孤立峰值会部分制造自己的“跨通道支持”。JIP 直接消除这一循环，而不是再设计一个更复杂的 gate。

---

## 3. 新模型：Jackknife Influence Pooling

### 3.1 2×2 局部表示

对一个 batch 中某个 `2×2` cell，令

\[
X\in\mathbb{R}^{C\times 4},
\]

其中 `C` 是该 MSHNet stage 的通道数，四列对应 cell 的四个空间位置。对通道 `c`：

\[
m_{1,c}=\max_s X_{c,s},\qquad
m_{2,c}=\operatorname{2ndmax}_s X_{c,s},
\]

\[
g_c=m_{1,c}-m_{2,c}\ge 0.
\]

`g_c` 不是任意 top-k 混合，而是一个明确的结构反事实量：删除最强位点后，最大池化输出会减少多少。

### 3.2 并列感知的秩 ownership

定义通道 `c` 对四个位置的最大位点所有权：

\[
q_{c,s}=\frac{\mathbf{1}[X_{c,s}=m_{1,c}]}
{\sum_{t=1}^{4}\mathbf{1}[X_{c,t}=m_{1,c}]}.
\]

- 唯一最大值时，`q_c` 是 one-hot；
- 多个位置并列最大时，vote 在并列位置上均分；
- 不使用 z-score、softmax、temperature 或 epsilon；
- ownership 作为 stop-gradient 统计，不引入可学习分支。

### 3.3 含自身与排除自身的支持总体

包含当前通道的总体 vote：

\[
\bar q=\frac{1}{C}\sum_{d=1}^{C}q_d.
\]

排除当前通道后的 peer vote：

\[
\bar q_{-c}=\frac{1}{C-1}\sum_{d\ne c}q_d.
\]

当前峰值与两种总体的 agreement 为：

\[
a_c^{+}=q_c^\top\bar q,
\qquad
a_c^{-}=q_c^\top\bar q_{-c}.
\]

### 3.4 精确 Jackknife 自影响

定义峰值对自身“群体支持”的有限样本影响：

\[
\iota_c=[a_c^{+}-a_c^{-}]_{+}.
\]

代数化简得：

\[
\iota_c=\frac{1}{C}
\left[\lVert q_c\rVert_2^2-q_c^\top\bar q_{-c}\right]_{+}.
\]

它回答一个非常具体的问题：

> 把通道 `c` 从支持总体中删掉以后，通道 `c` 的最大位点看起来少了多少支持？

这不是学习到的注意力权重，也不是人为设置的删除比例。

### 3.5 JIP 输出

JIP 只从最大值的独占 gap 中扣除精确自影响：

\[
\boxed{
\operatorname{JIP}(X)_c
=m_{1,c}-\operatorname{sg}(\iota_c)(m_{1,c}-m_{2,c})
}
\]

等价地：

\[
\operatorname{JIP}(X)_c
=m_{2,c}+(1-\operatorname{sg}(\iota_c))g_c.
\]

`sg` 表示 stop-gradient。梯度仍通过 `m1` 和 `m2` 的 order-statistic 路径传播，但不通过支持统计构造一条可被网络直接推回 identity 的连续门控路径。

---

## 4. 为什么这条方程不是拍脑袋调参

### 4.1 严格有界

因为 `q_c` 是概率向量，且自影响由一个样本对总体平均的贡献产生：

\[
0\le\iota_c\le\frac{1}{C}.
\]

因此：

\[
m_{2,c}\le \operatorname{JIP}(X)_c\le m_{1,c}.
\]

JIP 永远不会产生超出最大值/次大值区间的新激活，也不会反向增强峰值。

### 4.2 唯一最大值下的直观形式

若 `q_c` 为 one-hot，令 `\pi_c` 是其余 `C-1` 个通道中，最大位点与通道 `c` 相同的比例，则：

\[
\boxed{\iota_c=\frac{1-\pi_c}{C}}.
\]

于是：

- peers 全部在同一位置达到最大：`π=1`，`ι=0`，严格等于 MaxPool；
- 没有 peer 在同一位置达到最大：`π=0`，`ι=1/C`，只删除 gap 的 `1/C`；
- 支持越一致，删除越小；
- 一个通道不能靠自己的 vote 让 `π` 增大。

### 4.3 零假设下自动落入仓库观察到的有利区间

若背景杂波下各通道最大位点在四个位置上近似独立均匀，则

\[
\mathbb{E}[\pi_c]=\frac14,
\qquad
\mathbb{E}[\iota_c]=\frac{3}{4C}.
\]

对 canonical MSHNet：

| boundary | 输入通道 C | 最大删除 `1/C` | 随机位点零假设期望删除 `3/(4C)` |
|---:|---:|---:|---:|
| 0 | 16 | 6.2500% | **4.6875%** |
| 1 | 32 | 3.1250% | 2.3438% |
| 2 | 64 | 1.5625% | 1.1719% |
| 3 | 128 | 0.7813% | 0.5859% |

第一边界的零假设期望删除 **4.6875%**，几乎正好落在仓库冻结审计最优的 `α≈0.05` 附近。这不是把 0.05 手工写回模型，而是由 `2×2` 的四个位置和 stage0 的 16 个通道自动推导出来。

同时，随着通道数增加，修正自然衰减：早期高分辨率处更积极地纠正孤立峰值，深层语义特征处趋近原始 MaxPool。无需 stage-specific coefficient。

### 4.4 对真实小目标与热杂波的可证伪假设

JIP 的方法假设不是“目标一定亮、杂波一定暗”，而是：

- 经过 `conv_init + encoder0` 后，真实小目标更可能让多个通道在同一 `2×2` 位置形成一致极值；
- 孤立热噪点或纹理尖峰更容易只在少数通道形成位置独占极值；
- 因而 target cell 的 `π` 应高于 matched-background/false-component cell，`ι` 应更低。

这必须通过机制统计验证。若 target 与 false component 的 `π/ι` 分布没有显著差异，论文主张即被否证，不能靠加模块掩盖。

### 4.5 结构性质

JIP 具有以下可测试性质：

1. **参数与 buffer 为 0**；
2. **通道置换等变**；
3. 对每个 cell 四个位置的共同置换保持输出不变；
4. 对 `a>0` 的正仿射变换满足 `JIP(aX+b)=aJIP(X)+b`；
5. 并列最大值时 `m1-m2=0`，输出严格保持最大值；
6. perfect peer support 时严格退化为 MaxPool；
7. 输出被 `[m2,m1]` 严格夹住；
8. correction 上界随通道数自动衰减；
9. state dict 与 canonical MSHNet 严格兼容。

---

## 5. 为什么不是“模块堆叠”

JIP-MSHNet 不采用“backbone + attention + frequency + edge + dynamic fusion + new loss”的常见堆法。它满足以下物理约束：

- 只定义一个下采样算子；
- 只替换原模型本来就存在的 MaxPool，不新增并行支路；
- 不新增卷积、MLP、attention、router、head 或监督项；
- 不改变 decoder、skip concat、side heads、final fusion 与 loss；
- `JIP-2/3/4` 只是同一方程沿原生连续边界的扩展，不是多个异构模块；
- primary comparison 只允许 `MSHNet vs JIP-MSHNet`；历史结构只能单独作为对照。

论文图应画成一条主干：把四个 MaxPool 位置标出，最终被选择的连续前缀用同一个 `JIP` 标识。不要画多个花哨子模块框。

---

## 6. 与近期方法的明确区分

| 方法 | 核心方向 | 与 JIP-MSHNet 的边界 |
|---|---|---|
| MSHNet, CVPR 2024 | SLS loss + 简单 multi-scale head | JIP 保持 SLS 与 multi-scale head；只研究其下采样自证偏差 |
| PConv-SD, AAAI 2025 | Pinwheel-shaped convolution + scale-based dynamic loss | JIP 不改卷积形状、不做 scale-conditioned loss |
| LoMix, NeurIPS 2025 | 学习多尺度 decoder logits 的组合与训练权重 | JIP 不混合新 logits、不做 NAS/权重搜索，作用于 encoder pooling |
| DEFANet, AAAI 2026 | edge-target dual path、frequency enhancement、edge-guided integration | JIP 没有双分支、边缘标签、DCT 或融合模块 |
| NS-FPN, CVPR 2026 | low-frequency guided purification + spiral feature sampling | JIP 不构建 FPN 插件，不做频域纯化或螺旋采样 |
| InvDet, CVPR 2026 | invertible encoder + reconstruction guidance/TARM/GCTM | JIP 不增加逆路径或重建目标，只做有界局部下采样 |
| DPP, CVPR 2018 | 学习 detail-preserving pooling | JIP 无学习权重，目标是删除“自身 vote 造成的支持污染” |
| LIP, ICCV 2019 | 子网络预测局部重要性权重 | JIP 无 importance predictor |
| SoftPool, ICCV 2021 | 按激活幅值的 softmax 权重池化 | JIP 只使用秩位置和 leave-one-channel 影响，不按幅值 softmax |
| APS, CVPR 2021 / TIPS, WACV 2025 | polyphase 选择与 shift invariance | JIP 不以平移不变性为目标，不选择全通道统一 polyphase |
| polynomial / ordinal pooling | 学习或设置 order-statistic 权重 | JIP 的 `m2` 只定义删除最大位点的反事实，系数来自精确 Jackknife 影响 |

### 6.1 可使用的新颖性表述

> Existing adaptive pooling methods learn importance, rank, or polyphase selection rules. JIP instead identifies a finite-sample self-confirmation bias: a channel participates in the spatial support population that is then used to validate its own maximum. We remove only the exact leave-one-channel influence of this self vote from the max-to-second-max counterfactual gap.

### 6.2 不可使用的过度表述

不能直接写：

- “first adaptive pooling method”；
- “first leave-one-out pooling”——仓库自身已有 soft helper，外部也可能存在相近思想；
- “causal responsibility learning”；
- “guaranteed to preserve targets”；
- “zero computational overhead”；
- “already achieves SOTA”。

更稳妥的表述是：

> 在本轮检索的 IRSTD 与神经网络 pooling 文献中，未发现与“排除当前通道自身 vote、计算其对最大位点支持的精确有限样本影响，再以该影响有界修正 max-to-second gap”完全相同的方法；投稿前仍需进行最终 Scholar/DBLP/Google Patents 排重。

---

## 7. 可直接落地的实现

### 7.1 `model/jackknife_influence_pool.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JackknifeInfluencePool2d(nn.Module):
    """Parameter-free, tie-aware 2x2 jackknife influence pooling."""

    def forward(
        self,
        x: Tensor,
        *,
        return_state: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 4:
            raise ValueError("x must be a BCHW tensor")
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            raise ValueError("H and W must be divisible by 2")
        if x.shape[1] < 2:
            raise ValueError("JIP requires at least two channels")

        b, c, h, w = x.shape
        # [B, 4C, H/2, W/2] -> [B, C, 4, H/2, W/2]
        cells = F.pixel_unshuffle(x, 2).view(b, c, 4, h // 2, w // 2)

        top2 = torch.topk(cells, k=2, dim=2, sorted=True).values
        maximum = top2[:, :, 0]
        second = top2[:, :, 1]
        exclusive_gap = maximum - second

        # The support statistic is deliberately stop-gradient.
        with torch.no_grad():
            ownership = (cells == maximum.unsqueeze(2)).to(torch.float32)
            ownership = ownership / ownership.sum(
                dim=2, keepdim=True
            ).clamp_min(1.0)

            peer_population = (
                ownership.sum(dim=1, keepdim=True) - ownership
            ) / float(c - 1)
            population = ownership.mean(dim=1, keepdim=True)

            agreement_with_self = (ownership * population).sum(dim=2)
            agreement_without_self = (ownership * peer_population).sum(dim=2)
            self_influence = (
                agreement_with_self - agreement_without_self
            ).clamp(0.0, 1.0 / float(c))
            self_influence = self_influence.to(dtype=x.dtype)

        output = maximum - self_influence * exclusive_gap
        if not return_state:
            return output

        return output, {
            "maximum": maximum,
            "second": second,
            "exclusive_gap": exclusive_gap,
            "agreement_with_self": agreement_with_self,
            "agreement_without_self": agreement_without_self,
            "self_influence": self_influence,
            "survival": 1.0 - self_influence,
        }
```

### 7.2 `model/jip_mshnet.py`

```python
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from model.baselines.mshnet_deterministic import MSHNet as DeterministicMSHNet
from model.jackknife_influence_pool import JackknifeInfluencePool2d


class JIPMSHNet(DeterministicMSHNet):
    """Canonical deterministic MSHNet with JIP on a continuous pool prefix."""

    def __init__(
        self,
        input_channels: int,
        *,
        active_stages: Sequence[int] = (0,),
    ) -> None:
        super().__init__(input_channels)
        stages = tuple(int(stage) for stage in active_stages)
        if len(stages) != len(set(stages)):
            raise ValueError("active_stages must be unique")
        if any(stage not in (0, 1, 2, 3) for stage in stages):
            raise ValueError("active_stages must be drawn from 0,1,2,3")
        if stages and stages != tuple(range(max(stages) + 1)):
            raise ValueError("JIP stages must form a prefix: (0,), (0,1), ...")
        self.active_stages = stages
        self.jip_pool = JackknifeInfluencePool2d()

    def _pool(self, x: Tensor, stage: int) -> Tensor:
        if stage in self.active_stages:
            return self.jip_pool(x)
        return self.pool(x)

    def forward(self, x: Tensor, warm_flag: bool):
        e0 = self.encoder_0(self.conv_init(x))
        e1 = self.encoder_1(self._pool(e0, 0))
        e2 = self.encoder_2(self._pool(e1, 1))
        e3 = self.encoder_3(self._pool(e2, 2))
        middle = self.middle_layer(self._pool(e3, 3))

        d3 = self.decoder_3(torch.cat([e3, self.up(middle)], dim=1))
        d2 = self.decoder_2(torch.cat([e2, self.up(d3)], dim=1))
        d1 = self.decoder_1(torch.cat([e1, self.up(d2)], dim=1))
        d0 = self.decoder_0(torch.cat([e0, self.up(d1)], dim=1))

        if not warm_flag:
            return [], self.output_0(d0)

        masks = [
            self.output_0(d0),
            self.output_1(d1),
            self.output_2(d2),
            self.output_3(d3),
        ]
        output = self.final(
            torch.cat(
                [
                    masks[0],
                    self.up(masks[1]),
                    self.up_4(masks[2]),
                    self.up_8(masks[3]),
                ],
                dim=1,
            )
        )
        return masks, output


JIP_VARIANTS = {
    "jip1": (0,),
    "jip2": (0, 1),
    "jip3": (0, 1, 2),
    "jip4": (0, 1, 2, 3),
}
```

### 7.3 `main.py` 的最小接入

只做以下几处物理修改，不触碰 loss 与训练循环：

```python
from model.jip_mshnet import JIPMSHNet, JIP_VARIANTS
```

将 variant choices 扩为：

```python
choices=[
    'workbench', 'official', 'deterministic',
    'sdr', 'oso', 'dsf', 'dcdf', 'ccfd', 'spt0',
    'jip1', 'jip2', 'jip3', 'jip4',
]
```

在模型构造分支中：

```python
if args.mshnet_variant in JIP_VARIANTS:
    model = JIPMSHNet(
        args.input_channels,
        active_stages=JIP_VARIANTS[args.mshnet_variant],
    )
```

在 method name 中：

```python
if args.model_type == 'mshnet' and args.mshnet_variant in JIP_VARIANTS:
    return f"JIP{len(JIP_VARIANTS[args.mshnet_variant])}-MSHNet"
```

metadata 必须保存：

```python
"jip_active_stages": list(JIP_VARIANTS.get(args.mshnet_variant, ())),
"jip_statistic_detached": True,
"jip_formula_version": "hard_tie_aware_absolute_jackknife_v1",
```

### 7.4 不允许增加的 CLI 超参数

JIP 主模型不得添加：

- `--jip-alpha`；
- `--jip-temperature`；
- `--jip-eps`；
- `--jip-learnable`；
- `--jip-ramp`；
- `--jip-gate-bias`。

唯一结构选择是 `jip1/jip2/jip3/jip4`，且按预注册前缀门槛确定一次。

---

## 8. 已完成的本地代码级验证

在当前环境中已使用 canonical MSHNet 源码构造原型并运行以下测试：

```text
{
  'tests_passed': True,
  'parameter_count': 0,
  'buffer_count': 0,
  'max_observed_influence_random_C16': 0.0625,
  'theoretical_max_influence_C16': 0.0625,
  'affine_max_abs_error': 8.881784197001252e-16,
  'channel_permutation_max_abs_error': 0.0,
  'cell_site_permutation_max_abs_error': 0.0,
  'worst_case_four_boundary_survival': 0.8870279788970947
}
```

模型包装测试结果：

```text
{
  'strict_state_dict_compatible': True,
  'baseline_parameters': 4065513,
  'jip_parameters': 4065513,
  'parameter_delta': 0,
  'mask_shapes': [
      (1, 1, 32, 32),
      (1, 1, 16, 16),
      (1, 1, 8, 8),
      (1, 1, 4, 4)
  ],
  'output_shape': (1, 1, 32, 32),
  'active_stages': (0,)
}
```

已验证项目：

- strict state-dict load；
- 参数量完全相同；
- side-output 与 final-output 尺寸一致；
- 输出位于 `[m2,m1]`；
- unanimous support 时严格等于 MaxPool；
- 最大自影响严格为 `1/C`；
- tie case 前向稳定；
- 正仿射等变；
- 通道置换等变；
- cell 位点共同置换不变；
- finite、non-zero backward；
- 零参数、零 buffer。

**这只证明方程与代码实现成立，不等于模型在数据集上已经成功。** 公共仓库没有随代码提供本地 `datasets/` 与正式 `weight/`，当前环境不能诚实地伪造 400 epoch 训练结果。

---

## 9. 必须新增的单元与审计测试

### 9.1 单元测试

建议新增 `tests/test_jackknife_influence_pool.py`：

1. `test_shape_and_bounds`；
2. `test_unanimous_support_is_exact_maxpool`；
3. `test_unsupported_unique_peak_is_exact_one_over_c`；
4. `test_tie_forward_is_exact_maximum`；
5. `test_positive_affine_equivariance`；
6. `test_channel_permutation_equivariance`；
7. `test_cell_site_permutation_invariance`；
8. `test_finite_backward_cpu`；
9. `test_deterministic_cuda_backward`；
10. `test_zero_parameters_and_buffers`。

### 9.2 模型身份测试

建议新增 `tests/test_jip_mshnet.py`：

- `jip1` 与 deterministic baseline state-dict keys 完全相同；
- strict load 无 missing/unexpected keys；
- 参数数与 trainable 参数数相同；
- `active_stages=()` 时 forward 与 deterministic baseline 数值完全一致；
- `jip1` 只有 boundary0 输出发生变化；
- `jip2` 相比 `jip1` 只新增 boundary1；
- warm path 与 non-warm path 都可前后向；
- AMP fp16/bf16 finite；
- 目标 GPU 上 `torch.use_deterministic_algorithms(True)` 不报错。

### 9.3 机制审计

新增 `tools/audit_jip_mechanism.py`，在 target、matched prediction、false component、missed target、safe background 五类区域记录：

- `π`：peer same-site proportion；
- `ι`：self influence；
- `m1-m2`：exclusive gap；
- 实际删除量 `ι(m1-m2)`；
- JIP 前后 feature energy；
- boundary0 到 middle 的 matched-vs-false AUC；
- target cell 与 false-component cell 的 `ι` 分布和 effect size。

主张成立的必要机制信号：

\[
\operatorname{median}(\iota_{false})>
\operatorname{median}(\iota_{target}),
\]

且差异不能只由 `m1` 幅值造成。

---

## 10. 从前往后的冻结式开发流程

### 10.1 冻结规则

每一阶段只允许三种状态：

- `PENDING`：尚未完成预注册测试；
- `DESIGN PASS / FROZEN`：所有门槛通过，公式与已有 active boundary 永久冻结；
- `DESIGN FAIL / REVERT NEWEST BOUNDARY`：只撤回本阶段新增边界，保留上一个通过前缀。

不得出现“失败后把 alpha 从 0.05 调成 0.03”“再加一个 attention 救回来”“换 loss 补指标”等行为。

### 10.2 阶段表

| 阶段 | 唯一允许的改动 | 当前状态 | 通过后冻结内容 |
|---|---|---|---|
| A | JIP 方程、代码、性质测试 | **CODE PASS** | 方程 v1，不再改定义 |
| B | boundary0：MaxPool → JIP | 待服务器实验 | `JIP-1` 完整模型 |
| C | 保持 B 不动，只扩展 boundary1 | 锁定，需 B 先通过 | `JIP-2` |
| D | 保持 B/C 不动，只扩展 boundary2 | 锁定 | `JIP-3` |
| E | 保持 B/C/D 不动，只扩展 boundary3 | 锁定 | `JIP-4` |
| F | 选定最深通过前缀，跨数据集确认 | 待前缀确定 | 最终投稿模型 |

### 10.3 最小完整模型原则

`JIP-1` 一旦通过就是完整模型，不需要等待后面再叠东西。后续前缀只是检验同一自影响原理在更深分辨率边界上是否仍有益。

这保证：

- 后段失败不会让前面已经成功的模型失效；
- 论文始终只有一个核心机制；
- 截止日期前可以先冻结 `JIP-1`；
- 不会因为追求更深前缀而无限迭代。

---

## 11. 实验协议：先保证可信，再谈“成功”

### 11.1 baseline 必须固定

主对照必须使用：

```text
--model-type mshnet
--mshnet-variant deterministic
--deep-supervision legacy_exact
--fusion-regularizer none
```

JIP 除 `--mshnet-variant jip1/2/3/4` 外，其余 seed、manifest、数据增强、optimizer、学习率、warm epoch、总 epoch、测试间隔和 checkpoint selection 完全一致。

不能使用 `workbench` baseline；不能把 legacy DEA head、fusion alpha 或历史 regularizer 混入。

### 11.2 数据划分与选择泄漏

仓库过去的冻结 test audit 只能作为机制线索。新前缀选择必须：

1. 从官方 training set 内固定划分 development train/validation；
2. 保存文件列表与 SHA256；
3. 只用 development validation 选择 `JIP-1/2/3/4`；
4. official test 在前缀冻结前保持封存；
5. 前缀冻结后，用完整 official train 重训并一次性报告 official test。

### 11.3 配对种子

- 开发门槛：至少 3 个 paired seeds；
- 最终 NUAA-SIRST：建议 5 个 paired seeds；
- NUDT-SIRST、IRSTD-1K：各至少 3 个 paired seeds；
- 每个 seed 的 baseline 与 JIP 使用相同初始化 seed、data-order seed 与 split manifest。

### 11.4 checkpoint 选择

保持仓库当前规则：每个 run 在其固定测试/验证时点中，独立选择自己的 best-IoU checkpoint。禁止：

- 用同 epoch 结果替代 independently selected best；
- 看见中期领先就提前宣判；
- 看见中期落后就调结构；
- 只挑最有利 seed。

### 11.5 主指标

至少报告：

- pixel IoU；
- nIoU（与领域标准一致时）；
- target-level PD；
- FA/Mpixel；
- raw-logit component FROC；
- FPPI budgets：`0.01, 0.05, 0.1, 0.2, 0.5, 1.0`；
- false component count / image；
- Params、FLOPs、真实 latency、peak memory。

组件 FROC 必须直接在 raw logits 上构造阈值，避免 float32 sigmoid 饱和破坏排序。

---

## 12. 预注册成功门槛

### 12.1 `CODE PASS`——已完成

必须同时满足：

- 参数增量 0；
- strict state-dict compatibility；
- 输出尺寸一致；
- CPU/CUDA finite backward；
- deterministic gate；
- 有界性和等变性测试；
- `active_stages=()` 与 baseline identity。

目前除目标 GPU deterministic CUDA gate 外，其余原型检查已经完成；正式仓库接入后需再跑一次完整测试集。

### 12.2 `JIP-1 DESIGN PASS`

在 development validation 的 3 个 paired seeds 上同时满足：

1. 三个 seed 的 IoU 差值均不为负；
2. paired mean `ΔIoU ≥ +0.003`（绝对值，即至少 +0.3 个百分点）；
3. paired mean `ΔPD ≥ -0.002`；
4. 至少 2/3 seeds 的 FA/Mpixel 更低；
5. 至少 2/3 seeds 的六预算 component-FROC 平均 PD 更高；
6. 低 FPPI `0.01/0.05` 不出现系统性下降；
7. 目标区域的删除量显著小于 false-component 区域；
8. 真实推理 latency 增量不高于 15%。

满足后，JIP 方程与 boundary0 永久冻结，`JIP-1` 已构成完整模型。

### 12.3 后续前缀门槛

`JIP-(k+1)` 必须相对已经冻结的 `JIP-k` 满足：

- mean `ΔIoU > 0`；
- PD 下降不超过 0.2 pp；
- component-FROC 不下降；
- 新边界的 `ι` 仍能区分 false 与 target；
- 效率仍在预算内。

否则撤销新边界，最终模型就是 `JIP-k`。不能回头调 boundary0。

### 12.4 `PAPER READY`

建议至少满足：

- 在三个公开数据集中的至少两个，paired mean IoU 提升达到或超过 +0.5 pp；
- 三个数据集平均增益为正；
- 没有数据集出现超过 0.5 pp 的 mean PD 损失；
- 至少两个数据集的 component-FROC 改善；
- 参数增加为 0；
- 机制分布支持“self-confirmation removal”而不仅是一般正则化；
- 与 fixed-α、SPT0、Soft-LOCI、random control 的归因对照成立。

若只降低 FA 而 IoU/PD 重演 SPT0 的失败，不得以“trade-off”包装为主模型成功。

---

## 13. 必做消融：证明不是普通 pooling 换名

| 对照 | 目的 | 预期关系 |
|---|---|---|
| MaxPool | canonical baseline | 主参考 |
| fixed `α=0.046875` | 检查输入自适应是否优于同平均强度常数删除 | JIP 应更稳健 |
| fixed `α=0.02/0.05/0.10` | 复核仓库冻结线索 | 形成幅度曲线 |
| hard self-including consensus | 验证“包含自身 vote”有污染 | 应不如 leave-one-out |
| Soft-LOCI | 对照仓库已有 softmax ownership helper | 证明 hard/tie-aware、无温度形式的必要性 |
| counterfactual self-support ratio | 对照归一化比例式过强删除 | JIP 应更保护 PD |
| SPT0 | 正式失败的强删除对照 | JIP 应避免 IoU/PD 损失 |
| random same-budget deletion | 排除只是小幅噪声正则 | JIP 应显著优于 random |
| shuffled peer ownership | 破坏跨通道空间对应，但保留边际分布 | 验证空间支持是关键 |
| JIP without stop-gradient | 检查直接可学习统计是否产生 identity escape | detached 版本应更可控 |
| JIP-1/2/3/4 | 确定同一原理的有效前缀 | 按冻结协议选择 |

最重要的归因对照是：

1. JIP vs 同期望删除强度 fixed α；
2. JIP vs self-including hard consensus；
3. JIP vs shuffled peer ownership；
4. JIP vs Soft-LOCI；
5. JIP vs SPT0。

只有这些对照成立，论文才能把贡献写成“精确自影响消除”，而不是“换了一个 pooling”。

---

## 14. 统计报告

### 14.1 推荐统计量

- 每个 dataset 报 paired seed mean ± std；
- 同时对 image-level IoU/PD 做 hierarchical paired bootstrap；
- 报 `Δ` 的 95% confidence interval；
- 对 `ι_target` 与 `ι_false` 报 Cliff’s delta 或 rank-biserial effect size；
- 对 FROC 报六预算均值及每个预算的 paired difference。

### 14.2 不允许的报告方式

- 只报单 seed；
- baseline 用 epoch379，方法用任意最有利 epoch，但不说明独立选择规则；
- 只报 pixel FA、不报 component FROC；
- 用 test set 选 `JIP-1/2/3/4`；
- 把失败 seed 删除；
- 把 `0.000x` 的随机波动写成显著提升。

---

## 15. AAAI-27 倒排计划

### 2026-07-13

- 冻结 JIP 数学定义；
- 完成 operator 与 MSHNet wrapper 原型；
- 完成代码级性质测试；
- 不再改公式。

### 2026-07-14

- 正式接入仓库 `main.py`；
- 添加 `jip1/jip2/jip3/jip4`；
- 跑完整 pytest 与目标 GPU deterministic gate；
- 用 baseline checkpoint 做 development split frozen audit；
- 启动 3 个 paired seeds 的 `JIP-1`。

### 2026-07-15

- 检查训练/梯度/机制日志是否异常；
- 只允许修复实现 bug，不允许改变方程；
- baseline paired runs 同步进行；
- 完成 novelty final search。

### 2026-07-16

- 若 `JIP-1` 达到开发门槛，正式冻结 boundary0，启动 `JIP-2`；
- 若较长训练尚未结束，只能形成 interim 判断，不能伪造 DESIGN PASS；
- 同时准备论文方法、定理与实验协议章节。

### 2026-07-17

- 决定是否扩展 `JIP-3`；
- 最迟在本日确定摘要中的模型名称、问题定义和核心方程；
- `JIP-1` 已通过时，即使更深前缀未完成，也以 `JIP-1` 作为完整模型。

### 2026-07-18 至 2026-07-19

- 冻结最终前缀；
- 启动/完成 NUAA-SIRST 5-seed confirmation；
- 启动 NUDT-SIRST 与 IRSTD-1K；
- 完成 fixed-α、SPT0、Soft-LOCI、shuffled-peer 核心对照。

### 2026-07-20

- 冻结标题、TL;DR 与摘要；
- 摘要中只写已经验证的定性与定量结果；
- 不留 placeholder；
- 检查 AAAI topics 与 nominated reviewer。

### 2026-07-21

- 按项目内部截止提交摘要，不把 AoE 的台北时间余量当作常规开发时间；
- 保存 OpenReview submission snapshot。

### 2026-07-22 至 2026-07-27

- 完成跨数据集与效率实验；
- 完成主表、消融、FROC、机制图、失败案例；
- 写完全文并内审。

### 2026-07-28

- 项目内部完成全文提交；
- 不依赖台北时间 7 月 29 日 19:59 的最后余量。

---

## 16. 论文立意与贡献写法

### 16.1 一句话立意

> MSHNet 的问题不在于缺少更多尺度或更复杂注意力，而在于最大池化允许一个通道的局部峰值参与构造“跨通道支持”，再用这个被自身污染的支持验证它自己。

### 16.2 建议贡献点

1. **问题发现**：通过 MSHNet 的分段组件轨迹和 strongest-site deletion audit，定位跨分辨率传输中的峰值自确认偏差，而非继续在末端融合堆模块。
2. **方法推导**：提出 tie-aware Jackknife self-influence，精确计算当前通道 vote 对其自身空间支持的贡献，并只从 max-to-second counterfactual gap 中删除该贡献。
3. **天然校准**：证明修正严格受 `1/C` 约束；在随机四位点零假设下期望为 `3/(4C)`，在 MSHNet 第一边界自动得到 4.6875%，与独立冻结审计的有效小删除区间一致。
4. **简洁架构**：构建零参数、state-dict compatible 的 JIP-MSHNet；按输入到深层的连续前缀部署同一方程，不增加分支、head、loss 或 inference module。
5. **机制验证**：通过 paired multi-seed、component-FROC、fixed-budget/random/shuffled-peer controls 和 target-vs-false influence distributions，检验提升是否确由 self-confirmation removal 产生。

### 16.3 论文标题候选

首选：

> **A Peak Should Not Vote for Itself: Jackknife Influence Pooling for Infrared Small Target Detection**

备选：

> **Removing Self-Confirmed Peaks: Parameter-Free Jackknife Downsampling for Infrared Small Target Detection**

> **Jackknife Evidence Transport for Scale-Sensitive Infrared Small Target Detection**

不建议在标题中同时写 MSHNet、multi-scale fusion、counterfactual responsibility、frequency 等多个概念。

### 16.4 TL;DR 候选

> We replace MSHNet’s max-pooling with a parameter-free jackknife operator that removes only the support a channel contributes to validating its own local maximum, yielding bounded, channel-adaptive suppression of isolated clutter without adding branches, parameters, or losses.

### 16.5 英文摘要骨架

> Infrared small target detectors must preserve weak target evidence while preventing isolated clutter peaks from surviving repeated downsampling. We revisit MSHNet and find that the main limitation is not a lack of additional heads or fusion modules: at a max-pooling boundary, a channel can participate in the cross-channel spatial support population that is subsequently used to validate its own maximum. We term this effect peak self-confirmation. To remove it, we introduce Jackknife Influence Pooling (JIP), a parameter-free operator that compares support computed with and without the current channel and subtracts only the resulting finite-sample influence from the max-to-second-max counterfactual gap. The correction is bounded by the inverse channel count, vanishes under unanimous peer support, and is naturally calibrated to early high-resolution stages without learned gates or stage-specific coefficients. Replacing a continuous prefix of MSHNet’s native pooling layers yields JIP-MSHNet while preserving the original decoder, multi-scale heads, fusion layer, loss, and parameter count. **[Only after experiments: add one verified sentence summarizing multi-dataset IoU, PD, false-alarm and component-FROC results.]** Mechanism analyses and matched controls distinguish JIP from fixed rank mixing, self-including consensus, and learned adaptive pooling.

提交前必须删除方括号提示，并用真实结果替换；不能填入预期数值。

---

## 17. 风险与预先处置

### 风险 1：JIP 太接近仓库已有 Soft-LOCI helper

处置：

- 不隐瞒内部已有 helper；
- Soft-LOCI 作为正式 ablation；
- 主贡献强调 hard/tie-aware finite-sample formulation、零假设校准、stop-gradient、连续前缀架构与完整机制实验；
- 若 Soft-LOCI 在公平训练中稳定优于 JIP，则不能回避，需重新评估主方法归因。

### 风险 2：硬 argmax ownership 梯度不连续

处置：

- ownership 本来就是 stop-gradient 统计；
- 主值梯度通过 `m1/m2` 传播；
- 在目标 GPU 上测试 deterministic behavior；
- 报告 tie rate；
- 不通过引入 temperature 来“平滑”主模型，soft 版本只作对照。

### 风险 3：只降低 FA，再次损失 PD

处置：

- correction 最大仅 `1/C`，远弱于 SPT0 的 50%；
- `JIP-1` 先行；
- PD 和低预算 FROC 是硬门槛；
- 若失败，不加后端救援模块。

### 风险 4：更深 boundary 修正过小、没有收益

处置：

- 这不影响 `JIP-1` 作为完整模型；
- 前缀协议会停在最后一个通过边界；
- 深层 correction 自然衰减本身可作为“避免破坏语义”的设计性质。

### 风险 5：审稿人认为只是 pooling trick

处置：

- 以“self-vote contamination / finite-sample jackknife influence”为主线，不以“新 pooling layer”作唯一卖点；
- 给出代数推导、界、零假设期望和 MSHNet 结构诊断；
- 加入 self-including、shuffled-peer、fixed-budget、Soft-LOCI controls；
- 用 component-level mechanism evidence 证明不是一般正则化。

### 风险 6：实验选择泄漏

处置：

- 用 train 内 development validation 选前缀；
- official test 封存；
- 保存 manifest hash；
- 在论文中公开前缀选择规则。

---

## 18. 立即执行清单

按优先级执行：

1. 将第 7 节两份代码写入仓库；
2. 为 `main.py` 增加 `jip1–jip4`，但第一轮只跑 `jip1`；
3. 跑完整单元测试和目标 GPU deterministic gate；
4. 建立 train-only development split 与 SHA256 manifest；
5. 用相同 baseline checkpoint 做 frozen mechanism audit；
6. 启动 deterministic baseline 与 `JIP-1` 的 3 个 paired seeds；
7. 同步记录 `π、ι、gap、actual deletion`；
8. `JIP-1` 通过后立即冻结，不再改公式；
9. 后续只允许把同一算子扩展到下一边界；
10. 7 月 20 日根据已验证结果冻结摘要。

### 建议 run label

```text
jip1_nuaa_dev_seed20260713_formula-hard-tie-jackknife-v1
jip1_nuaa_dev_seed20260714_formula-hard-tie-jackknife-v1
jip1_nuaa_dev_seed20260715_formula-hard-tie-jackknife-v1
```

正式 run metadata 必须记录：

- git commit；
- dataset manifest hash；
- baseline checkpoint hash；
- seed；
- active stages；
- formula version；
- PyTorch/CUDA/cuDNN versions；
- deterministic flags；
- independent best-checkpoint selection rule。

---

## 19. 最终边界声明

这份文档已经完成的是：

- 重新抓取并以最新 commit 为权威；
- 否决把历史失败模型继续包装为最终方案；
- 从 MSHNet 原生结构与现有诊断中选定最前端、最有定量依据的结构问题；
- 给出单一原理、零参数、非模块堆叠的完整模型；
- 给出数学推导、性质、代码、测试、前缀冻结规则、归因消融与 AAAI 时间表；
- 完成原型代码级验证。

尚未完成、也不能虚构的是：

- JIP-MSHNet 的 400 epoch paired training；
- 多数据集性能；
- 最终 `DESIGN PASS`；
- SOTA 结论。

“不要再失败”在本项目中的可执行含义应当是：**不再无边界地试错、不再在失败结构上继续堆模块、不再移动验收标准；每一步只有预注册的通过、冻结或撤销新边界，前面已经通过的模型不被后续实验破坏。**

---

## 20. 主要来源

### 仓库与实现

1. [最新提交：Add structural counterfactual experiments and audits](https://github.com/Arialliy/dea3090/commit/43c8c8367c21b64cae9e719868aaccda5cc6d329)
2. [最新提交下的 README](https://github.com/Arialliy/dea3090/blob/43c8c8367c21b64cae9e719868aaccda5cc6d329/README.md)
3. [main.py](https://github.com/Arialliy/dea3090/blob/43c8c8367c21b64cae9e719868aaccda5cc6d329/main.py)
4. [canonical MSHNet](https://github.com/Arialliy/dea3090/blob/43c8c8367c21b64cae9e719868aaccda5cc6d329/model/baselines/mshnet_official.py)
5. [deterministic MSHNet](https://github.com/Arialliy/dea3090/blob/43c8c8367c21b64cae9e719868aaccda5cc6d329/model/baselines/mshnet_deterministic.py)
6. [SPT0 implementation](https://github.com/Arialliy/dea3090/blob/43c8c8367c21b64cae9e719868aaccda5cc6d329/model/support_persistence_transport.py)
7. [order-statistic pooling controls](https://github.com/Arialliy/dea3090/blob/43c8c8367c21b64cae9e719868aaccda5cc6d329/utils/order_statistic_pool.py)
8. [SIED prototype](https://github.com/Arialliy/dea3090/blob/43c8c8367c21b64cae9e719868aaccda5cc6d329/model/dea_scale_interaction_exchange.py)

### 近期 IRSTD / 多尺度方法

9. [MSHNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html)
10. [PConv-SD, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32996)
11. [LoMix, NeurIPS 2025](https://openreview.net/forum?id=87c2JwNJa0)
12. [DEFANet, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37368)
13. [NS-FPN, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html)
14. [InvDet, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html)

### Pooling 邻近文献

15. [Detail-Preserving Pooling, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Saeedan_Detail-Preserving_Pooling_in_CVPR_2018_paper.html)
16. [Local Importance-Based Pooling, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gao_LIP_Local_Importance-Based_Pooling_ICCV_2019_paper.pdf)
17. [SoftPool, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Stergiou_Refining_Activation_Downsampling_With_SoftPool_ICCV_2021_paper.pdf)
18. [Adaptive Polyphase Sampling, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chaman_Truly_Shift-Invariant_Convolutional_Neural_Networks_CVPR_2021_paper.html)
19. [Translation Invariant Polyphase Sampling, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Saha_Improving_Shift_Invariance_in_Convolutional_Neural_Networks_with_Translation_Invariant_WACV_2025_paper.html)

### AAAI-27

20. [AAAI-27 Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)
21. [AAAI-27 Paper Modification Guidelines](https://aaai.org/conference/aaai/aaai-27/paper-modification-guidelines/)
22. [AAAI-27 Main Technical Track Call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
