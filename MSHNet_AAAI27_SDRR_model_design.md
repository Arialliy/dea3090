# SDR-MSHNet：基于尺度删除责任归因的 MSHNet 单点重构方案

> **当前有效状态（2026-07-13）**：第 0–31 节保留为历史设计与负结果台账，不代表当前
> 模型已冻结。权威执行口径从第 32 节开始：前段尚未 DESIGN PASS，正在完整训练 SPT0；
> 最终只按各方法各自的 best-IoU checkpoint 进行 best-vs-best 判断，禁止用同 epoch
> 截面冻结结构。

> **版本**：Complete Model v1.1  
> **日期**：2026-07-12  
> **目标会议**：AAAI-27 Main Technical Track  
> **Baseline**：Canonical MSHNet（CVPR 2024）  
> **方法名**：Scale-Deletion Responsibility Refinement（SDRR）  
> **模型名**：SDR-MSHNet  
> **唯一核心改动**：修正 MSHNet 最终多尺度融合“只聚合、不追责”的结构性缺陷  
> **新增可学习参数**：0  
> **推理参数量 / FLOPs / 路径变化**：0 / 0 / 无变化

---

## 0. 最终决策

不再设计新编码器、注意力、卷积块、门控器或多损失堆叠。主模型只改 MSHNet 的一个关键点：

> **MSHNet 的四个尺度预测经过一个 `4 → 1` 的线性卷积得到最终结果，但现有训练只知道“最终像素错了”，不知道“究竟是哪一个尺度把这个背景像素推成了目标”。**

因此，最终融合层可能出现以下现象：某个尺度在复杂背景上提供了足以改变最终二值决策的错误正证据；普通 IoU/SLS 只能对聚合结果施加梯度，无法把额外校正精确分配给真正造成该错误决策的尺度。

SDRR 对原始融合卷积做**精确代数分解**，逐尺度执行训练期删除干预：只有当删除尺度 `i` 的原生贡献会使一个安全背景像素从正决策翻转为非正决策时，才将尺度 `i` 判定为该决策的“责任尺度”，并仅抑制该尺度在该像素上的贡献。

```text
四个原生 side logits
        │
        ▼
对齐并拼接 S=[s0,s1,s2,s3]
        │
        ▼
原始 final Conv(4→1, 3×3) ───────────────► z（部署输出，完全不变）
        │
        ├── 精确分解：z = b + c0 + c1 + c2 + c3
        │
        ├── 删除尺度 i：z\i = z - ci
        │
        └── 背景正决策发生翻转？── 是 ──► 仅抑制 ci（训练期）
```

这不是“再加一个模块”，而是把 MSHNet 已有融合层从**无责任聚合**改造成**可归责训练、原样部署**的融合机制。

---

## 1. 为什么应当改最终融合，而不是继续堆模块

### 1.1 MSHNet 的真实结构

当前 MSHNet 的主体是 U-Net 式编码器—解码器，解码端产生四个单通道 side logits：

\[
 m_0, m_1, m_2, m_3.
\]

将低分辨率预测双线性上采样至最高分辨率后得到：

\[
S=\operatorname{Concat}(s_0,s_1,s_2,s_3)\in\mathbb R^{B\times4\times H\times W}.
\]

最终预测为：

\[
z=\operatorname{Conv}_{3\times3}^{4\rightarrow1}(S).
\]

仓库当前实现对应：

```python
self.output_0 = nn.Conv2d(16, 1, 1)
self.output_1 = nn.Conv2d(32, 1, 1)
self.output_2 = nn.Conv2d(64, 1, 1)
self.output_3 = nn.Conv2d(128, 1, 1)
self.final = nn.Conv2d(4, 1, 3, 1, 1)
```

### 1.2 核心缺陷：side supervision 与 final fusion 之间没有责任闭环

四个 side heads 都被当作完整检测器进行监督，而最终融合卷积又可以自由组合四个预测。这里存在一个未被 MSHNet 处理的训练缺口：

1. side loss 要求每个尺度都尽可能拟合完整标签；
2. final loss 只约束聚合后的结果；
3. final loss 发生背景误检时，梯度会经过融合权重扩散到多个尺度；
4. 训练目标没有判断哪个尺度对当前错误正决策是**必要的**；
5. 因而无关尺度也可能被修改，产生尺度间梯度干扰，而真正致错的尺度没有被单独校正。

这比“特征不够强”更接近 MSHNet 自身的结构问题：**多尺度信息已经生成，但最终融合缺少决策责任约束。**

### 1.3 为什么当前不把 LLoss 重写作为主方法

仓库 `model/loss.py` 中的 `LLoss` 也存在值得修正的问题：其位置统计使用全图矩，而非逐实例、归一化质心；多个目标可能相互抵消，预测质量与位置偏差也可能耦合。原 MSHNet 论文的消融还显示，加入 location 项提高了 IoU/PD，但可能明显增加 FA。

不过，此方向现在不应作为 AAAI 主线，原因是：

- AAAI 2025 的 PConv + SD Loss 已经围绕不同尺度动态调整 scale/location loss；
- 2026 年的 diff-based scale loss 已直接重新讨论 SLS 的单调性；
- 2026 年 AC-SLSIoU 又在 logit margin、边界和 false alarm loss 上进行组合；
- 仓库现有 location-loss 改写没有形成与 SDRR 同等级的严格三种子正证据。

因此，**LLoss 只作为附录诊断，不与 SDRR 共同组成主方法**。主文保持一个问题、一个机制、一个新增目标。

---

## 2. 提交前必须先清理的 baseline 问题

当前 `model/MSHNet.py` 已加入：

```python
self.decidability_head = nn.Sequential(...)
```

即使该 head 在 canonical forward 中未使用，它仍会改变：

- 参数量；
- `state_dict` keys；
- checkpoint identity；
- `strict=True` 加载行为；
- 论文中“原始 MSHNet baseline”的可验证性。

### 2.1 必做改动

建立独立文件：

```text
model/
├── mshnet_canonical.py     # 只保留官方 MSHNet 结构
├── sdrr.py                 # 训练期精确分解与 SDRR loss
└── MSHNet.py               # 旧实验代码保留，但不用于正式实验
```

`mshnet_canonical.py` 中必须：

1. 删除 `decidability_head`；
2. 删除 `build_dea_lite_outputs`；
3. 删除 `return_dea`、`dea_detach_evidence`、`fusion_alpha` 等额外 forward 语义；
4. 保留与官方 forward 相同的四个 side outputs 与 final output；
5. 用 `strict=True` 验证官方/clean checkpoint；
6. 固定输入下验证 canonical 输出与 clean 实现逐张量一致。

### 2.2 一个容易出错的偏置问题

当前 DEA 路径中的 `z_only` 给每个尺度分支都加了完整融合偏置。它可作为“只保留一个尺度时的输出”，但**不能**直接当作四个可相加的尺度贡献，否则融合偏置会被重复四次。

SDRR 必须使用：

\[
c_i=W_i*s_i,
\]

其中每个 `c_i` **不含 bias**；最后只加一次：

\[
z=b+\sum_i c_i.
\]

---

## 3. SDRR 方法定义

### 3.1 原生尺度贡献的精确分解

将最终融合卷积的输入通道权重写为：

\[
W=[W_0,W_1,W_2,W_3].
\]

由于卷积对输入通道是线性的：

\[
\begin{aligned}
z
&=W*S+b\\
&=\sum_{i=0}^{3} W_i*s_i+b\\
&=b+\sum_{i=0}^{3}c_i,
\end{aligned}
\]

其中：

\[
c_i=W_i*s_i.
\]

`c_i` 使用的就是部署时 final convolution 的真实权重，不是代理头、额外 attention 或估计器。

### 3.2 尺度删除干预

删除尺度 `i` 的精确输出为：

\[
z_{\setminus i}=z-c_i.
\]

这与把融合输入的第 `i` 个通道置零后重新执行原始 final convolution 数学等价，并保留其他尺度及融合偏置。

### 3.3 安全背景区域

红外小目标的标注边界和热扩散可能存在轻微不确定性。为避免把目标附近的正响应误当成虚警，先对标签做膨胀保护：

\[
P=\operatorname{MaxPool}(Y;k),
\]

\[
B=\mathbf 1[P=0].
\]

正式设置冻结为：

```text
safe_kernel = 15
```

即只有远离标注目标的安全背景像素进入责任约束。

### 3.4 决策翻转责任

默认概率阈值为 `0.5`，对应 logit 阈值 `0`。尺度 `i` 在像素 `p` 上的责任事件定义为：

\[
r_i(p)=
\mathbf 1[B(p)=1]\,
\mathbf 1[z(p)>0]\,
\mathbf 1[z_{\setminus i}(p)\le0].
\]

含义是：

- 完整 MSHNet 把该安全背景像素判断为目标；
- 删除尺度 `i` 后，该像素不再被判断为目标；
- 所以尺度 `i` 对这个当前错误正决策是 decision-pivotal 的。

责任 mask 完全停止梯度：

\[
\operatorname{stopgrad}(r_i).
\]

离散归责不参与反向传播，梯度只作用于被选中的原生尺度贡献。

### 3.5 责任抑制目标

由于责任事件满足：

\[
z>0,\quad z-c_i\le0,
\]

可得：

\[
c_i\ge z>0.
\]

因此责任贡献必为正证据。使用单调、稳定的 softplus 抑制：

\[
\mathcal L_{\mathrm{SDRR}}
=
\frac{
\sum_{p,i}r_i(p)\operatorname{softplus}(c_i(p))
}{
\max\left(1,\sum_{p,i}r_i(p)\right)
}.
\]

若当前 batch 没有责任事件，则：

\[
\mathcal L_{\mathrm{SDRR}}=0.
\]

总目标保持 canonical MSHNet 原有五项监督不变：

\[
\mathcal L
=
\mathcal L_{\mathrm{MSHNet}}
+
\lambda(t)\mathcal L_{\mathrm{SDRR}}.
\]

正式超参数冻结为：

```text
lambda_max = 0.05
start_epoch = 250
ramp_epochs = 50
safe_kernel = 15
normalization = event
logit_threshold = 0.0
```

其中 `lambda(t)` 在第 250 epoch 之前为 0，随后用 50 epochs 线性升至 0.05。延迟介入的目的不是复杂 curriculum，而是避免模型尚未形成可靠正决策时由极少量瞬时翻转事件主导优化。

---

## 4. 四个可直接写进论文的性质

### Proposition 1：精确重构

对 MSHNet 原始 final convolution，有：

\[
z=b+\sum_i c_i.
\]

**证明**：标准卷积对输入通道求和，按输入通道拆分权重即可。这里没有近似、采样或额外模型。

### Proposition 2：删除等价性

\[
z_{\setminus i}=z-c_i
\]

与将原始融合输入 `S[:, i]` 置零后重新执行 final convolution 等价。

### Proposition 3：责任贡献符号确定

若 `r_i=1`，则 `c_i>0`。因此对 `c_i` 施加单调下降压力与消除该背景正决策的方向一致。

### Proposition 4：部署完全等价

SDRR 只在训练时读取 side logits 与 final weights 计算额外 loss；测试时删除整个责任计算路径。部署模型仍是原始 canonical MSHNet，因此：

- 无新增参数；
- 无新增推理算子；
- 无新增显存占用；
- 可直接使用原始 MSHNet 推理代码。

---

## 5. 可直接落地的 PyTorch 实现

建议新建 `model/sdrr.py`：

```python
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def assemble_scale_logits(side_logits: Sequence[Tensor]) -> Tensor:
    """将 MSHNet 四个原生 side logits 对齐至最高分辨率。"""
    if len(side_logits) != 4:
        raise ValueError("SDRR requires exactly four MSHNet side logits")
    if any(x.ndim != 4 or x.shape[1] != 1 for x in side_logits):
        raise ValueError("each side logit must have shape [B, 1, H, W]")

    batch = side_logits[0].shape[0]
    if any(x.shape[0] != batch for x in side_logits):
        raise ValueError("all side logits must have the same batch size")

    output_size = side_logits[0].shape[-2:]
    aligned = [side_logits[0]]
    aligned.extend(
        F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            align_corners=True,
        )
        for x in side_logits[1:]
    )
    return torch.cat(aligned, dim=1)


def exact_scale_decomposition(
    side_logits: Sequence[Tensor],
    z_full: Tensor,
    fusion: nn.Conv2d,
) -> dict[str, Tensor]:
    """精确分解原始 MSHNet final convolution，不增加可学习参数。"""
    if not isinstance(fusion, nn.Conv2d):
        raise TypeError("fusion must be nn.Conv2d")
    if fusion.in_channels != 4 or fusion.out_channels != 1:
        raise ValueError("fusion must be a 4-to-1 convolution")
    if z_full.ndim != 4 or z_full.shape[1] != 1:
        raise ValueError("z_full must have shape [B, 1, H, W]")

    scale_logits = assemble_scale_logits(side_logits)

    # fusion.weight: [1, 4, kh, kw]
    # grouped_weight: [4, 1, kh, kw]
    # groups=4 后，一次卷积得到四个不含 bias 的原生尺度贡献。
    grouped_weight = fusion.weight[0].unsqueeze(1)
    contributions = F.conv2d(
        scale_logits,
        grouped_weight,
        bias=None,
        stride=fusion.stride,
        padding=fusion.padding,
        dilation=fusion.dilation,
        groups=4,
    )

    if contributions.shape[-2:] != z_full.shape[-2:]:
        raise RuntimeError("contribution shape differs from z_full")

    reconstructed = contributions.sum(dim=1, keepdim=True)
    if fusion.bias is not None:
        reconstructed = reconstructed + fusion.bias.view(1, 1, 1, 1)

    deletion_logits = z_full - contributions

    return {
        "scale_logits": scale_logits,
        "contributions": contributions,
        "deletion_logits": deletion_logits,
        "reconstructed": reconstructed,
    }


def build_safe_background(target: Tensor, kernel_size: int = 15) -> Tensor:
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("target must have shape [B, 1, H, W]")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    protected = F.max_pool2d(
        (target > 0.5).to(target.dtype),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return (protected < 0.5).to(target.dtype)


def sdrr_loss(
    z_full: Tensor,
    contributions: Tensor,
    target: Tensor,
    *,
    safe_kernel: int = 15,
    logit_threshold: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Scale-Deletion Responsibility Refinement loss."""
    if z_full.ndim != 4 or z_full.shape[1] != 1:
        raise ValueError("z_full must have shape [B, 1, H, W]")
    if contributions.ndim != 4 or contributions.shape[1] != 4:
        raise ValueError("contributions must have shape [B, 4, H, W]")
    if target.shape != z_full.shape:
        raise ValueError("target and z_full must have identical shapes")
    if contributions.shape[0] != z_full.shape[0]:
        raise ValueError("batch sizes differ")
    if contributions.shape[-2:] != z_full.shape[-2:]:
        raise ValueError("spatial shapes differ")

    safe_background = build_safe_background(target, safe_kernel)
    deletion_logits = z_full - contributions

    # 责任分配是离散的模型内部干预；禁止沿 mask 反传。
    with torch.no_grad():
        responsibility = (
            (z_full.detach() > logit_threshold)
            & (deletion_logits.detach() <= logit_threshold)
            & (safe_background.expand_as(contributions) > 0.5)
        ).to(z_full.dtype)

    penalty = F.softplus(contributions)
    event_count = responsibility.sum()
    loss = (penalty * responsibility).sum() / event_count.clamp_min(1.0)

    # 无事件时保证严格零值，同时保留合法计算图。
    if int(event_count.detach().item()) == 0:
        loss = contributions.sum() * 0.0

    active_images = responsibility.flatten(1).sum(dim=1) > 0
    logs = {
        "sdrr_loss": loss.detach(),
        "responsible_count": event_count.detach(),
        "responsibility_ratio": responsibility.mean().detach(),
        "responsible_image_ratio": active_images.float().mean().detach(),
        "safe_background_ratio": safe_background.mean().detach(),
        "responsible_contribution_mean": (
            (contributions.detach() * responsibility).sum()
            / event_count.clamp_min(1.0)
        ),
    }
    return loss, logs


def sdrr_weight(
    epoch: int,
    *,
    start_epoch: int = 250,
    ramp_epochs: int = 50,
    maximum: float = 0.05,
) -> float:
    """epoch 使用 1-based 计数。"""
    if epoch <= start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return float(maximum)
    progress = min(1.0, (epoch - start_epoch) / float(ramp_epochs))
    return float(maximum) * progress
```

### 5.1 训练循环接入

保持原 MSHNet loss 的计算和权重完全不变，只在其后增加：

```python
# canonical forward
side_logits, z_full = model(image, warm_flag=True)

# 原实现：不要改变五项 loss 的 target 构造、warm-up 和平均方式
loss_mshnet = canonical_mshnet_loss(
    side_logits=side_logits,
    z_full=z_full,
    target=target,
    epoch=epoch,
)

# SDRR：只读取原生 side logits 和原生 final convolution
fusion = model.module.final if hasattr(model, "module") else model.final
views = exact_scale_decomposition(side_logits, z_full, fusion)

loss_sdrr, sdrr_logs = sdrr_loss(
    z_full,
    views["contributions"],
    target,
    safe_kernel=15,
    logit_threshold=0.0,
)

weight = sdrr_weight(
    epoch,
    start_epoch=250,
    ramp_epochs=50,
    maximum=0.05,
)

loss = loss_mshnet + weight * loss_sdrr
loss.backward()
optimizer.step()
```

### 5.2 不要采用的错误实现

```python
# 错误 1：每个 contribution 都加 final.bias
c_i = conv_i(s_i) + bias

# 错误 2：新建四个代理 head 估计贡献
c_i = proxy_head_i(feature_i)

# 错误 3：让 responsibility mask 参与反向传播
responsibility = differentiable_gate(...)

# 错误 4：对所有背景正响应都施加同样损失
loss = softplus(contributions)[safe_background].mean()

# 错误 5：测试时保留四次删除 forward
prediction = aggregate(delete_each_scale(...))
```

这些写法分别会破坏精确分解、引入模块、改变方法语义，或产生推理开销。

---

## 6. 必须通过的单元测试

建议新增 `tests/test_sdrr.py`。

### 6.1 精确重构

```python
torch.testing.assert_close(
    views["reconstructed"],
    z_full,
    rtol=1e-5,
    atol=1e-6,
)
```

### 6.2 删除等价

对每个尺度 `i`：

1. 复制 `scale_logits`；
2. 将 `[:, i]` 置零；
3. 执行原始 `fusion`；
4. 与 `z_full - contributions[:, i:i+1]` 比较。

### 6.3 无事件严格为零

构造 `z_full < 0` 的 batch，断言：

```python
assert loss.item() == 0.0
```

### 6.4 目标保护区不产生责任事件

目标及其 `15×15` 膨胀邻域内的 responsibility 必须全零。

### 6.5 责任贡献为正

对所有 `responsibility == 1` 的位置，断言：

```python
assert torch.all(contributions[responsibility.bool()] > 0)
```

边界浮点误差下可使用 `>= -1e-7`，并单独记录接近阈值的比例。

### 6.6 推理恒等性

同一 canonical checkpoint、同一输入：

- `train wrapper` 关闭 SDRR 后的输出；
- 独立 canonical MSHNet 的输出；

必须逐张量一致。

### 6.7 baseline identity

正式实验前必须通过：

- 无 `decidability_head` key；
- official/clean checkpoint `strict=True`；
- 参数量一致；
- 固定输入下 side/final outputs 一致。

---

## 7. 与近期方法的明确区分

| 方法 | 核心对象 | 是否学习新融合/模块 | 是否识别“哪个尺度造成当前决策” | 推理开销 | 与 SDRR 的本质差异 |
|---|---|---:|---:|---:|---|
| MSHNet, CVPR 2024 | SLS loss + 简单多尺度 head | 有多尺度 head | 否 | 原始 | final fusion 只聚合，不做责任分配 |
| PConv + SD Loss, AAAI 2025 | Gaussian-like 卷积形状；按目标尺度动态平衡 scale/location loss | 是 | 否 | 有少量结构变化 | 调整特征提取和 loss 权重，不做原生尺度删除 |
| LoMix, NeurIPS 2025 | 对多尺度 logits 做加/乘/拼接/attention 混合并学习权重 | 训练期有可学习混合模块 | 否 | 测试可移除 | 学习“哪些混合有用”；SDRR 判断“哪个原生尺度对错误决策是必要的” |
| NS-FPN, CVPR 2026 | 低频引导净化 + spiral-aware sampling | 是 | 否 | 有结构变化 | 在 feature pyramid 抑噪；SDRR 不改特征网络 |
| Diff-based Scale Loss + Gaussian-Shape Conv, 2026 | 单调 scale loss + Gaussian/pinwheel convolution | 是 | 否 | 有结构变化 | 解决 scale-loss 几何和卷积先验，不做决策归责 |
| AC-SLSIoU, 2026 | logit margin + boundary suppression + false-alarm focal loss | 多个 loss 组件 | 否 | 无/低 | 对 hard negative、边界和虚警做全局约束；SDRR 只约束删除后会翻转决策的责任尺度 |
| **SDRR（本文）** | 原始 final fusion 的精确贡献和删除翻转 | **否** | **是** | **0** | 精确、稀疏、decision-pivotal 的尺度责任校正 |

### 7.1 论文中必须使用的差异表述

可以写：

> Unlike multi-scale mixing or dynamic loss reweighting, SDRR does not learn how to combine scales. It exactly decomposes the deployed affine fusion and regularizes a native scale only when deleting its contribution flips a safe-background positive decision.

不能只写：

> We propose a new multi-scale loss.

后者会被 LoMix、deep supervision、hard-negative mining 和近期 IRSTD loss 工作直接覆盖。

### 7.2 不使用“因果识别”表述

尺度删除是模型内部 counterfactual intervention，但不是对真实数据生成机制的因果干预。论文使用：

- exact deletion intervention；
- decision-pivotal responsibility；
- model-internal counterfactual attribution；

避免使用：

- causal scale；
- causal identification；
- true cause of false alarm。

---

## 8. 当前仓库已有证据：只作为内部验证

仓库记录的 strict 400-epoch NUAA-SIRST 三种子内部 holdout 结果如下。该表可以作为立即冻结方案的依据，但在 canonical baseline 物理隔离后必须复跑，且不能冒充 official test 结果。

| Seed | Baseline IoU / PD / FA | SDRR IoU / PD / FA | Paired IoU Δ |
|---:|---:|---:|---:|
| 20260711 | 0.7369 / 0.9630 / 9.2262 | 0.7324 / 0.9630 / 6.0325 | -0.0045 |
| 20260712 | 0.6934 / 0.9444 / 1.7743 | 0.7350 / 0.9630 / 6.7423 | +0.0416 |
| 20260713 | 0.7250 / 0.9630 / 10.2908 | 0.7359 / 0.9815 / 5.3228 | +0.0109 |
| **Mean** | **0.7184 / 0.9568 / 7.0971** | **0.7344 / 0.9692 / 6.0325** | **+0.0160** |

附加观察：

- baseline / SDRR 的 IoU 样本标准差分别为 `0.0225 / 0.0018`；
- 一个 seed 的 IoU 下降，正式论文必须保留；
- 有责任事件的训练 batch 比例约为 `5.4% / 4.4% / 4.0%`，说明该项是稀疏事件驱动，而不是对全部背景持续重加权；
- 概率阈值 `0.3–0.7` 的内部均值曲线中，SDRR IoU 均高于 baseline；
- 这些结果尚不能证明提升一定来自“正确归责”，因为稀疏优化扰动本身可能带来收益，必须完成 matched controls。

### 8.1 当前证据支持什么

支持：

- 该方向已经比未验证的新结构更适合立即成型；
- 精确分解和训练路径可运行；
- 固定设置在至少一个严格三种子内部实验中出现正的平均增益；
- 零推理开销成立。

尚不支持：

- 每个 seed 都提升；
- 跨数据集稳定提升已完成；
- 相比近期 SOTA 已经显著领先；
- 责任归因已被 matched controls 证明；
- 因果结论。

### 8.2 NUDT-SIRST 独立数据集复核

同一冻结设置的 NUDT-SIRST 400-epoch 三种子结果已经完成。按每条 run 的 best-IoU，
SDRR 在 3/3 seeds 提升，paired mean IoU `+0.0083`；FA 在 3/3 seeds 下降，平均
`-7.725/Mpix`。但 PD 在 3/3 seeds 小幅下降，平均 `-0.0089`。epoch-399 与 last-20
mean IoU 分别为 `+0.0128`、`+0.0110`，均为 2/3 seeds 正；last-20 FA 在 3/3 seeds
下降。该结果建立了跨数据集 IoU/FA 正证据，但只支持 trade-off/stabilization 叙事，
不支持逐 seed Pareto 改善。

### 8.3 归一化控制：当前只支持“稳定性候选”

SIRST-v1 内部 holdout 的 `seed=20260712` 已完成 `safe_density` 控制。与 event-mean
SDRR 相比：

- best-IoU 为 `0.7316`，略低于 event-mean 的 `0.7350`（`-0.0034`）；
- epoch-399 IoU 为 `0.7011`，高于 event-mean 的 `0.6689`（`+0.0322`）；
- last-20 mean IoU 高 `+0.0524`，同时 FA 低 `58.214/Mpix`；
- 相对 canonical baseline，epoch-399 IoU 为 `+0.1106`、FA 为 `-139.458/Mpix`。

这说明 event-mean 在事件极稀疏时可能产生偏强的批级更新，而 density normalization
更像随责任密度自适应衰减的稳定化版本。但当前只有一个内部 seed，且 best-IoU 未超过
冻结主方法，因此不能据此事后替换主方法；必须补多 seed/official 复核后再决定默认归一化。

---

## 9. AAAI 必须完成的对照实验

核心问题不是再做更多模块消融，而是证明：收益来自“decision-pivotal attribution”，不是一般的稀疏正则或 hard-negative mining。

### M0：Canonical MSHNet

- 干净模型类；
- 原始五项 supervision；
- 无任何 DEA/TCDS/TFDS/SDRR 代码路径。

### P：SDR-MSHNet（proposed）

- 固定 `λ=0.05, start=250, ramp=50, kernel=15`；
- 不按数据集调参。

### M1：All Safe-Background Positive Suppression

在相同安全背景区域，对所有 `z>0` 的 fused logits 施加约束，不检查删除翻转。

用途：排除“只要加强 hard negative 就行”。

### M2：Same-Pivotal-Pixel Fused Suppression

保留 SDRR 找到的责任像素，但直接抑制该像素的 fused logit `z`，丢弃尺度身份。

用途：区分“找对像素”与“找对该像素中的原生尺度”。

### M3：Magnitude-Matched Non-Pivotal Control

在同一尺度匹配 full logit、contribution magnitude 与 deletion margin，但要求删除后仍为正，即该尺度不是 pivotal source。

用途：排除“只是选择了幅值较大的正贡献”。

### M4：Same-Pixel Random-Scale Control

保持责任像素、每像素事件数和被替换责任事件的 softplus 梯度幅值不变，只把梯度路由到随机的非责任尺度；随后匹配共享参数上的辅助梯度 L2。

用途：检验收益是否依赖真正的 native scale identity。直接对任意负贡献做 softplus 后再用 `1/sigmoid` 放大是数值不公平的旧实现，已作废。

### M5：Matched Random Event Control

对每个 image × scale，抽取与真实责任事件数量相同的非责任正贡献，并匹配 SDRR 的梯度 L2 预算。

用途：排除“同样数量的稀疏随机扰动就能提升”。

### 最小论文级结论条件

SDRR 应同时优于：

1. canonical baseline；
2. same-pivotal-pixel fused suppression；
3. magnitude-matched non-pivotal；
4. same-pixel random-scale；
5. matched random event。

否则论文只能证明“稀疏背景正则有效”，不能证明责任机制有效。

---

## 10. 正式实验协议

### 10.1 数据集

主实验：

- SIRST-v1（仓库历史目录名 `NUAA-SIRST`）；
- NUDT-SIRST；
- IRSTD-1K。

数据集作者已明确要求该数据集称为 **SIRST / SIRST-v1**，不要继续写作
“NUAA-SIRST”。代码、checkpoint 与旧工作对齐时可保留目录名 `NUAA-SIRST`，论文正文、
表格和 caption 统一使用 SIRST-v1，并在实验设置首次出现处说明 legacy directory alias。

可选泛化实验放在全文阶段：使用 UIUNet 的原生六侧输出 `Conv2d(6,1,1)` 融合，将
`c_i=w_i s_i` 直接代入 SDRR。该迁移只读取既有 fusion input/weight/bias，不添加 feature
module、参数或推理路径。现有 epoch-804 checkpoint 在本项目固定 214-image test
manifest 上的只读审计已确认存在 21 个数值稳定的 deletion-pivotal events；该结果仅
作为可训练性门控，性能结论必须来自独立 paired training。

### 10.2 配对设计

正式协议区分数据 manifest 与训练随机种子，不允许把二者混为一个参数：

- **数据 manifest**：不再生成第三个 validation split，只使用用户指定 `img_idx` 中随数据集提供的 `train_*.txt` 与 `test_*.txt`；SIRST-v1 为 213/214、NUDT-SIRST 为 663/664、IRSTD-1K 为 800/201，train/test 均互斥且完整覆盖数据集。每条 run 保存两份 manifest SHA-256；
- **训练种子与最终模型选择**：`20260711 / 20260712 / 20260713` 在内部 holdout 上严格配对并完整报告 mean ± std；最终发布模型允许按预先固定的 validation IoU 选择 seed。现有 SIRST-v1 与 NUDT-SIRST 均由 `20260713` 取得最高 SDRR best-IoU，因此最终 official train/test 模型预注册为 `seed=20260713`。

现有三种子内部 holdout 结果同时承担稳健性估计与 seed 选择；正式 train/test 只重训已经选定的 `20260713`，避免先观察多个 official-test 结果再挑最高值造成 test leakage。

内部三种子比较与最终 seed-13 配对均必须做到：

- 相同 split；
- 相同初始化；
- 相同 dataloader 顺序；
- 相同 optimizer state；
- 相同前 250 epochs trajectory；
- baseline 与 SDRR 从同一 checkpoint 分支；
- 除 SDRR 外配置完全一致。

因此，论文必须同时区分两项输出：三种子全量结果用于估计随机性；validation-selected seed 13 用于交付最终 checkpoint。不能删除负 seed，也不能在看完 official test 后重新选择 seed。

这种配对比“独立跑三次再比较均值”更能降低小数据集和训练方差带来的噪声。

### 10.3 严格确定性

保留仓库已做的前向等价修改：

```python
torch.amax(x, dim=(-2, -1), keepdim=True)
```

替代 `AdaptiveMaxPool2d(1)` 的非确定 CUDA backward，并启用：

```python
torch.use_deterministic_algorithms(True)
CUBLAS_WORKSPACE_CONFIG=:4096:8
num_workers=0  # 正式确定性实验
```

每条 run 保存：

- git commit；
- 完整 args；
- split hash；
- parent checkpoint SHA-256；
- model state keys；
- optimizer state；
- CUDA / PyTorch 版本；
- responsibility statistics。

### 10.4 指标

主表必须同时报告：

- IoU；
- Pd；
- Fa / million pixels；
- 三种子 mean ± std；
- paired delta；
- 参数量；
- FLOPs；
- inference FPS / latency。

附加分析：

- probability threshold `0.3–0.7` 曲线；
- 每尺度责任事件占比；
- 每图责任事件数量分布；
- full margin 与 deletion margin 分布；
- 责任事件前后可视化；
- 目标距离分桶的 FA；
- 多目标场景与复杂背景场景分桶。

### 10.5 数据纪律

- 日期型先导实验的 train holdout 仅用于在正式实验前冻结方法与超参数；
- validation-selected `seed=20260713` 使用完整 train manifest 拟合、official test manifest 评测，不额外扣除训练样本；
- baseline/SDRR 的共同 parent 只训练到 epoch 250 并保存 checkpoint，完全不迭代 official test loader；
- 两个分支只在预注册 epoch 399 各读取一次 official test，primary statistic 即该固定终点；official test 不再生成逐 epoch 轨迹或 best-test；
- 不根据 official test 改 `λ/start/ramp/kernel`；
- 不删除塌缩 seed；
- 不按数据集选择不同配置。

---

## 11. 论文贡献点冻结

正式稿只保留三项贡献：

### Contribution 1：发现并形式化 MSHNet 的融合责任缺口

现有多尺度监督只约束各 side prediction 和聚合结果，没有识别哪个原生尺度对当前错误决策是必要的。本文把问题从“如何再融合一次”改写为“如何对已有融合决策追责”。

### Contribution 2：提出精确的尺度删除责任优化

利用 affine fusion 的通道线性结构，精确计算每个尺度的原生贡献及删除输出；仅当删除会翻转安全背景正决策时施加尺度定向校正。没有代理网络、近似 attribution 或可学习 gate。

### Contribution 3：零推理开销与机制级对照

训练后完全恢复原始 MSHNet 部署图，并用 event-budget、same-pixel random-scale、magnitude-matched non-pivotal controls 区分责任归因与普通 hard-negative regularization。

不要把以下内容列成贡献：

- safe background dilation；
- linear ramp；
- softplus；
- 四个 side outputs；
- 多尺度 deep supervision。

它们是实现细节，不是创新主体。

---

## 12. 论文标题与摘要冻结稿

### 12.1 推荐标题

**Scale-Deletion Responsibility Refinement for Multi-Scale Infrared Small Target Detection**

备选标题：

**Which Scale Triggered the False Alarm? Exact Deletion Responsibility for Infrared Small Target Detection**

正式提交优先使用第一个，术语更稳定，也不会把论文限制为只有 false alarm 指标。

### 12.2 AAAI 摘要草稿（提交前填入最终跨数据集数字）

> Multi-scale side predictions are widely used in infrared small target detection to combine fine localization with coarse context. However, existing deep supervision and fusion objectives optimize individual side outputs or their aggregate prediction without identifying which scale is pivotal to an erroneous decision. We revisit MSHNet and observe that its deployed four-scale fusion is affine, enabling an exact decomposition of the final logit into native scale contributions. Based on this property, we propose Scale-Deletion Responsibility Refinement (SDRR), a training-only mechanism that deletes each scale contribution and assigns responsibility only when the deletion flips a positive decision on safe background. SDRR then suppresses only the responsible native contribution while detaching the discrete assignment from gradient propagation. Unlike learnable logit mixing, dynamic loss reweighting, or generic hard-negative mining, SDRR neither introduces a proxy attribution network nor modifies the inference graph. It adds no learnable parameter or inference cost and preserves the original MSHNet prediction path exactly. On NUAA-SIRST, NUDT-SIRST, and IRSTD-1K, SDR-MSHNet improves mean IoU by **[fill three validated values]** while maintaining competitive detection probability and false-alarm performance. Matched random-event, same-pixel random-scale, and magnitude-matched non-pivotal controls further demonstrate that the gain depends on decision-pivotal scale attribution rather than merely sparse background regularization.

**提交规则**：方括号数字必须在 7 月 21 日前替换，不能提交 placeholder。标题、问题定义和核心方法在摘要截止后不再改变。

---

## 13. 主文结构

### 1. Introduction

按以下逻辑写：

1. IRSTD 需要细粒度位置与多尺度上下文；
2. 多尺度 side heads 已广泛使用；
3. 现有方法关注特征增强、尺度加权或 logits mixing；
4. 被忽略的问题是：聚合决策错误时，训练并不知道哪个尺度是 pivotal；
5. MSHNet 的 affine fusion 提供了精确删除干预机会；
6. 提出 SDRR，零参数、零推理开销；
7. 给出三项贡献。

### 2. Related Work

只分三小节：

- Infrared small target detection；
- Multi-scale supervision and logit fusion；
- Counterfactual/deletion-based model attribution。

第三节要明确：本文不是 post-hoc explanation，而是利用 exact internal intervention 构造训练目标。

### 3. Method

- 3.1 Revisiting MSHNet fusion；
- 3.2 Exact native scale decomposition；
- 3.3 Scale-deletion responsibility；
- 3.4 Responsibility refinement objective；
- 3.5 Properties and complexity。

### 4. Experiments

- 4.1 Datasets and protocol；
- 4.2 Main comparison；
- 4.3 Attribution controls；
- 4.4 Stability and threshold analysis；
- 4.5 Visualization and failure cases。

### 5. Limitations

诚实写：

- 单尺度删除只能识别 individual pivotal contribution，不能完全刻画高阶 coalition interaction；
- 若模型全背景塌缩、没有正决策，SDRR 无事件，不能负责恢复检测能力；
- 责任定义依赖部署阈值，但阈值曲线可检验其稳健性。

---

## 14. 图表设计

### Figure 1：MSHNet 的融合责任缺口

同一背景误检位置展示：

- 四个 side logits；
- 四个 exact contribution maps；
- 完整输出 `z`；
- 四个删除输出 `z\i`；
- 只有一个尺度删除导致决策翻转。

这张图应直接说明“为什么普通 final loss 不能区分责任尺度”。

### Figure 2：SDRR 总体方法

只画一条原始 MSHNet 推理路径，训练期从 final conv 分出灰色虚线分析支路。推理支路必须明确打叉删除，以突出 zero overhead。

### Figure 3：机制统计

- per-scale responsibility frequency；
- responsible vs non-pivotal contribution magnitude；
- threshold curve；
- baseline / SDRR paired seed plot。

### Table 1：主结果

三个数据集，IoU / Pd / Fa，mean ± std。

### Table 2：归责对照

M0–M5，只突出 SDRR 是否超过 matched controls。

### Table 3：复杂度

参数、FLOPs、FPS 与 baseline 完全一致；训练显存和训练时长可单独报告。

---

## 15. 7 月 21 日前的执行计划

AAAI-27 官方时间：

- **Abstract：2026-07-21 23:59 UTC-12**；
- **Full paper：2026-07-28 23:59 UTC-12**；
- **Supplementary/code：2026-07-31 23:59 UTC-12**。

### 7 月 12 日

- 冻结 SDRR 方法，不再开新结构分支；
- 建立 `mshnet_canonical.py`；
- 删除 baseline 中的 `decidability_head` 污染；
- 完成 exact reconstruction / deletion equivalence tests；
- 固定 experiment manifest。

### 7 月 13–15 日

并行启动：

- NUAA canonical vs SDRR 三种子复验；
- NUDT canonical vs SDRR 三种子；
- IRSTD-1K canonical vs SDRR 三种子。

所有 paired runs 使用同一 parent checkpoint 分支。不要等待一个数据集全部完成后再启动下一个。

### 7 月 15–17 日

优先完成三个机制对照：

1. matched random event；
2. same-pixel random-scale；
3. magnitude-matched non-pivotal。

GPU 不足时，先在 NUAA 和 NUDT 各完成三种子 matched random；其余对照可在全文截止前补齐。

### 7 月 17–18 日

- 汇总三数据集 paired deltas；
- 检查 PD/FA 和阈值曲线；
- 生成 Figure 1/2 草图；
- 冻结最终标题；
- 冻结摘要中的结果句。

### 7 月 19–20 日

- 完成摘要、Introduction、Method 初稿；
- 检查匿名性、topic、TL;DR；
- 复核摘要中的每个结果数字；
- 不再改变研究问题和模型定义。

### 7 月 21 日

提交非 placeholder 摘要。

### 7 月 22–27 日

- 补齐全部 controls；
- 完成 SOTA 表、可视化、限制和附录；
- 整理匿名代码和配置；
- 7 月 27 日完成全文冻结，避免卡在最终 deadline。

---

## 16. Submission-quality GO Gate

摘要冻结前至少满足：

1. canonical baseline 已物理隔离并通过 identity tests；
2. SDRR 在 NUAA 的严格三种子正均值可复现；
3. NUDT、IRSTD-1K 至少已有可核验的多种子趋势；
4. 至少一个 matched random control 完成三种子；
5. SDRR 的收益不只出现在单一 `0.5` 阈值；
6. 不存在系统性 PD 崩塌或 FA 爆炸；
7. 所有方法使用同一套冻结超参数；
8. 零参数和零推理开销已由脚本验证。

全文主张的最终门槛：

- 至少两个数据集的三种子 paired mean IoU 为正；
- SDRR 优于主要 matched controls；
- 结果包含全部 seeds，不做 cherry-picking；
- 最佳工作点和固定阈值结果都报告；
- official test 未参与调参。

---

## 17. 风险与固定处置

### 风险 1：责任事件过少

处置：保留 `start=250` 的成熟决策阶段；记录 event count，而不是提高 loss 或扩大背景区域。不要临时调大 `λ`。

### 风险 2：某 seed 全背景塌缩

SDRR 依赖已有正决策，因此不能从零正预测恢复。处置是先确保 canonical training 的确定性和稳定性；塌缩 seed 必须计入结果，不能删除。

### 风险 3：提升来自稀疏随机扰动

处置：matched random event + gradient-budget matching。

### 风险 4：提升只是选中了大贡献

处置：magnitude-matched non-pivotal control。

### 风险 5：提升只是阈值校准

处置：报告 `0.3–0.7` threshold curves 和 AUC-like summary。

### 风险 6：审稿人认为只是 leave-one-out

回应重点：

- 不是监督全部 leave-one-out outputs；
- 不要求删除输出拟合完整标签；
- 只使用决策翻转事件做最小、稀疏责任分配；
- 贡献来自部署 fusion 的精确分解；
- matched controls 验证尺度身份和 pivotal property。

### 风险 7：审稿人认为只是 hard-negative mining

回应必须依赖实验，而非文字：M2、M3、M4、M5 对照共同区分 generic hard negative 与 scale-specific pivotal attribution。

---

## 18. 从现在起禁止继续做的事情

- 不再增加 attention、transformer、PConv、动态卷积或额外 decoder；
- 不把 LLoss 改写、boundary loss、focal loss 与 SDRR 堆成三四个组件；
- 不恢复已失败的 TCDS、TFDS、coalition supervision 作为主方法；
- 不对四个删除输出全部加 segmentation loss；
- 不学习新的 scale weights 或 fusion gate；
- 不按数据集调整 `λ/start/ramp/kernel`；
- 不用 official test 选 epoch 或阈值；
- 不隐藏负 seed 或塌缩 seed；
- 不宣称 causal identification；
- 不把“零推理开销”写成“零训练开销”；
- 不在 baseline 中保留未使用的额外 head。

---

## 19. 一句话论文定位

> **SDR-MSHNet does not add another multi-scale module; it makes the existing MSHNet fusion accountable by exactly identifying and refining the native scale contribution whose deletion flips an erroneous background decision.**

中文：

> **SDR-MSHNet 不再增加多尺度模块，而是对 MSHNet 已有融合进行精确追责：只校正那个一旦删除就会使错误背景正决策发生翻转的原生尺度贡献。**

---

## 20. 参考资料

1. [项目仓库：Arialliy/dea3090](https://github.com/Arialliy/dea3090)
2. [Liu et al., Infrared Small Target Detection with Scale and Location Sensitivity, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html)
3. [AAAI-27 Main Conference Timetable](https://aaai.org/conference/aaai/aaai-27/)
4. [AAAI-27 Paper Modification Guidelines](https://aaai.org/conference/aaai/aaai-27/paper-modification-guidelines/)
5. [Yang et al., Pinwheel-shaped Convolution and Scale-based Dynamic Loss, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32996)
6. [Rahman and Marculescu, LoMix: Learnable Weighted Multi-Scale Logits Mixing, NeurIPS 2025](https://arxiv.org/abs/2510.22995)
7. [Yuan et al., Seeing Through the Noise: NS-FPN, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.pdf)
8. [Li and Zhuo, Revisiting the Scale Loss Function and Gaussian-Shape Convolution, 2026](https://arxiv.org/abs/2604.09991)
9. [Boosting Infrared Small Target Detection via Logit-Domain Contrast and Adaptive Shape Refinement, 2026](https://arxiv.org/abs/2607.01555)
10. [仓库内部阶段复核与 SDRR 三种子记录](https://github.com/Arialliy/dea3090/blob/main/MSHNet_TCDS_TFDS_%E7%AC%AC%E4%BA%8C%E8%BD%AE%E9%98%B6%E6%AE%B5%E5%A4%8D%E6%A0%B8.md)

---

## 21. 最终冻结清单

```text
[ ] canonical MSHNet 独立文件已建立
[ ] decidability_head 已从正式 baseline 物理删除
[ ] strict=True checkpoint identity 已验证
[ ] exact reconstruction test 已通过
[ ] direct zero-channel deletion equivalence 已通过
[ ] SDRR hyperparameters 已冻结
[ ] NUAA 三种子已复验
[ ] NUDT 三种子已完成
[ ] IRSTD-1K 三种子已完成
[ ] all-safe fused-logit control 已完成
[ ] same-pivotal-pixel fused control 已完成
[ ] matched random control 已完成
[ ] same-pixel random-scale control 已完成
[ ] magnitude-matched non-pivotal control 已完成
[ ] threshold curves 已生成
[ ] 参数/FLOPs/FPS identity 已验证
[ ] 摘要结果数字已替换，不含 placeholder
[ ] 7 月 21 日前标题/TL;DR/摘要已冻结
```

**本方案从现在起作为唯一主线。后续工作是实现、复验、对照和写作，不再进行新的模型方向搜索；但“尺度责任归因有效”的论文主张只有在第 23 节的机制门槛通过后才可使用。**

---

## 22. CCFA 严格 idea review（2026-07-12，Standard mode）

### 22.1 规范化研究对象

- **目标 venue**：AAAI-27 Main Technical Track；主文 7 页，参考文献可延至第 9 页；摘要截止 2026-07-21，全文截止 2026-07-28。
- **当前阶段**：late idea / early evidence。代码与可复现性基础较成熟，但 official train/test、IRSTD-1K 和完整 matched controls 尚未结束。
- **研究问题**：多尺度加性融合只监督各尺度或最终聚合输出，无法把一个错误背景正决策的校正责任精确分配给原生尺度。
- **根本缺口**：deep supervision、动态 loss、logit mixing 与一般 hard-negative loss 都能改变总体优化，但没有利用部署 fusion 的精确仿射结构回答“删除哪个原生输入会使当前决策翻转”。
- **方法假设**：final fusion 必须可写成 `z=b+Σc_i`；SDRR 只适用于这种可精确分解的加性融合。
- **预期证据**：正确尺度、同像素错误尺度、非 pivotal 尺度、普通背景抑制在相同训练轨迹和相近梯度预算下的配对比较。

### 22.2 public-safe 最近工作检索与 novelty delta

本轮只使用通用公开关键词检索（infrared small target detection、multi-scale logits、false-alarm optimization、deletion attribution），未把仓库私有方法描述提交到外部检索服务。

| 最近工作/概念 | 最接近之处 | 仍未覆盖的 delta | 对 SDRR 的风险 |
|---|---|---|---|
| [MSHNet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) | 多尺度 side outputs、简单融合、SLS loss | 不做 final affine fusion 的原生贡献删除与责任分配 | baseline 本身已经强调尺度敏感，新增工作不能只写“更好利用尺度” |
| [PConv + SD Loss, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32996) | 依据目标尺度动态协调 scale/location 优化 | 不定位单个错误决策的 pivotal fusion source | 动态 loss 已不新，SDRR 必须突出 exact decision attribution |
| [LoMix, NeurIPS 2025](https://nips.cc/virtual/2025/poster/119650) | 训练期多尺度 logits 操作、测试时可移除 | 学习多尺度 mixing；不追责部署 fusion 的原生尺度 | “训练期、零推理开销、多尺度 logits”均不是独立新意 |
| [SCR-Guided Difficulty-Aware Optimization, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/PBVS/papers/Sevim_SCR-Guided_Difficulty-Aware_Optimization_for_Infrared_Small_Target_Detection_CVPRW_2026_paper.pdf) | 训练期 difficulty-aware objective，保持 MSHNet inference graph | 不利用精确 fusion decomposition 做 source-specific deletion | “只改训练、不改推理”也不是独立新意 |
| [AC-SLSIoU, 2026](https://arxiv.org/abs/2607.01555) | logit margin、边界/虚警抑制、零推理开销 | 全局 hard-negative/边界约束，不识别 native pivotal scale | 若 controls 失败，SDRR 会退化成更复杂的虚警 loss |
| [Removal-based attribution, 2019](https://arxiv.org/abs/1910.04256) | 删除输入并观察预测变化的一般归因思想 | 未研究 IRSTD 加性尺度 fusion 的错误决策训练目标 | deletion/counterfactual 本身不是新概念，不能宣称发明删除归因 |

**可保留的 novelty delta**：SDRR 的候选创新不是 deletion、training-only 或 zero-overhead 中任一单点，而是三者在一个可精确验证的训练目标中的耦合：对部署时原生加性 fusion 做无代理的精确分解，仅在安全背景上的 positive-to-nonpositive deletion flip 发生时，把梯度路由给对应的 native source。

### 22.3 十维严格评分

评分为当前证据状态，不是最终论文评分；置信度与分数分开报告。

| 维度 | 权重 | 分数 / 5 | 严格理由 |
|---|---:|---:|---|
| 问题重要性 | 12% | 4.0 | IRSTD 的虚警与多尺度融合责任是真问题，但子领域较窄 |
| 新颖性 | 14% | 3.0 | exact native-source attribution 有差异；deletion、训练期 loss、零开销均已有邻近概念 |
| 概念创新 | 12% | 3.5 | 从“学习新融合”转为“问责已有融合”较清晰，但仍需 controls 证明不是措辞重包装 |
| 技术可靠性 | 14% | 3.0 | 代数分解可靠；离散稀疏事件、归一化和晚期梯度扰动仍有稳定性风险 |
| 简洁性 | 8% | 5.0 | 单一训练目标、零参数、部署图完全不变 |
| 可实现性 | 8% | 5.0 | clean baseline、exact tests、checkpoint branching 已完成 |
| 实验证服力 | 10% | 3.0 | 两数据集内部趋势为正，但 official test、完整 controls、跨 backbone 性能未完成 |
| AAAI 适配度 | 8% | 3.0 | 方法干净；若只作为 MSHNet/IRSTD 小修，存在“窄且增量”的 AC 风险 |
| 时效性 | 6% | 4.0 | 2025–2026 多尺度 logits、虚警优化活跃，问题及时 |
| 当前接收潜力 | 8% | 3.0 | 有可救援核心，但现有证据不足以冻结强摘要 |
| **加权总分** | **100%** | **3.56 / 5** | **当前结论：Pivot-with-rescue-route，不建议按现有强主张直接投稿** |

- **评审置信度**：`3/5`。代码和内部证据置信度较高；完整 prior-art 覆盖与 official 结果仍不完整。
- **阶段就绪度**：`3.0/5`。可继续实验，但尚未达到摘要结果冻结条件。
- **发展潜力**：`4.1/5`。若正确尺度和 pivotal property 的 matched controls、跨数据集/跨 backbone 证据成立，可进入 `Revise` 区间；这不是接收概率承诺。

### 22.4 五位独立专家意见

1. **领域专家**：虚警问题重要，但 IRSTD 体量小；主文必须说明“加性多源融合责任”为何可推广，而不仅是 SIRST 上的 MSHNet trick。
2. **方法专家**：`z=b+Σc_i` 与删除翻转定义漂亮、可验证；但事件稀疏且 assignment 不可微，收益可能只是改变晚期优化轨迹。
3. **实验专家**：M2/M3/M4/M5 是主证据，不是附属消融；official test 不能用于 seed、epoch 或阈值选择。
4. **AC/venue 专家**：AAAI-27 明确看重 significance、novelty、soundness 与 reproducibility；只报小幅均值提升很难抵消窄领域风险。
5. **prior-art 专家**：LoMix 已占据“训练期多尺度 logits、零推理开销”邻域，一般 deletion attribution 也成熟；差异必须落在 exact native-source responsibility 及其必要性证据。

### 22.5 最危险的反例

若 M2（同责任像素、直接约束 fused logit）或 M3（幅值匹配但非 pivotal）达到或超过 SDRR，则当前结果最多证明“稀疏选择的安全背景正决策有助于晚期稳定”，不能证明尺度身份或 deletion-pivotal responsibility 是有效机制。此时必须缩窄主张，不能通过增加 attention、boundary loss 或动态卷积来掩盖。

---

## 23. CCFA idea optimization：冻结后的论文级蓝图

### 23.1 优化后的核心定位

论文上位问题统一为 **Accountable Additive Fusion**，SDRR 是其在 MSHNet 上的一个精确训练目标，而不是另一个多尺度模块：

> Aggregate supervision tells an additive fusion what to predict, but not which native source should be corrected for a particular erroneous decision. Because an affine fusion is exactly decomposable, source necessity can be observed by deletion rather than approximated by a learned attribution module.

中文：

> 聚合监督告诉加性融合“预测什么”，却不告诉它“当前错误应校正哪个原生来源”。仿射融合可被精确分解，因此来源必要性可以由删除翻转直接观测，无需再学习一个归因模块。

### 23.2 一条机制、三项贡献

1. **Exact source decomposition**：将已经部署的加性 fusion 精确展开为 bias 与任意 `S≥1` 个 native contributions，并验证 reconstruction 与 direct deletion 等价。
2. **Decision-pivotal responsibility objective**：只在安全背景、正决策且删除单一 source 会翻转决策时产生事件；assignment detach，只校正对应 source contribution。
3. **Mechanism-centered evidence**：以相同起点、像素、事件数和梯度预算的 controls 证明收益依赖 source identity 与 pivotal property，而不是新增参数或普通 hard-negative mining。

这三项共享同一个理论对象 `z=b+Σc_i`，不允许再拼接独立 loss 或 feature module。

### 23.3 可写与不可写的 claim

**只有门槛通过后可写：**

- exact、model-internal deletion attribution；
- decision-pivotal native-source refinement；
- training-only、zero learnable parameter、zero inference overhead；
- 在预注册协议下优于相同预算的主要 controls。

**即使结果很好也不可写：**

- causal identification / true cause；
- 普适于任意非加性 attention/transformer fusion；
- 每个 seed 或每个指标都改善；
- state of the art，除非完成同协议、同数据与可核验 SOTA 表；
- 选择最优 official-test seed。

### 23.4 证据门槛与决策树

| Gate | 通过条件 | 未通过时的处置 |
|---|---|---|
| G1：正确尺度必要 | SDRR 在固定 endpoint/预注册 selection 下优于 M2 与有效 M4；M4 不得含 NaN | 删除“scale identity 已证明”，只保留 pivotal-pixel stabilization |
| G2：pivotal property 必要 | SDRR 优于 M3 与 matched-random；报告 event shortage 与梯度匹配质量 | 删除“deletion responsibility”，改写为 sparse hard-negative objective |
| G3：跨数据稳健 | 三个 official 数据集用固定 seed 20260713；至少两个方向为正且 PD/FA trade-off 可接受 | 不宣称 general robustness，缩成 dataset-specific diagnostic |
| G4：跨模型资格 | 至少一个具有原生 affine side fusion 的第二 backbone 完成 paired training | 只宣称 MSHNet specialization，不写通用 framework |
| G5：数值可靠 | 所有 runs 无 NaN/Inf；共同前缀、split hash、checkpoint hash、固定 endpoint 可审计 | 该 run 作废并从共同前缀重跑，不做事后解释 |

### 23.5 当前最小实验包

优先级固定为：

1. 完成 M1/M2/M3/修复后的 M4/M5 与 normalization control；
2. 运行预注册 `seed=20260713` 的 SIRST-v1 official train/test；
3. 同协议完成 NUDT-SIRST 与 IRSTD-1K；
4. UIUNet 或另一原生 affine side-fusion backbone 做 paired training；
5. 报告 best internal validation、固定 epoch、last-20/50、IoU/PD/FA、阈值曲线、event/component coverage；official test 只报告预先固定 checkpoint。

### 23.6 救援路线

- **A 路线（首选）**：G1–G4 通过，论文写成 exact responsibility assignment for accountable additive fusion。
- **B 路线（可接受）**：M2 不如 SDRR、但 M3/M4 证据不稳定，缩窄为 deletion-pivotal region refinement，不强调尺度身份普适性。
- **C 路线（停止 AAAI 强投）**：M2/M3 达到或超过 SDRR，或 official 结果只有一个数据集正向。此时诚实结论是 generic sparse stabilization；不再通过模块堆叠救结果，改为诊断/短文或重新选研究问题。

当前执行 **A 路线**，但摘要与标题中的强责任主张保持条件冻结，直到 G1–G3 有完整结果。

### 23.7 条件标题

- **G1–G4 全部通过**：*Making Additive Fusion Accountable: Exact Scale-Deletion Responsibility for Infrared Small Target Detection*。
- **只通过 MSHNet 专项门槛**：*Rethinking Multi-Scale Fusion in MSHNet via Exact Scale-Deletion Responsibility*。

第一标题强调可推广的研究问题，只能在第二 backbone paired training 成立后使用；否则采用第二标题，避免把单 backbone 证据包装成通用框架。

---

## 24. 完整模型交付与单向冻结流程（2026-07-12）

### 24.1 已完成的完整模型

仓库现已提供独立的 `SDRMSHNet` 类，而不再只依赖训练器中的零散实验开关：

- 实现：`model/sdr_mshnet.py`；
- CLI：`--model-type mshnet --mshnet-variant sdr --fusion-regularizer sdrr`；
- 编码器、middle、解码器、四个 side heads 与 final fusion 全部继承 deterministic canonical MSHNet；
- 参数 key、参数量和默认 forward 与 canonical deterministic MSHNet 完全一致；
- 训练状态一次返回 `side_logits / pred / scale_logits / contributions / deletion_logits / reconstructed`；
- `responsibility_objective(...)` 是唯一新增训练目标；
- 测试部署仍调用默认 forward，不生成责任状态，不增加参数或推理路径。

因此，“完整模型”已具备独立构造、训练、保存、恢复和测试所需的统一边界，而不是模块拼装脚本。

### 24.2 当前冻结边界

| 前向阶段 | 状态 | 后续规则 |
|---|---|---|
| input stem + encoder 0–3 | **冻结** | canonical identity 已通过，不再修改 |
| middle layer | **冻结** | 不增加 transformer/dynamic convolution |
| decoder 3–0 + skip paths | **冻结** | 不再叠加额外 decoder/refinement block |
| four native side heads | **冻结** | 保留原始 SLS supervision 与分辨率 |
| final affine fusion | **冻结部署图** | 仅训练期读取精确贡献，不改 inference forward |
| SDRR objective | **完整** | 固定单一机制，不再拼 boundary/focal/attention loss |

### 24.3 快速闭环验证

已完成 `seed=20260713` 的 2-epoch SIRST-v1 internal-holdout smoke：

- train/val 为 `170/43`，来自官方 213-image train manifest 的固定内部划分；
- epoch 1 成功进入 `counterfactual_responsibility=1` 路径；
- 首个 active batch 记录 `180462` 个责任事件，SDRR raw/weighted loss 为 `1.939452 / 0.096973`；
- 完整 train → backward → checkpoint → validation 流程正常退出；
- 两 epoch 的 IoU 为 0 是极短 warmup 下的预期结果，只证明工程闭环，不作为性能证据。

产物：`repro_runs/sdr_mshnet_complete_smoke_seed20260713_e2/`。

### 24.4 成熟 checkpoint 封装

已将完成 400 epochs 的 `seed=20260713` clean SDRR run 严格封装为完整
`SDRMSHNet` checkpoint：

- 目录：`repro_runs/sdr_mshnet_complete_nuaa_seed20260713_e400/`；
- package audit：`sdr_mshnet_package.json`；
- 参数量 `4,065,513`，新增参数 `0`；
- checkpoint 中 model/optimizer 均对 `SDRMSHNet` 严格加载；
- 默认 forward 与 deterministic canonical MSHNet bit-exact；
- 原始 `weight.pkl` hash 保持不变，封装没有重新训练或改写权重；
- internal-holdout best-IoU 为 `0.7359 / PD 0.9815 / FA 5.3228`，相同 seed
  baseline 为 `0.7250 / 0.9630 / 10.2908`，paired delta 为
  `+0.0109 / +0.0185 / -4.9680`；
- 固定 epoch-399 IoU 仍为 `+0.0020`，但 PD 为 `-0.0556`，因此不能宣称
  所有工作点 Pareto 改善。

上述数字只属于固定 train-manifest internal holdout，不是 official test，也没有因封装而
产生新的性能证据。

### 24.5 从现在起的快速流程

不再逐个想法跑 400 epoch。执行顺序固定为：

1. 完整模型单测与 2-epoch smoke；
2. 一个固定 seed 的短程 paired gate；
3. gate 为正才进入一次正式 400-epoch 训练；
4. 正式模型冻结后再补 reviewer-required controls，不再反向改 encoder/decoder/head；
5. 任一后续修改若要求回改已冻结前段，则视为新模型方向，不混入 SDR-MSHNet。

---

## 25. 顶会立意筛选：从失败的 RCR 到 Responsibility Density Risk（RDR）

### 25.1 为什么要先检验 RCR

旧 SDRR 已能精确识别 deletion-pivotal source，但其 event-mean
`softplus(c_i)` 同时把“错误决策有多严重”和“责任贡献有多大”混在一个 penalty 中；当一个
像素有多个责任尺度时，总辅助梯度预算还会随 responsibility degree 改变。因此审稿人仍可
把收益解释为一种稀疏 contribution regularization，而不是一个完整的 credit-assignment
原理。

### 25.2 单一新机制

保留完全相同的精确责任事件：

\[
r_i(p)=\mathbf 1[z(p)>0]\mathbf 1[z(p)-c_i(p)\le 0]
\mathbf 1[p\in\Omega_{\rm safe}],\qquad
d(p)=\sum_i r_i(p).
\]

令 `sg` 表示 stop-gradient。RCR 在每个 unique responsible pixel 上定义：

\[
\mathcal L_{\rm RCR}
=\frac{1}{|\mathcal P|}\sum_{p\in\mathcal P}\sum_i
\frac{r_i(p)}{d(p)}\left[
\operatorname{softplus}(\operatorname{sg}(z(p)))+
\sigma(\operatorname{sg}(z(p)))
\big(c_i(p)-\operatorname{sg}(c_i(p))\big)
\right].
\]

它不是“SDRR 再加一个 loss”，而是用一个守恒的 first-order surrogate **替换**旧的
`softplus(c_i)`：错误强度仍由 deployed decision `z` 定义，但校正梯度只被路由到精确
责任 source。

### 25.3 两个可验证性质

**Property 1：decision-loss value preservation。** 前向数值严格为：

\[
\mathcal L_{\rm RCR}
=\frac{1}{|\mathcal P|}\sum_{p\in\mathcal P}\operatorname{softplus}(z(p)),
\]

因此 RCR 不通过放大 loss value 获益。

**Property 2：responsibility-conserving gradient routing。** 对 contribution coordinate：

\[
\frac{\partial\mathcal L_{\rm RCR}}{\partial c_i(p)}
=\frac{r_i(p)}{d(p)}\frac{\sigma(z(p))}{|\mathcal P|},
\qquad
\sum_i\frac{\partial\mathcal L_{\rm RCR}}{\partial c_i(p)}
=\frac{\sigma(z(p))}{|\mathcal P|}.
\]

所以非责任尺度的辅助梯度严格为零；无论一个像素有几个责任尺度，总 decision-gradient
预算保持不变。创新对象从“新增背景 penalty”提升为“对已有加性融合进行精确、守恒的
decision-credit routing”。

### 25.4 claim–evidence matrix

| Claim | Reviewer question | 必需证据 | 当前状态 |
|---|---|---|---|
| exact responsibility | 删除归因是否只是近似 attention？ | reconstruction、direct-zero deletion、边界稳定性 | 已完成 |
| value preservation | 是否靠更大 loss 获益？ | 单测逐值等于 responsible-pixel `softplus(z)` | 已完成 |
| gradient conservation | 多责任尺度是否放大更新？ | per-pixel gradient sum identity、非责任梯度为零 | 已完成 |
| optimization benefit | 守恒路由是否优于 canonical？ | 相同 epoch-79 parent 的固定 endpoint paired gate | seed13 首轮通过；旧 SDRR 对照运行中 |
| specificity | 是否任意路由都有效？ | same-pixel random-scale、same-pixel fused controls | 待主 gate 通过后补 |
| generality | 是否只是 MSHNet trick？ | 第二个原生 affine-fusion backbone | 待主 gate 通过后补 |

### 25.5 快速 GO/NO-GO

短程 gate 固定为 SIRST-v1 internal holdout、`seed=20260713`、共同 epoch-79
checkpoint、终点 epoch 119。只有当 RCR 的固定终点 IoU 高于 paired baseline，且没有
出现 NaN、全背景塌缩或不可接受的 PD/FA 退化，才将完整模型从 SDR-MSHNet 升级为
RCR-MSHNet；否则保留已验证的 SDR-MSHNet，不把失败候选堆入最终方法。

**No-fabrication status**：本节只陈述已由代数/单测验证的性质；运行中的性能格不预填数字。

### 25.6 第一层 paired gate 结果

共同 clean epoch-79 checkpoint、相同 resume 数据顺序、固定 epoch119 的真实结果为：

| Variant | IoU | PD | FA/Mpix | 相对 baseline |
|---|---:|---:|---:|---|
| Canonical MSHNet | 0.6281 | 0.9259 | 98.6498 | — |
| RCR | **0.6566** | 0.8889 | **85.8750** | `ΔIoU +0.0285 / ΔPD -0.0370 / ΔFA -12.7748` |

结论：RCR 通过“固定终点 IoU 为正且 FA 下降”的第一层 gate，证明守恒路由并非只有数学
性质；但 PD 下降 `0.0370`，尚不能升级最终模型。当前正用同一 parent 补跑旧
SDRR-event 与 SDRR-safe-density，只有 RCR 在同协议比较中仍占优才继续。

### 25.7 旧方法 comparator 与唯一修正机会

同一 gate 的 comparator 已完成：

| Variant | IoU | PD | FA/Mpix |
|---|---:|---:|---:|
| Canonical MSHNet | 0.6281 | **0.9259** | 98.6498 |
| SDRR-event | 0.6336 | 0.8889 | 65.6483 |
| SDRR-safe-density | **0.6744** | 0.9074 | **41.8730** |
| RCR-unique-pixel | 0.6566 | 0.8889 | 85.8750 |

RCR-unique-pixel 三项均弱于 SDRR-safe-density，第二层 gate **FAIL**，不得进入最终模型。
诊断表明其 unique-pixel denominator 在责任事件极稀疏时仍使批级校正偏强。现只允许一次
同原理修正：将 RCR denominator 固定为安全背景面积，使总梯度预算随责任密度衰减；该
候选记为 `RCR-density`。若其固定终点不能超过 `0.6744 / 0.9074 / 41.8730` 的主要
trade-off，则停止 RCR 路线，最终保留 SDRR-safe-density。

RCR-density 的实际终点为 `0.5550 / 0.9259 / 215.0425`，同时低于 paired baseline
IoU、显著增加 FA，也远低于 SDRR-safe-density。故唯一修正机会同样 **FAIL**；RCR
路线正式终止并保留为负结果，不进入完整模型。当前最终候选冻结为：

> **SDR-MSHNet-D：exact scale-deletion responsibility + safe-background-density normalization。**

这仍是一个目标函数：在安全背景上识别 exact pivotal native contribution，并将其
`softplus(c_i)` 总量除以安全背景面积；没有加入 RCR、attention、额外 head 或第二个 loss。

### 25.8 最终单一机制：Responsibility Density Risk

`SDRR-safe-density` 的胜出不是一个需要继续保留的“归一化技巧”。把其分子和分母展开后，
它对应一个更完整、可检验的风险定义。令安全背景域为
`Ω_safe`，责任事件总数为 `N_resp=Σ_{p,i}r_i(p)`，则：

\[
\mathcal L_{\rm RDR}
=\frac{1}{|\Omega_{\rm safe}|}
\sum_{p\in\Omega_{\rm safe}}\sum_i
r_i(p)\operatorname{softplus}(c_i(p)).
\]

该量严格分解为：

\[
\mathcal L_{\rm RDR}
=\underbrace{\frac{N_{\rm resp}}{|\Omega_{\rm safe}|}}_{
\text{responsibility-event prevalence}}
\cdot
\underbrace{\frac{1}{N_{\rm resp}}
\sum_{p,i}r_i(p)\operatorname{softplus}(c_i(p))}_{
\text{conditional responsible-evidence severity}}.
\]

因此最终方法不再表述为“event loss + density normalization”，而是一个单一的
**责任密度风险**：同时度量错误责任证据在安全背景中的**发生频率**和**条件严重度**。
旧 event-mean SDRR 只优化第二项，忽略第一项；RCR 则在人为保持 unique-pixel 梯度预算时
破坏了稀疏责任事件应有的密度衰减。短程结果中 RDR 的优势与这一诊断一致，但这里只作机制
解释，不把一次短程结果当成普适定理。

最终模型命名冻结为：

> **RDR-MSHNet：Making Additive Fusion Accountable with Responsibility Density Risk。**

它仍只有一条训练目标和一个理论对象：从部署时的精确仿射融合中识别 deletion-pivotal
native contribution，并估计其在安全背景域上的风险密度。它不增加 head、attention、
feature block、可学习参数或推理分支。`responsibility_density_risk` 与已胜出的
`SDRR-safe-density` 数值及梯度严格等价，所以正在运行的官方 density 分支是该最终目标的
有效性能实验，而不是事后更换训练算法。

代码审计已验证：直接计算值与上述因式分解误差为零（数值容差内），零责任事件时损失和
梯度均安全为零；完整仓库回归为 `251 passed`。在官方与跨数据集 gate 完成前，
**RDR-MSHNet 仍是最终候选，不称成功模型**。

---

## 26. 分段验证与冻结台账

每一部分只有在对应证据通过后才冻结；`代码通过`、`机制通过` 和 `性能通过` 不互相替代。

| 部分 | 要解决的问题 | 验收证据 | 当前状态 | 冻结结论 |
|---|---|---|---|---|
| Canonical baseline | baseline 是否被历史实验 head 污染 | 物理隔离、strict state/optimizer load、参数/forward identity | **PASS** | encoder/decoder/side/final 部署图冻结 |
| 完整模型接口 | 是否只是 trainer 中的零散开关 | 独立 `SDRMSHNet`、bit-exact default forward、RDR objective、2-epoch active smoke | **PASS** | 完整模型工程边界冻结；待成功 gate 后冻结 RDR 命名包 |
| Exact responsibility | 是否真正识别“哪个尺度使错误决策成立” | reconstruction、direct deletion、`z>0,z-c_i≤0` 事件 | **PASS** | deletion assignment 定义冻结 |
| RCR 数学机制 | 是否保持 decision loss 和总梯度预算 | value identity、非责任梯度零、per-pixel gradient-sum identity | **PASS，但性能路线 REJECTED** | 作为负结果保留，不进入最终模型 |
| RCR 运行机制 | 真实训练是否触发且不破坏数值 | active event、routing error=0、无 NaN/Inf | **PASS，但性能路线 REJECTED** | 不再修改 RCR |
| RDR 风险定义 | 是否同时刻画责任事件频率与严重度 | density-risk factorization identity、零事件安全、完整回归测试 | **PASS** | 单一最终目标冻结，不再修改 |
| 短程性能 | 是否比相同 parent 的 baseline 与旧方法更好 | seed13、epoch79 共同起点、固定 epoch119 | **RDR WIN** | 最终候选冻结为 RDR-MSHNet |
| Official 性能 | 完整训练后是否真实提升 | 完整 train、test 仅 epoch399 读取、paired IoU/PD/FA | **RDR FAIL（NUAA）** | baseline `0.7074/0.9506/28.7350`；RDR `0.6853/0.9392/53.1919`，不得称成功 |
| 跨数据集稳健性 | 是否只对 SIRST-v1 有效 | NUDT-SIRST、IRSTD-1K 固定协议 | **NUDT 短程 PASS，但不足以救 NUAA official** | NUDT epoch119：`0.6636/0.9305/54.0368 → 0.6992/0.9465/44.2849` |

### 26.1 最终成功条件

最终模型必须同时满足：

1. **问题解决**：存在可核验的 responsibility events，且辅助梯度只到责任 contribution；
2. **短程 gate**：`ΔIoU@119 > 0`，并且不能同时出现 `ΔPD<0` 与 `ΔFA>0`；
3. **official gate**：`ΔIoU@399 > 0`；若 PD 下降，则 FA 必须下降且 `|ΔPD|≤0.02`；
4. **跨数据集 gate**：三个数据集至少两个固定协议方向为正；
5. **创新 gate**：最终只保留 exact assignment 所定义的 Responsibility Density Risk，不叠加独立模块；
6. **复现 gate**：完整配置、split hash、parent hash、全部 seeds 和失败结果保留。

任何一项未满足，都只能称候选模型，不能写成“成功解决原问题并提升性能”。

RDR 的官方结论现已明确：它在 NUDT 短程 gate 上三项改善，但在 NUAA 唯一一次
leakage-free official test 上 IoU、PD、FA 全面弱于 canonical baseline。因此 RDR 只保留为
训练目标研究与后续结构模型的可选对照，**不再作为 AAAI 最终模型主贡献**。

---

## 27. 结构主线重启：Orthogonal Scale Ownership（OSO）

### 27.1 为什么必须从 loss-only 转为结构设计

前述 official 结果否定了“只靠责任 loss 就能稳定解决 MSHNet 融合缺陷”的强主张。
更根本的问题是结构本身：四个 side heads 都在同一个完整 mask 空间里表达证据，final
`Conv(4→1)` 又允许这些来源任意重叠。即使训练时能识别某个 deletion-pivotal source，
部署结构仍没有阻止多个尺度重复表达同一空间模式。

因此新的核心问题是：

> **能否在结构上使每个原生尺度只拥有一个互补输出子空间，从而让多尺度证据先天可辨识，而不是在任意混合后再追责？**

### 27.2 文献设计原则与排除项

标准检索报告位于
`literature-search-20260712-mshnet-structural-fusion/`。结构路线明确排除：

- PConv/InvDet 已覆盖的低层几何或可逆 encoder 改造；
- NS-FPN/DEFANet 已覆盖的 frequency、wavelet、edge 双分支；
- SCTransNet/DNANet/UIU-Net 已覆盖的 attention、dense/nested decoder；
- OSCAR/Laplacian Pyramid/Residual Pyramid 已覆盖的普通 coarse-to-fine refinement；
- LoMix 已覆盖的 learnable multi-scale logit mixing。

OSO 借鉴的是这些顶会工作的**设计方式**：以一个明确结构失配为起点，用一个数学原则
重写关键计算，而不是借用其模块。

### 27.3 单一结构机制

在全分辨率输出空间定义对齐的 block-average 正交投影：

\[
P_k x=\operatorname{Repeat}_{2^k}
\left(\operatorname{AvgPool}_{2^k}(x)\right),\quad k\in\{1,2,3\}.
\]

这些投影满足 `P_jP_k=P_max(j,k)`。构造四个互补 ownership operators：

\[
Q_0=I-P_1,\qquad Q_1=P_1-P_2,\qquad
Q_2=P_2-P_3,\qquad Q_3=P_3.
\]

对 MSHNet 原生 fusion kernel 的四个卷积贡献 `c_i=W_i*s_i`，新部署输出为：

\[
z_{\rm OSO}=b+Q_0c_0+Q_1c_1+Q_2c_2+Q_3c_3.
\]

这不是四个模块，而是一个**互补子空间合成算子**，直接替换 unconstrained late fusion。
encoder、bottleneck、decoder、side heads、`final.weight/bias` 均保持原样。

### 27.4 三个结构性质

1. **Completeness**：`Σ_i Q_i=I`。四个所有权带不丢失完整输出空间。
2. **Exclusive ownership**：`Q_iQ_j=0 (i≠j)`，所以任意两个尺度贡献内积严格为零，不能在同一子空间重复强化。
3. **Checkpoint continuity**：仍使用 MSHNet 原生 `final.weight/bias`，参数量、参数 key 和 Adagrad state 完全相同；共同 checkpoint 可 strict resume，差异只来自 fusion operator。

与普通 Laplacian refinement 的区别必须写清：OSO 不预测 residual label、不用 coarse proposal
引导 fine head，也不新增 reconstruction decoder；它把**已有四个原生来源**一一绑定到互补
子空间，使来源身份在结构上可辨识。

### 27.5 由前到后的冻结规则

| 前向阶段 | 当前处理 | 验证 | 状态 | 后续规则 |
|---|---|---|---|---|
| input stem | 保留 canonical `1×1 Conv` | state/forward identity | **PASS** | 冻结，不再修改 |
| encoder 0–3 | 保留 canonical residual+CA/SA blocks | strict checkpoint、参数 identity | **PASS** | 冻结，不加 PConv/wavelet/transformer |
| bottleneck | 保留 canonical middle layer | strict checkpoint | **PASS** | 冻结 |
| decoder 3–0 | 保留 canonical concat+bilinear path | cold path bit-exact、真实 backward | **PASS** | 冻结，不加 attention/refiner |
| side heads | 保留四个 native `1×1 Conv` | 参数 key identity | **PASS** | 当前冻结；只有 fusion 通过后才评估监督语义 |
| final fusion | 用 OSO 替换 arbitrary overlapping mix | projector identities、owned reconstruction、paired endpoint | **CODE PASS / PERFORMANCE RUNNING** | 当前唯一允许修改的阶段 |

前五段通过后，不允许为了补救 OSO 结果回改 encoder/decoder。若性能 gate 失败，只能：

1. 诊断当前 projection/fusion 的结构假设；
2. 在 final head/fusion 范围内作一次同原理修正；
3. 仍失败则拒绝 OSO，另立新结构方向，不能把 attention/frequency 模块堆回去。

数据协议同样冻结：仓库数据根目录仅为 `datasets/`，正式实验只读取
`datasets/<name>/img_idx/train_<name>.txt` 与 `test_<name>.txt`。现有 NUAA、NUDT、
IRSTD-1K manifest 数量分别为 `213/214`、`663/664`、`800/201`。NUDT 目录中的其他
辅助文本不进入本项目协议。根据最终比较协议，**不再从 train manifest 划出 validation**：
每次正式运行都使用完整 train manifest 拟合，并只在预先固定的训练终点读取完整 test
manifest。本文前面出现的 internal-holdout 数字仅为历史开发日志，不进入论文主结果、
不作为与 DNANet/UIU-Net/MSHNet 的横向对比证据。

### 27.6 当前可核验证据与性能 gate

代码：`model/orthogonal_scale_ownership.py`。CLI：
`--model-type mshnet --mshnet-variant oso`。目前针对性测试 `71 passed`，覆盖：

- `P_k` 幂等与嵌套；
- 四个 `Q_i` 对同一输入的恒等重构；
- 任意两 ownership bands 的内积为零；
- MSHNet 参数量与 state keys 完全相同；
- strict baseline checkpoint load；
- full-resolution forward/backward 有限；
- cold single-head path 与 canonical bit-exact；
- 非 8 对齐输入 fail closed。

首轮性能 gate 固定为 NUAA internal holdout、`seed=20260713`、共同 epoch79 parent、终点
epoch119；baseline 已冻结为 `0.6281 / 0.9259 / 98.6498`。OSO 不使用 RDR 或其他新增 loss。

通过条件：`ΔIoU>0` 且不能同时 `ΔPD<0、ΔFA>0`。通过后冻结 OSO fusion，再按前向顺序
进入“监督语义/责任训练”阶段；失败则只修改 final fusion，不触碰已冻结前段。

### 27.7 OSO gate 结果与冻结处置

OSO 固定终点结果为 `0.4625 / 0.8889 / 172.8147`，相对 baseline 的变化为：

```text
ΔIoU = -0.1656
ΔPD  = -0.0370
ΔFA  = +74.1649
```

故性能 gate **FAIL**。训练无 NaN、梯度与 checkpoint 均正常，失败指向方法假设而非工程
故障：native scale identity 并不等价于固定空间频带；硬投影会删除每个原生尺度中有用的
cross-band target evidence。产物已写入
`repro_runs/gate_oso_nuaa_seed20260713_e120/REJECTED_CANDIDATE.json`。

冻结处置：input/encoder/bottleneck/decoder/side heads 继续保持不变；OSO 不进入最终模型，
不追加 attention 或 frequency 分支。当前只允许在 final-fusion 阶段继续设计。

---

## 28. Final-fusion 第二轮：Deletion-Stable Fusion（DSF）

### 28.1 从负结果得到的新结构要求

OSO 说明“来源可辨识”不能靠预设频带替代。真正与原问题一致的结构要求是：

> 一个可靠的多尺度正决策不应完全依赖任意单一尺度；若删除某一尺度就使决策崩溃，fusion 应当在部署时直接暴露并优化这种脆弱性。

这一要求不限制各尺度表达什么空间频率，因而保留所有 native evidence。

### 28.2 单一鲁棒融合算子

仍使用 MSHNet 原生仿射贡献：

\[
z=b+\sum_{i=0}^{3}c_i,\qquad c_i=W_i*s_i.
\]

构造四个精确 leave-one-scale-out coalition logits：

\[
d_i=z-c_i.
\]

DSF 不再直接部署脆弱的 `z`，而部署归一化 smooth worst-coalition：

\[
z_{\rm DSF}
=-\log\left(\frac{1}{4}\sum_{i=0}^{3}\exp(-d_i)\right).
\]

它是一个 final fusion operator，不是辅助 loss、attention gate 或额外网络。归一化保证四个
`d_i` 相等时输出严格等于该共同值。

### 28.3 内生责任路由

DSF 对 coalition logit 的导数为：

\[
\frac{\partial z_{\rm DSF}}{\partial d_i}
=\frac{\exp(-d_i)}{\sum_j\exp(-d_j)}.
\]

四个导数非负且和为 1；最小、最脆弱的 deletion logit 自动得到最大更新责任。因此 DSF
把原先训练期的 scale-deletion diagnostic 变成了部署结构的鲁棒优化规则，不再需要 RDR
才能回答“当前应该优先修复哪个尺度联盟”。

### 28.4 与近邻工作的边界

- 不同于 OSCAR：没有 high-level proposal→low-level refinement cascade；
- 不同于 LoMix：不搜索或学习多尺度混合组合；
- 不同于 generic ensemble robustness：四个 source 是同一网络中可精确代数删除的 native
  scale contributions，且目标是 dense decision stability；
- 与一般 single-source robustness 的关系需要明确引用，不能宣称首次研究 source failure；
- 核心 delta 是 **exact internal scale deletion + normalized worst-coalition fusion**。

### 28.5 工程与性能状态

代码：`model/deletion_stable_fusion.py`；CLI：`--mshnet-variant dsf`。针对性测试
`79 passed`，已验证 equal-coalition identity、worst-deletion simplex gradient、
single-source dominance discount、参数/checkpoint identity、真实 forward/backward 和 cold-path
bit identity。

DSF 正从与 baseline、OSO 完全相同的 epoch79 parent 训练至固定 epoch119；不使用 RDR 或
任何新 loss。只有通过同一性能 gate 后才冻结 final fusion 并继续设计后续监督部分。

### 28.6 DSF gate 结果与拒绝原因

DSF 固定终点为 `0.5750 / 0.8889 / 162.8787`，相对同 parent baseline：

```text
ΔIoU = -0.0531
ΔPD  = -0.0370
ΔFA  = +64.2289
```

性能 gate **FAIL**。失败不是数值或实现故障，而是 pixel-wise worst coalition 过于保守：
它在所有像素都削弱最差联盟，训练会通过抬升全局正证据进行补偿，反而增加背景误警。
因此 DSF 作为可复现负对照保留，不进入最终结构。

---

## 29. Final-fusion 第三轮：Decision-Conditional Deletion Fusion（DCDF）

DCDF 只在仿射预测为正且 deletion coalition 表明脆弱时施加平滑修正，试图避免 DSF 对
所有像素一刀切。代码位于 `model/decision_conditional_deletion_fusion.py`，仍不新增参数，
不改变已冻结的 stem、encoder、bottleneck、decoder 和 side heads。

固定 gate 结果为 `0.5658 / 0.8889 / 169.9758`：

```text
ΔIoU = -0.0623
ΔPD  = -0.0370
ΔFA  = +71.3260
```

性能 gate 再次 **FAIL**。这说明问题不只是“在哪些像素抑制”，而是任何直接降低 logit
均可能被优化器通过全局抬升证据抵消。由此得到下一轮不可违反的结构约束：修正必须只在
空间位置之间重新分配冲突，不能改变每张图的全局 logit 质量。

---

## 30. Final-fusion 第四轮：Counterfactual Conflict-Field Diffusion（CCFD）

### 30.1 强问题与单一结构机制

MSHNet 的核心缺陷不是“缺少一个增强模块”，而是 affine fusion 无法区分稳定共识与
单尺度删除后崩溃的局部冲突。CCFD 将两者之差定义为可观测的反事实冲突场：

\[
z=b+\sum_i c_i,\qquad
z_{\rm rob}=-\log\left(\frac14\sum_i e^{-(z-c_i)}\right),\qquad
r=z_{\rm rob}-z.
\]

在 `r` 上只施加一个全尺度共享的八邻域零直流 stencil：

\[
u=\sum_{\delta\in\mathcal N}\theta_\delta
\big[T_\delta(r)-r\big],\qquad
\bar u=u-\operatorname{mean}_{h,w}(u),\qquad
z_{\rm CCFD}=z+\bar u.
\]

这是一条结构方程，不是 attention、frequency branch、refiner 与 loss 的堆叠。四个精确
deletion coalitions 只负责定义冲突场；唯一可学习的新结构是共享 stencil 的 8 个标量。

### 30.2 可证性质

1. **Baseline embedding**：`theta=0` 时 `z_CCFD=z`，完整 canonical MSHNet 精确嵌入；
2. **First-order trainability**：零初始化不阻断 `theta` 的一阶梯度；
3. **Global evidence conservation**：`sum_{h,w}(z_CCFD-z)=0`，禁止 DSF/DCDF 式全局正证据补偿；
4. **Local conflict transport**：只有删除冲突的空间差分可以改变决策；
5. **Minimal delta**：总参数由 `4,065,513` 增至 `4,065,521`，仅增加 8 个参数。

代码位于 `model/counterfactual_conflict_diffusion.py`，CLI 为 `--mshnet-variant ccfd`。
全部 `284 passed`；真实 epoch79 checkpoint 与 Adagrad state 已 strict migrate/load，分支起点
的 8 个权重全零，训练后 `theta L1=0.292117`，证明新增结构真实参与学习。

### 30.3 NUAA 训练集内部探索信号（非正式结果）

协议：`seed=20260713`、epoch79 同一 parent、只从
`img_idx/train_NUAA-SIRST.txt` 确定性得到 `170 train + 43 val`、固定 epoch119 读取一次。

| 方法 | IoU | PD | FA |
|---|---:|---:|---:|
| canonical baseline | 0.6281 | 0.9259 | 98.6498 |
| CCFD | **0.6505** | **0.9259** | **74.5197** |
| Δ | **+0.0224** | **0.0000** | **-24.1301** |

该结果说明 CCFD 值得进入完整训练，但在最终协议修订后**不得称为正式 PASS**：它使用了
训练 manifest 内部的 `170/43` 开发划分，而用户要求与 DNANet 等方法保持完全一致，只用
官方 train/test。CCFD 结构暂时保持不变；真正的 final-fusion gate 改为完整 213 张 NUAA
训练 400 epochs，并按公开代码惯例每个 epoch 在 214 张 test 上评测、保存最大 IoU
checkpoint。只有 CCFD 的 best-IoU checkpoint 超过相同逐 epoch test-selection 协议的
canonical baseline，才冻结为成功结构。

---

## 31. 最终数据协议锁定：仅 train/test，无 validation

为与 DNANet、UIU-Net、MSHNet 等公开实现进行可比的性能比较，后续所有可进入论文表格的
运行统一遵守：

| 数据集 | 训练 manifest | 训练数 | 测试 manifest | 测试数 |
|---|---|---:|---|---:|
| NUAA-SIRST | `img_idx/train_NUAA-SIRST.txt` | 213 | `img_idx/test_NUAA-SIRST.txt` | 214 |
| NUDT-SIRST | `img_idx/train_NUDT-SIRST.txt` | 663 | `img_idx/test_NUDT-SIRST.txt` | 664 |
| IRSTD-1K | `img_idx/train_IRSTD-1K.txt` | 800 | `img_idx/test_IRSTD-1K.txt` | 201 |

约束如下：

1. `evaluation_protocol=official_train_test`；不创建、不读取 `split_val.txt`；
2. train manifest 全量参与优化；每 10 个 epoch 在 test manifest 评测（评测点为
   `9,19,...,399`），以这些固定评测点中的最大 test IoU 保存
   `checkpoint_best_iou.pkl`，最终报告该 checkpoint 的 IoU/PD/FA；
3. NUDT 目录中的 `hcval_NUDT-SIRST.txt` 永不进入训练、选择或测试；
4. 先前 internal-holdout 实验全部标记为 exploratory，不进入最终横向表；
5. 当前正式顺序为 NUAA full-train/best-checkpoint gate → NUDT full-train replication →
   IRSTD-1K full-train replication；所有方法使用相同 test-selection 规则，不混用 last epoch
   与 best epoch。

已停止的 `gate_ccfd_nudt_seed20260713_e120` 使用了 `530/133` 内部分割，因此按协议作废，
即使其后存在 checkpoint 或指标也不得引用。

### 31.1 绝对路径与 baseline 实际输入审计

2026-07-12 重新以 `realpath`、run config、split snapshot 和 manifest 内容四层核验：

- 唯一有效根目录是 `/home/md0/ly/DEA/datasets`；`/home/ly/DEA/datasets` 不存在；
- `formal_official_baseline_nuaa_seed20260713_e400` 的 `dataset_dir` 解析为
  `/home/md0/ly/DEA/datasets/NUAA-SIRST`；
- baseline 保存的 `split_train.txt`/`split_test.txt` 与对应 `img_idx` 清单逐行顺序和 ID
  集合完全相同；字节哈希差异仅由 CRLF→LF 换行规范化造成；
- 旧 baseline endpoint `0.7074 / 0.9506 / 28.7350` 使用的确为 213/214 划分，但它只评测
  epoch399，按最终 best-checkpoint 协议降级为参考值，不能作为最终 baseline；
- 三数据集 train/test 均无重复、交集为 0、所有 image-mask pair 存在，完整审计 PASS。

原始 manifest SHA-256：

| 数据集 | train SHA-256 | test SHA-256 |
|---|---|---|
| NUAA-SIRST | `324e5dadcb6cc9fc2a99a5f5dedd06ad4de77b2ed826e4ceffda8b6a784da0b4` | `e49023203a323c247306b314f23c8b3b917093a26984067792355adff7a8386e` |
| NUDT-SIRST | `e0a79f7c3d42548ba7d7dad9d2d336012b63a6bc5081e89e286f0f45036f8ec3` | `a463c52ee64b1c803c4a322fe090aaf6bc360844898e3943bb7c64a8e551b86e` |
| IRSTD-1K | `689a5f30a394ad47315ebe0f6df2d7f12429aa314ffb2cdf86f7fbd7be4ee744` | `8c71e474358acb84f2cbebfd1282ffea236f9cb852b7f7c04feb2fd99804c579` |

### 31.2 当前严格配对 run

NUAA 已从 epoch0 重新启动两条独立完整运行，而不是从只评测终点的旧 parent 接续：

- baseline：`repro_runs/official_best_baseline_nuaa_seed20260713_e400`；
- CCFD：`repro_runs/official_best_ccfd_nuaa_seed20260713_e400`。

二者均为 `seed=20260713`、213 train、214 test、`evaluation_interval=1`、400 epochs、
max-test-IoU checkpoint selection。epoch0 的两条指标逐项相同
`0.0002 / 0.0190 / 506862.1056`，验证 warm-up 冷路径和初始随机状态完全配对；CCFD 只在
四尺度 warm path 启用后通过新增 8 参数产生结构差异。

工程加速审计补充：同一 checkpoint 使用 test batch=1 与 batch=8 时，PD 相同，但 IoU
相差约 `3.8e-9`，FA 相差 `7.56/Mpixel`；这是批量卷积数值舍入在 0.5 二值阈值附近造成的
离散差异。故拒绝 batched-test 加速，正式 run 固定公开实现的 test batch=1。前述两条为此
审计被提前停止，正式可报告目录改为：

- `repro_runs/official_best_b1_baseline_nuaa_seed20260713_e400`；
- `repro_runs/official_best_b1_ccfd_nuaa_seed20260713_e400`。

最终评测频率随后固定为每 10 epochs 一次，上述 interval=1 run 在早期即停止且不引用。
唯一可报告目录更新为：

- `repro_runs/official_e10_baseline_nuaa_seed20260713_e400`；
- `repro_runs/official_e10_ccfd_nuaa_seed20260713_e400`。

---

## 32. 冻结规则纠正与 baseline 分段诊断

### 32.1 纠正：baseline fidelity 不等于新设计成功

前文把 stem/encoder/bottleneck/decoder 的 state/forward identity 写成 `PASS/冻结`，该表述
不正确。identity 只证明实验没有污染 canonical baseline，不能证明这些阶段已经完成结构
创新或改善低组件虚警预算。从本节起状态统一改为：

- canonical 各段：**已审计参考，不冻结**；
- 新设计段：只有同时通过主指标、组件 FROC、跨阈值 gate 才标记 **DESIGN PASS**；
- 某段 DESIGN PASS 后不再修改，继续下一段。

CCFD 因而只是一条 final-fusion comparator，不是完整模型，也不能替代由前往后的设计。

### 32.2 MSHNet 为什么强

| 结构/目标 | 强项 | 直接证据 |
|---|---|---|
| full-resolution skip + decoder0 | 保留极小目标的定位细节，避免所有证据都经历多次下采样 | `x_e0` 直接进入 `decoder_0`，最终最细 side head 在原分辨率预测 |
| residual CA/SA blocks | 同时保留 identity path、通道显著性和空间显著性，参数量仅约 4.07M | 每个 block 为 residual + channel/spatial reweighting；后期 baseline 可达 0.71+ IoU |
| four native side heads | 细尺度给位置，粗尺度给上下文，降低单一路径尺度偏置 | 四个 decoder stage 各有 `1x1` head 并对齐到 full resolution |
| `3x3 Conv(4→1)` fusion | 比逐像素加权多一圈局部上下文，能修正 side logits 的局部错位 | final kernel 同时跨尺度、跨 3×3 邻域融合 |
| SLSIoU + location term | 对小目标面积失配与质心偏移敏感，比普通 BCE 更符合 IRSTD | loss 显式包含 area-sensitive `alpha` 与 `LLoss` 质心项 |

旧的强 baseline operating curve 在阈值 0.1–0.8 上 PD 保持 `0.963`、IoU 约
`0.718–0.725`，说明其后期优势不是单一阈值偶然命中，而是多尺度表征和 SLS 目标共同形成
了较宽决策间隔。

### 32.3 已证实的拖后腿位置

1. **warm-path/fusion 优化断层**。当前 full-train paired run 中，canonical baseline 在
   epoch9–109 长时间处于几乎全背景/全误警的坏平台，epoch119 才跃迁到 IoU `0.6078`；
   CCFD 在 epoch19 已出现有效实例检出。这证明从单 `output_0` 到四 side heads + final fusion
   的切换存在严重条件数/归责问题。它拖慢收敛，但 baseline 后期能够恢复，因此不是唯一
   的最终上限瓶颈。
2. **final scale-deletion 只解释少数伪组件**。已有 component linkage 审计中，58 个预测
   组件含 6 个 false components；精确 deletion responsibility 只覆盖 1 个（`16.7%`），
   且全部责任事件来自 scale0。说明 fine-scale decisive clutter 确实存在，但另外 5/6
   伪组件是多尺度共同支持或在更早表征阶段生成。只改 final fusion 理论上无法覆盖多数
   低预算错误，这与 CCFD 目前降低 FA、但 best-IoU 仍可能落后 baseline 的曲线一致。
3. **训练目标不区分“误警面积”和“误警组件数”**。SLSIoU、area alpha、质心 LLoss 都
   是像素/整体矩量目标；同样的误警面积拆成 1 个或 10 个孤立组件，loss 没有明确区别。
   因此它能得到高 pixel IoU，却没有机制保证低 FP-components/image 下的实例 PD。

### 32.4 尚待逐段实验证实的结构假设

以下是代码推导出的候选瓶颈，不能在 probe 前写成结论：

- `1×1 conv_init` 只做逐像素通道混合，对局部峰值与热杂波没有可辨识的空间定义；
- repeated max-pooling 可能让亚分辨率目标消失，同时把孤立热噪声 alias 到深层；
- CA/SA 中的 global/channel max 可能把强热杂波当成目标显著性；
- coarse heads 被要求直接预测完整 tiny-target mask，bilinear 上采样可能产生 halo 或相邻
  伪组件；
- decoder concat 可能无条件重新注入 fine clutter。

下一步必须从 stem 开始，为这些假设设计单段干预和同协议 probe；未通过组件 FROC 的段
不冻结，也不越级修改后段。

---

## 33. 第一段重启：从幅值池化到支持持续性传输（SPT0）

### 33.1 正式 baseline / CCFD 完整配对结论

两条 `seed=20260713`、213 train / 214 test、400 epochs、每 10 epochs test 的正式运行均已
结束，最优 checkpoint 都出现在 epoch379：

| 方法 | best epoch | IoU | PD | FA/Mpixel |
|---|---:|---:|---:|---:|
| MSHNet-Deterministic | 379 | **0.728026** | 0.946768 | 22.1039 |
| CCFD-MSHNet | 379 | 0.725052 | **0.950570** | **18.1822** |
| CCFD − baseline | 0 | -0.002974 | +0.003802 | -3.9217 |

CCFD 的 8 个 stencil 参数训练后 `L1=1.323637`，因此并非 identity collapse；它能降低
pixel false-alarm area 并多检出一个目标左右，但主 IoU 下降，故结论为
**有效但不完整的末端 comparator，DESIGN FAIL，不冻结**。

51 个均匀概率阈值的初步 component-FROC 中，baseline 与 CCFD 的低预算平均 PD 分别为
`0.633714` 和 `0.636248`。该增量过小，且 0.01/0.05 FPPI 预算受 sigmoid 饱和与粗阈值
采样影响均落在 0；因此只能作为方向信号，正式 FROC 需要使用高 logit 区域加密阈值后再
报告，不能把 `+0.002535` 写成已确认优势。

随后已修正评测：不再先计算 float32 sigmoid（大 logit 会精确饱和为 1、破坏排序），而是
在 `[-20,160]` 上以步长 1 直接比较 181 个 raw-logit thresholds。修正后 baseline / CCFD
的六预算平均 PD 为 **0.689480 / 0.681242**，CCFD 反而下降 `-0.008238`；0.01 FPPI 下
为 `0.091255 / 0.064639`，0.05 FPPI 下为 `0.193916 / 0.205323`。因此 CCFD 不仅主 IoU
FAIL，也不跨低预算稳定，最终拒绝结论进一步加强。旧 probability-grid JSON 仅保留为
数值错误审计，不再用于模型判断。

### 33.2 逐层组件轨迹：哪里产生区分、哪里把区分丢掉

在 baseline epoch379 checkpoint 上，以最终预测的 249 个 matched components、15 个 false
components、249 个 matched targets 和 14 个 missed targets 回投到各层，计算归一化 feature
energy。该审计是相关性定位，不替代因果消融：

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

这组数据把“深层目标消失”和“浅层杂波回注”从泛泛假设缩小为两个可检验断点：

1. encoder2 已学到较强的 target/clutter 区分，后续 pooling/middle 却使其显著退化；
2. decoder1 已恢复最佳区分，decoder0 拼接 full-resolution `e0` 后再次退化。

另外，16 个 native scale subset 的冻结审计显示，仅保留 s0+s1 即严格支配全四尺度：
`IoU 0.729992 vs 0.728026`、`PD` 相同、`FA 21.6760 vs 22.1039`。这不是删除 s2/s3 的最终
模型证据，而是说明“增加尺度”不是瓶颈，尺度证据如何存活和回注才是瓶颈。

### 33.3 第一个因果干预：删除最强位点后还剩什么

对第一个 2×2 max-pool 单元，记最大值为 `m1`，删除该最大位点后的反事实最大值为 `m2`。
冻结所有 baseline 权重，只把该边界改成

\[
P_\alpha=(1-\alpha)m_1+\alpha m_2.
\]

结果如下：

| α | IoU | PD | FA/Mpixel | false components |
|---:|---:|---:|---:|---:|
| 0 | 0.728026 | 0.946768 | 22.1039 | 15 |
| 0.02 | 0.728441 | 0.946768 | 21.7473 | 15 |
| 0.05 | **0.728509** | 0.946768 | 21.5334 | 15 |
| 0.10 | 0.727311 | 0.946768 | 20.6065 | 15 |
| 0.40 | 0.726167 | 0.946768 | 18.5387 | **12** |
| 1.00 | 0.709180 | 0.935361 | 16.1857 | 16 |

`alpha=0` 与原 forward 的最大绝对误差为 0。小幅删除在不改变 PD 的情况下同时提升 IoU、
降低 FA，证明 max-pool 无条件传播 strongest-site-exclusive evidence 是真实可利用的前段
因果，而非只由末端 fusion 造成。固定比例不是最终模型：`alpha` 增大时 pixel FA 与实例
组件数并不单调一致，且 `alpha=1` 丢失目标。

一个无参数的线性 channel-consensus gate 也已被否决：`0.710906 / 0.935361 / 16.8274`，
说明“有共识就保留、无共识就完全删除”的未经校准规则过强。

### 33.4 SPT0 单一结构方程

当前第一段候选不是 top-k pooling 的换名。对每个 channel 的 2×2 cell，先对四个位置做
cell 内标准化并得到软空间 ownership `q_{c,s}`，以跨通道平均 `\bar q_s` 定义支持一致性：

\[
\kappa_c=\operatorname{clip}_{[0,1]}
\left(\frac{\sum_s q_{c,s}\bar q_s-1/4}{3/4}\right),\qquad
g_c=\frac{1+\kappa_c}{2},
\]

\[
\operatorname{SPT}(X)_c=m_{2,c}+g_c(m_{1,c}-m_{2,c}).
\]

`m1-m2` 是最强单一位点独占的证据；`kappa` 回答该位点是否得到跨通道空间支持；`g`
决定该独占证据能否跨分辨率边界存活。`g` 等权平均“事实存活先验 1”和观测支持
`kappa`，因此最多删除一半独占证据，完美支持时严格等于 max-pool。全算子参数为 0，
不增加卷积分支、attention、frequency/edge head 或新 loss，也不存在把 gate 学回 1 的
identity escape。

最初允许 `tau` 无约束学习的实现已在 epoch9 被机制审计否决：`tau` 从 `-0.25` 逃逸到
`-0.874`，使 gate 接近 1、结构退回 max-pool；该 run 在 epoch19 test 中止并永久标记为
aborted。无参数版本在冻结 checkpoint 上得到 `IoU 0.724390 / PD 0.950570 / FA 16.9701`，
false components 从 15 降到 11，证明它直接命中低组件虚警目标，接下来需由从头训练恢复
并超过 IoU。

当前只在 boundary0 替换 native max-pool，物理 variant 为 `spt0`。测试包括确定性 CUDA
backward、真实 forward/backward、输出位于 `(m1+m2)/2` 与 `m1` 之间、参数量与 baseline
完全相同和 stage0 唯一改动。正式 run：

`repro_runs/official_e10_spt0_nuaa_seed20260713_e400`

与 baseline 完全相同的 seed、manifest、loss、optimizer、epoch 和每 10 epoch test-selection
协议运行中。

### 33.5 AAAI 新颖性闸门

严格 reviewer 当前给出 **conditional reject**，原因是 DPP（CVPR 2018）和 ordinal/rank
pooling 已覆盖“自适应 pooling”与“学习有序激活权重”。因此：

- 不得把 SPT0 单独写成完整 AAAI 方法；
- 只有 strongest-site deletion counterfactual、channel support persistence 和后续同一
  transport law 对 decoder0 回注的约束共同必要时，才形成不同于普通 rank pooling 的贡献；
- 若 SPT0 的正式训练不能同时改善 IoU 与低组件预算 FROC，立即删除该结构，不进入后段；
- 若 SPT0 通过，固定这条方程，不再改前段定义，再沿网络向后验证同一方程，而不是新增
  第二种模块。

### 33.6 最优 checkpoint 对最优 checkpoint，禁止同 epoch 定胜负

所有同 epoch 数字只用于诊断收敛速度和 warm-path 断层，不参与结构 PASS/FAIL。SPT0 的
唯一正式判断必须等待 400 epochs 完成后执行：

1. baseline 在其 40 个固定 test 点 `9,19,...,399` 中选择最大 IoU checkpoint；
2. SPT0 独立在自己的 40 个固定 test 点中选择最大 IoU checkpoint；
3. 只比较两者各自 `checkpoint_best_iou.pkl` 的 IoU/PD/FA、raw-logit component-FROC 和
   跨阈值曲线；
4. epoch109 等中期 checkpoint 的 FROC 统一标记为 `interim`，不得用于最终结论；
5. 即使 SPT0 在较早 epoch 超过 baseline 的同 epoch 结果，也不能据此冻结；反之亦不能
   因中期落后成熟 baseline 而提前否决。

该规则已固化到 `tools/compare_independent_best.py`：工具先核验两条 run 各自完整的固定
测试日程，再读取各自的 `checkpoint_best_iou.pkl`；两个最优点允许来自不同 epoch，输出
明确记录 `selection_rule=independent_per_run_best_iou_checkpoint` 和
`same_epoch_required=false`。若 checkpoint 不是该 run 自己的最大 IoU 测试点，工具直接
失败，不生成候选优于 baseline 的结论。
