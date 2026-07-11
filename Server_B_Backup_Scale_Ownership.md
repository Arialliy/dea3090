# Server B 备线：MSHNet 多尺度深监督冲突与 Scale Ownership

> **研究方向**：Resolution-Aware Scale Ownership for Deep Supervision  
> **执行服务器**：Server B  
> **建议分支**：`research/scale-ownership`  
> **代码基线**：`Arialliy/DEA` 的 `main`，文档分析时最新提交为 `9a363799baf904127a07cd05a24b8e526d4511c4`（短 SHA `9a36379`，2026-07-11）。实验开始前必须记录本机完整 SHA。  
> **核心约束**：本线第一阶段只做梯度与标签可表达性审计；没有跨数据集证据前，不实现 learned gate、attention 或新 decoder。

---

## 0. 一页执行摘要

### 0.1 本线要回答的唯一问题

当前 canonical MSHNet 在 warm 结束后，对 final 输出和四个 side heads 分别计算 SLS loss，再等权平均：

```python
loss = self.loss_fun(pred, labels, ...)
labels_for_scale = labels
for j in range(len(masks)):
    if j > 0:
        labels_for_scale = self.down(labels_for_scale)
    loss = loss + self.loss_fun(masks[j], labels_for_scale, ...)
loss = loss / (len(masks) + 1)
```

这隐含假设：

> 每一个分辨率的 side head 都应独立检测并定位所有目标。

但极小目标在 `H/4`、`H/8` 标签上可能只剩一个 coarse cell，多个邻近目标还可能合并。备线要验证：

> **低分辨率辅助监督是否对共享参数产生与 final objective 相反的梯度，并且冲突是否由目标在该尺度上的不可表达性系统性预测。**

### 0.2 严格执行顺序

1. 不训练新模型，先使用 clean baseline checkpoints 做离线梯度审计；
2. 建立目标在各尺度上的可表达性指标；
3. 证明“不可表达 → 负梯度冲突 → FP/FN”链条跨数据集成立；
4. 只有诊断 GO，才实现 deterministic Scale Ownership；
5. 首个方法只改变辅助监督拓扑，final inference graph 完全不变；
6. learned ownership、PCGrad、动态 attention 只能作为后续 control，不作为第一版主方法。

### 0.3 第一批任务

| ID | 任务 | 是否训练 | 输出 |
|---|---|---:|---|
| B-A0 | baseline loss/gradient exactness test | 否 | identity report |
| B-A1 | frozen checkpoint head-gradient audit | 否 | per-image/head JSONL |
| B-A2 | target representability audit | 否 | component-scale ledger |
| B-A3 | conflict/error association analysis | 否 | OOF statistics + GO/NO-GO |
| B-M0 | legacy deep supervision | 是 | baseline control |
| B-M1 | final-only | 是 | 判断 deep supervision 净作用 |
| B-M2 | side no-location control | 是 | 隔离 side LLoss |
| B-M3 | hard Scale Ownership | 是 | 第一候选 |
| B-M4 | random ownership matched control | 是 | 排除稀疏监督本身 |

### 0.4 备线停止条件

若负梯度余弦与尺度可表达性没有稳定关系，或它不能在目标面积、输出置信度、SCR/局部对比度之外解释 FP/FN，则立即 NO-GO，不再设计 ownership 模块。

---

## 1. 当前深监督机制与潜在缺陷

### 1.1 MSHNet side heads

完整多尺度阶段返回：

```text
side0: H×W，stride 1
side1: H/2×W/2，stride 2
side2: H/4×W/4，stride 4
side3: H/8×W/8，stride 8
final: side logits 上采样后融合
```

训练 target 通过连续 `MaxPool2d(2,2)` 下采样。Max pooling 保证小目标“存活”，但不保证其几何可表达性：

- 1–2 pixel 目标可能成为单个 coarse positive cell；
- 两个目标可能占据同一 cell 或在二值图上合并；
- centroid 的量化下界可能已经超过 PD 的匹配半径；
- coarse head 仍被要求承担位置和 scale-sensitive loss。

### 1.2 为什么不能直接做动态尺度权重

已有大量工作已按目标尺寸、对比度、SCR 或 feature attention 动态加权。若直接学习四个权重：

\[
L= L_f + \sum_s \alpha_s L_s,
\]

审稿人会认为这是常规 dynamic weighting，而且模型可能简单把困难 head 权重压到零。

本线必须先证明一个更具体的现象：

\[
\text{representation failure at scale }s
\Rightarrow
\nabla L_s^\top \nabla L_f < 0.
\]

方法改变的应是“哪个实例监督哪个 head”，即监督拓扑，而不只是四个全局标量。

### 1.3 本线不预设的结论

- 辅助监督不一定有害；
- 负余弦并不自动代表训练失败；
- coarse target 单像素化不一定必然产生冲突；
- final-only 不一定更好；
- 目标面积本身可能已解释全部现象。

因此先审计、再建模。

---

## 2. 研究假设与可证伪预测

### H1：尺度不可表达性提高梯度冲突概率

对 final loss 梯度 `g_f` 与第 `s` 个 side loss 梯度 `g_s`：

\[
C_{s,f}=\frac{g_s^\top g_f}{\|g_s\|\|g_f\|}.
\]

**预测**：component 在该尺度的有效面积越小、centroid 量化误差越大、组件合并越严重，`C_{s,f}<0` 的概率越高。

### H2：冲突具有层级结构

**预测**：冲突不仅出现在 final fusion 参数，还会在共享 encoder 或早期 decoder 参数上出现；若只在 side head 自身参数上出现，没有方法意义。

### H3：冲突与最终错误相关

**预测**：控制目标面积、数量、局部对比度和输出置信度后，head conflict 仍能预测：

- FN component；
- FP component；
- mixed FP/FN；
- centroid error。

### H4：Scale Ownership 优于同稀疏度随机监督

**预测**：几何确定的 ownership 优于随机 assignment、全局 loss reweight 和 final-only；否则收益不能归因于 resolution-aware supervision。

---

## 3. 分支、目录与两阶段纪律

### 3.1 初始化

```bash
git checkout main
git pull --ff-only
git rev-parse HEAD
git checkout -b research/scale-ownership
```

### 3.2 新增文件

```text
utils/instance_targets.py
tools/audit_multiscale_gradients.py
tools/analyze_multiscale_gradients.py
model/scale_ownership.py
model/masked_aux_loss.py
tools/run_scale_ownership.py
tools/summarize_scale_ownership.py
tests/test_gradient_audit.py
tests/test_scale_ownership.py
tests/test_masked_aux_loss.py
```

### 3.3 修改文件

```text
model/loss.py
utils/data.py
main.py
```

### 3.4 Server B 第一阶段禁止修改

在诊断 GO 前，不得修改：

```text
model/MSHNet.py 的 encoder/decoder/fusion
side head 数量
final convolution
推理 forward
```

也不得增加：

- learned router；
- channel/spatial attention；
- frequency branch；
- hard-negative loss；
- 新 decoder interaction；
- PCGrad optimizer wrapper。

---

## 4. Phase A：离线梯度冲突审计

### 4.1 使用哪些 checkpoint

优先使用 clean pipeline 的 canonical MSHNet：

```text
3 datasets × 3 seeds
seeds: 20260711, 20260712, 20260713
checkpoint: internal validation 选出的 canonical checkpoint
```

若 3×3 baseline 尚未全部完成：

1. 先对已完成 checkpoint 做脚本验证；
2. 不用历史 test-selected 或不同初始化的 checkpoint 补齐；
3. 最终 GO/NO-GO 必须等待规范化 3×3 审计。

### 4.2 审计模式

- `model.eval()`：冻结 BN running statistics 和 dropout 行为；
- 不使用 `torch.no_grad()`，因为需要参数梯度；
- batch size = 1，保证 image-level 归因；
- 不调用 optimizer，不写 `.grad`；
- 使用 `torch.autograd.grad`；
- 固定完整多尺度 forward；
- validation/internal holdout 用于诊断，official test sealed。

### 4.3 梯度参数组

首先定义稳定的 encoder sentinel 组：

```python
ENCODER_SENTINELS = {
    "shallow": (
        "conv_init.weight",
        "encoder_0.0.conv1.weight",
        "encoder_0.0.conv2.weight",
    ),
    "mid": (
        "encoder_1.0.conv1.weight",
        "encoder_2.0.conv1.weight",
    ),
    "deep": (
        "encoder_3.0.conv1.weight",
        "middle_layer.0.conv1.weight",
    ),
}
```

Decoder 参数不能对四个 side heads 使用同一个固定集合。比较 `side_s` 与 final 时，只能在**两者共同依赖的计算路径参数交集**上计算 cosine：

```python
COMMON_DECODER_BY_SIDE = {
    # side0 依赖 decoder3→2→1→0
    0: (
        "decoder_3.0.conv1.weight",
        "decoder_2.0.conv1.weight",
        "decoder_1.0.conv1.weight",
        "decoder_0.0.conv1.weight",
    ),
    # side1 依赖 decoder3→2→1
    1: (
        "decoder_3.0.conv1.weight",
        "decoder_2.0.conv1.weight",
        "decoder_1.0.conv1.weight",
    ),
    # side2 依赖 decoder3→2
    2: (
        "decoder_3.0.conv1.weight",
        "decoder_2.0.conv1.weight",
    ),
    # side3 只依赖 decoder3
    3: ("decoder_3.0.conv1.weight",),
}
```

若把 side3 不依赖的 decoder0/1/2 参数也拼入向量，side gradient 会被补零而 final gradient 非零，cosine 会被人为缩小。因此审计脚本必须 fail-closed 地验证每个参数对两项 loss 均有非空梯度；非共同参数不得用于主要 cosine。

需要根据实际 `state_dict` 名称做一次打印校验。报告中同时给出：

1. sentinel parameter 结果：低成本、全图像；
2. full common-path 结果：高成本、预注册子集。

不要只看 `final.weight` 或各 side output head，因为这些参数并非同一共享优化问题。

---

## 5. 新增 `tools/audit_multiscale_gradients.py`

### 5.1 核心接口

```python
@dataclass
class GradientStats:
    dot: float
    cosine: float
    norm_a: float
    norm_b: float
    conflict: bool
    norm_ratio: float


def select_named_parameters(model, names):
    named = dict(model.named_parameters())
    missing = [name for name in names if name not in named]
    if missing:
        raise KeyError(f"missing parameters: {missing}")
    return [named[name] for name in names]


def autograd_vector(loss, params, retain_graph):
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    flat = []
    for param, grad in zip(params, grads):
        if grad is None:
            raise RuntimeError(
                "selected parameter is not shared by both objectives; "
                "fix the side-specific common-path parameter set"
            )
        flat.append(grad.detach().reshape(-1))
    return torch.cat(flat)


def compare_gradients(g_a, g_b, eps=1e-12):
    dot = torch.dot(g_a, g_b)
    norm_a = g_a.norm()
    norm_b = g_b.norm()
    cosine = dot / (norm_a * norm_b).clamp_min(eps)
    return GradientStats(
        dot=float(dot.cpu()),
        cosine=float(cosine.cpu()),
        norm_a=float(norm_a.cpu()),
        norm_b=float(norm_b.cpu()),
        conflict=bool(dot < 0),
        norm_ratio=float((norm_a / norm_b.clamp_min(eps)).cpu()),
    )
```

### 5.2 避免显存爆炸的两级实现

#### Level 1：sentinel audit

- 每个 parameter group 只选择少数固定卷积权重；
- 对 validation 所有图像运行；
- 每图计算 final vs 4 side heads。

#### Level 2：full-group audit

- 对每个数据集/seed 使用预注册的图像子集；
- 子集按 target count、target size、failure taxonomy 分层抽样；
- 对 encoder/common decoder 组的所有参数计算精确 cosine；
- 抽样规则在看结果前写入 manifest。

### 5.3 审计主循环骨架

```python
model.eval()
criterion = SLSIoULoss()

for sample_index, (image, target) in enumerate(loader):
    image = image.to(device)
    target = target.to(device)

    masks, final_logit = model(image, True)
    if len(masks) != 4:
        raise RuntimeError("full multiscale graph required")

    final_loss = criterion(
        final_logit, target, warm_epoch=-1, epoch=0
    )

    targets = [target]
    for _ in range(3):
        targets.append(F.max_pool2d(targets[-1], 2, 2))

    side_losses = [
        criterion(mask, targets[s], warm_epoch=-1, epoch=0)
        for s, mask in enumerate(masks)
    ]

    for group_name, parameter_names in PARAM_GROUPS.items():
        # encoder sentinel 组可对所有 side 共用；decoder 组必须按 side 取共同路径。
        encoder_params = select_named_parameters(model, parameter_names)
        g_final_encoder = autograd_vector(
            final_loss, encoder_params, retain_graph=True
        )

        for side_index, side_loss in enumerate(side_losses):
            g_side_encoder = autograd_vector(
                side_loss, encoder_params, retain_graph=True
            )
            stats = compare_gradients(g_side_encoder, g_final_encoder)
            write_jsonl(...)

            decoder_names = COMMON_DECODER_BY_SIDE[side_index]
            decoder_params = select_named_parameters(model, decoder_names)
            g_final_decoder = autograd_vector(
                final_loss, decoder_params, retain_graph=True
            )
            g_side_decoder = autograd_vector(
                side_loss, decoder_params, retain_graph=True
            )
            decoder_stats = compare_gradients(
                g_side_decoder, g_final_decoder
            )
            write_jsonl(...)
```

### 5.4 优化计算

上面会为每个 group 重复计算 final gradient。实际实现应：

1. 对每个 group 计算一次 final gradient；
2. 对每个 side loss 计算一次；
3. 立即转 CPU 或只累计 dot/norm；
4. 每张图结束后释放 graph：

```python
del masks, final_logit, final_loss, side_losses
```

不要调用 `loss.backward()`，避免污染参数 `.grad`。

### 5.5 审计前后完整性检查

```python
before = {n: p.detach().clone() for n, p in model.named_parameters()}
# run audit
for name, param in model.named_parameters():
    assert torch.equal(before[name], param.detach())
    assert param.grad is None
```

同时检查 BN buffers 未变化。

---

## 6. 目标尺度可表达性审计

梯度冲突只有与一个可解释的标签几何变量稳定关联，才有论文价值。

### 6.1 每个 component、每个 head 记录的字段

设 head stride `d_s ∈ {1,2,4,8}`。

```text
component_id
native_area
native_equivalent_diameter
native_centroid_x/y
scale_index
stride
coarse_positive_cell_count
coarse_bbox_width/height
survival
centroid_quantization_lower_bound
component_collision_count
merged_with_other_component
coarse_component_count_ratio
representability_score
```

### 6.2 不能只看 MaxPool 后是否存活

`survival=1` 太弱。一个 component 即使保留一个 positive cell，也可能无法表达精确位置。

建议定义四个量。

#### A. 有效 cell 数

\[
a_{k,s}=|\operatorname{Pool}_s(G_k)|.
\]

#### B. 尺度内等效直径

\[
d_{k,s}=2\sqrt{a_{k,s}/\pi}.
\]

#### C. centroid 量化下界

把 native centroid 映射到 coarse grid 后，计算其到最近 coarse cell center 的距离，再映射回 native pixel。

#### D. collision/merge

对每个 component 独立 max-pool；若两个 component 在同一 coarse cell 为正，则标记 collision。再对 pooled union 做 connected components，检查组件数量是否减少。

### 6.3 可表达性 score

第一版采用确定性、不可学习的 score：

\[
q_{k,s}
=
q^{area}_{k,s}
q^{quant}_{k,s}
q^{merge}_{k,s},
\]

其中：

\[
q^{area}_{k,s}=\min\left(1,\frac{a_{k,s}}{a_{min}}\right),
\]

\[
q^{quant}_{k,s}=\exp(-e_{k,s}/\tau_q),
\]

\[
q^{merge}_{k,s}=\begin{cases}
0,&\text{collision/merge}\cr
1,&\text{otherwise.}
\end{cases}
\]

诊断默认：

```text
a_min = 2 coarse cells
tau_q = 1 native pixel
```

这些阈值只能在审计开始前修改一次。

### 6.4 图像级汇总

对于每个 head：

```text
min_q
mean_q
fraction_q_below_0.5
number_unrepresentable_components
number_collisions
max_quantization_error
```

再与该图像的 `cosine(side_s, final)` 对齐。

---

## 7. `utils/data.py`：为审计和 ownership 返回 instance map

与主线分支独立实现，不依赖 Server A merge。

```python
# imports
import numpy as np
from skimage import measure

# __init__
self.return_instance_map = (
    bool(getattr(args, "return_instance_map", False))
    and mode in {"train", "val"}
)

# __getitem__ after augmentation/resize
img_tensor = self.transform(img)
mask_tensor = (transforms.ToTensor()(mask) > 0.5).float()

if self.return_instance_map:
    labels = measure.label(
        mask_tensor[0].numpy().astype(np.uint8),
        connectivity=2,
        background=0,
    ).astype(np.int32)
    return img_tensor, mask_tensor, torch.from_numpy(labels)

return img_tensor, mask_tensor
```

这里审计需要 validation instance map，所以与主线“仅 train 返回”不同。必须同步修改 audit loader；正常 `main.py::test` 仍使用两元组，因此 regular training 时默认 `return_instance_map=False`。

---

## 8. 审计输出格式

### 8.1 `gradient_records.jsonl`

每行一个 image × side head × parameter group：

```json
{
  "dataset": "NUAA-SIRST",
  "seed": 20260711,
  "checkpoint": "...",
  "image_id": "...",
  "side_index": 3,
  "stride": 8,
  "parameter_group": "shallow",
  "gradient_dot": -0.0123,
  "gradient_cosine": -0.18,
  "side_grad_norm": 0.42,
  "final_grad_norm": 0.31,
  "conflict": true,
  "target_component_count": 2,
  "min_representability": 0.12,
  "mean_representability": 0.35,
  "collision_count": 1,
  "failure_taxonomy": "mixed_fp_fn"
}
```

### 8.2 `component_scale_records.jsonl`

每行一个 image × component × scale。

### 8.3 manifest

必须写入：

```text
source commit
checkpoint SHA256
split SHA256
parameter group names
loss implementation
warm/full graph setting
a_min / tau_q
component connectivity
sample selection policy
PyTorch/CUDA/cuDNN versions
```

---

## 9. `tools/analyze_multiscale_gradients.py`

### 9.1 基础统计

按 dataset、seed、head、parameter group 报告：

```text
mean/median cosine
negative cosine rate
negative dot-product rate
norm ratio
cosine quantiles
```

负 cosine 与负 dot 都要报告。cosine 只看方向，dot 同时反映实际更新冲突。

### 9.2 分层分析

至少按以下分层：

- target area quartile；
- target count：1 / >1；
- representability score；
- collision yes/no；
- failure taxonomy：perfect/localization/fp/fn/mixed；
- local contrast/SCR（若 clean audit 已有）；
- final confidence/margin；
- checkpoint seed。

### 9.3 增量预测能力

不能只给相关系数。建议 image-level OOF logistic/ordinal analysis：

#### Base features

```text
target area
component count
local contrast/SCR
final max/mean probability
predicted foreground mass
```

#### Added features

```text
per-head gradient cosine
negative conflict count
min/mean representability
collision count
```

比较：

- FN component 是否发生；
- FP component 是否发生；
- mixed FP/FN 是否发生。

使用 image-level folds，按 dataset/seed 分组，禁止在同一图像的重复记录上拆分训练和验证。

### 9.4 统计检验

- cluster bootstrap unit：`(dataset, image_id)`；
- seed 作为复现实验而不是独立图像样本；
- 报告 effect size 和 CI，不只报 p-value；
- 预注册主要比较：`side3 vs final` 在 `q<0.5` 与 `q>=0.5` 的 conflict-rate difference；
- secondary：side2、side1、不同 parameter groups。

---

## 10. Phase A 的 GO / NO-GO

### 10.1 诊断 GO

建议满足全部条件：

1. 至少两个数据集上，不可表达组的 negative-dot rate 比可表达组高 ≥ 15 个百分点；
2. 方向在至少两个 seed 上复现；
3. 关系出现在 shallow/mid/shared trunk，而不只是 head-specific 参数；
4. 加入 conflict/representability 后，对 FN 或 FP 的 OOF 预测指标有稳定增量；
5. 结果不能完全由 native target area 单变量解释；
6. full-group audit 与 sentinel audit 方向一致。

### 10.2 诊断 NO-GO

任一情形触发：

- conflict rate 很低或无 head/scale 规律；
- negative cosine 主要来自极小梯度数值噪声；
- 只在一个 seed 或一个 dataset 出现；
- area/confidence 已解释全部关联；
- conflict 与最终 FP/FN 无关；
- final-only 已明显更差且 ownership 没有可识别改进空间。

### 10.3 边界结果

若只发现 side LLoss，而不是分辨率，导致冲突：

- 不进入 Scale Ownership；
- 转为“side-head location supervision”小型 control；
- 与 Server A 结果合并解释；
- 备线不单独发展成论文。

---

## 11. Phase B：Deterministic Scale Ownership

只有 Phase A GO 后实施。

### 11.1 方法原则

- final head 始终监督所有目标；
- side head 只监督其可可靠表示的目标；
- 未分配给某 side head 的真实目标必须被 **ignore**，不能当背景；
- ownership 由 GT 几何确定，第一版不可学习；
- 推理图完全不变。

### 11.2 责任分数

对 component `k` 和 side head `s`，定义 native 等效直径：

\[
d_k=2\sqrt{A_k/\pi}.
\]

尺度内直径：

\[
\rho_{k,s}=d_k/\text{stride}_s.
\]

偏好目标在该 head 上覆盖约 `rho0=3` 个 cell：

\[
u_{k,s}
=q_{k,s}\exp\left[-\frac{(\log_2(\rho_{k,s}+\epsilon)-\log_2\rho_0)^2}{2\sigma^2}\right].
\]

推荐初始值：

```text
rho0 = 3.0
sigma = 0.75
q_min = 0.25
strides = [1, 2, 4, 8]
```

### 11.3 Hard ownership

```python
valid_scores = where(q >= q_min, u, -inf)
owner_scale = argmax(valid_scores)
```

若所有尺度都 invalid：

- fallback 到 side0；或
- 不给任何 side head，只有 final 监督。

两种策略都要做 control；第一版建议 fallback side0，以免完全失去辅助正样本。

### 11.4 Soft ownership（第二候选）

\[
r_{k,s}=\operatorname{softmax}(\log(u_{k,s}+\epsilon)/T),
\]

只在 hard ownership 明确有效后实现。否则 soft weighting 会退化为普通动态 loss weight。

---

## 12. 新增 `model/scale_ownership.py`

### 12.1 数据结构

```python
from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ScaleAssignment:
    # 每幅图一个 K×S 矩阵；K 可变，因此使用 list。
    responsibilities: List[torch.Tensor]
    hard_owner: List[torch.Tensor]
    representability: List[torch.Tensor]
    component_ids: List[torch.Tensor]
    strides: tuple[int, ...]
```

### 12.2 Builder 接口

```python
class ResolutionAwareScaleOwnership:
    def __init__(
        self,
        strides=(1, 2, 4, 8),
        preferred_diameter_cells=3.0,
        sigma=0.75,
        min_representability=0.25,
        mode="hard",
        temperature=0.5,
        fallback="side0",
    ):
        ...

    @torch.no_grad()
    def __call__(self, instance_map: torch.Tensor) -> ScaleAssignment:
        ...
```

`@torch.no_grad()` 是刻意的：assignment 来自 GT，不参与学习，避免把它误写成 learned routing。

### 12.3 必须保证的性质

- component id permutation invariant；
- hard mode 中每个 retained component 只有一个 owner；
- final 不在 assignment 中，因为 final 永远监督所有目标；
- collision 后的 coarse scale 可被 gate 掉；
- empty image 返回形状正确的空责任矩阵；
- 不依赖模型输出或 validation 指标。

---

## 13. 关键问题：未分配目标不能当作 side-head 背景

假设目标 `G_k` 分配给 side0，而 side3 的 target 直接删除该目标。如果 side3 的 valid mask 仍为 1，side3 会被要求把该目标位置预测成背景，这会制造新的冲突。

因此每个 side head 必须构造：

```text
positive target map
valid supervision map
```

- assigned component：positive，valid=1；
- 普通背景：negative，valid=1；
- unassigned component 及其小 dilation ring：ignore，valid=0。

这是 Scale Ownership 实现是否正确的核心。

---

## 14. 新增 `model/masked_aux_loss.py`

### 14.1 Masked scale-sensitive IoU

```python
class MaskedScaleIoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target, valid):
        if logits.shape != target.shape or target.shape != valid.shape:
            raise ValueError("logits/target/valid shapes must match")

        prob = torch.sigmoid(logits)
        valid = valid.to(prob.dtype)
        target = target.to(prob.dtype)

        p = prob * valid
        y = target * valid

        intersection = (p * y).sum(dim=(1, 2, 3))
        pred_sum = p.sum(dim=(1, 2, 3))
        target_sum = y.sum(dim=(1, 2, 3))
        union = pred_sum + target_sum - intersection

        dis = ((pred_sum - target_sum) / 2.0).square()
        alpha = (torch.minimum(pred_sum, target_sum) + dis) / (
            torch.maximum(pred_sum, target_sum) + dis + self.eps
        )
        iou = (intersection + self.eps) / (union + self.eps)

        # 无 assigned positive 的 head/image 仍可有背景约束，但不能让空 target
        # 的 IoU 数值主导。返回 per-sample，调用者按 active mask 汇总。
        loss_per_sample = 1.0 - alpha * iou
        return loss_per_sample
```

### 14.2 是否保留 side LLoss

第一版 ownership 的 side heads **不使用 global LLoss**。原因：

- component 被选择性分配后，全局 moment 的语义进一步复杂；
- coarse scale 的位置 loss 正是待验证冲突来源之一；
- final head 保留 canonical SLS，保证主任务仍有原始位置监督。

但必须有 control：

```text
B-M2 = 所有目标、所有 side heads，side 只用 scale-IoU，无 LLoss
```

这样 B-M3 相对 B-M2 的收益才能归因于 ownership，而不是简单去掉 side LLoss。

### 14.3 构造 side target/valid

```python
def build_side_supervision(instance_map, assignment, side_index, stride, ignore_dilation):
    assigned_native = torch.zeros_like(instance_map, dtype=torch.bool)
    unassigned_native = torch.zeros_like(instance_map, dtype=torch.bool)

    for b in range(instance_map.shape[0]):
        ids = assignment.component_ids[b]
        owners = assignment.hard_owner[b]
        for component_id, owner in zip(ids, owners):
            component = instance_map[b] == component_id
            if int(owner) == side_index:
                assigned_native[b] |= component
            else:
                unassigned_native[b] |= component

    target = assigned_native.float().unsqueeze(1)
    ignore = unassigned_native.float().unsqueeze(1)

    if stride > 1:
        target = F.max_pool2d(target, stride, stride)
        ignore = F.max_pool2d(ignore, stride, stride)

    if ignore_dilation > 1:
        pad = ignore_dilation // 2
        ignore = F.max_pool2d(
            ignore, ignore_dilation, 1, pad
        )

    target = (target > 0.5).float()
    valid = (ignore < 0.5).float()
    # assigned positive 必须强制 valid，即使 dilation 与其接触。
    valid = torch.maximum(valid, target)
    return target, valid
```

### 14.4 背景约束与空 assigned head

若某图在某 side head 没有 assigned component：

- 可以保留背景监督，但其权重应低；
- 或从该图的该 side loss 中跳过。

必须做 control：

```text
empty_side_policy = skip | background_only
```

推荐第一版 `skip`，避免大量空辅助监督把网络推向全背景。

---

## 15. `main.py` 修改

### 15.1 新 CLI

```python
parser.add_argument(
    "--deep-supervision",
    default="legacy",
    choices=[
        "legacy",
        "none",
        "side_no_location",
        "ownership_hard",
        "ownership_soft",
        "ownership_random_control",
    ],
)
parser.add_argument("--aux-loss-weight", type=float, default=0.8)
parser.add_argument("--ownership-preferred-cells", type=float, default=3.0)
parser.add_argument("--ownership-sigma", type=float, default=0.75)
parser.add_argument("--ownership-min-representability", type=float, default=0.25)
parser.add_argument("--ownership-temperature", type=float, default=0.5)
parser.add_argument(
    "--ownership-fallback",
    choices=["side0", "final_only"],
    default="side0",
)
parser.add_argument("--ownership-ignore-dilation", type=int, default=3)
parser.add_argument(
    "--empty-side-policy",
    choices=["skip", "background_only"],
    default="skip",
)
parser.add_argument("--gradient-audit", action="store_true")
parser.add_argument("--gradient-audit-interval", type=int, default=0)
```

### 15.2 数据集开关

```python
args.return_instance_map = args.deep_supervision in {
    "ownership_hard",
    "ownership_soft",
    "ownership_random_control",
}
```

### 15.3 保留 legacy 精确路径

绝不能把所有模式统一进一个“看起来更整洁”的新函数后破坏 baseline。推荐：

```python
if self.args.deep_supervision == "legacy":
    loss = self.loss_fun(pred, labels, self.warm_epoch, epoch)
    labels_for_scale = labels
    for j in range(len(masks)):
        if j > 0:
            labels_for_scale = self.down(labels_for_scale)
        loss = loss + self.loss_fun(
            masks[j], labels_for_scale, self.warm_epoch, epoch
        )
    loss = loss / (len(masks) + 1)
else:
    loss, ds_logs = self.compute_alternative_deep_supervision(...)
```

### 15.4 Alternative path

```python
def compute_alternative_deep_supervision(
    self,
    pred,
    masks,
    labels,
    instance_map,
    epoch,
):
    final_loss = self.loss_fun(
        pred, labels, self.warm_epoch, epoch
    )

    if not masks or self.args.deep_supervision == "none":
        return final_loss, {"final_loss": final_loss.detach()}

    # Control: all components at all heads, but side has no location.
    if self.args.deep_supervision == "side_no_location":
        aux_losses = []
        target_s = labels
        for s, side_logit in enumerate(masks):
            if s > 0:
                target_s = self.down(target_s)
            aux_losses.append(
                self.loss_fun(
                    side_logit,
                    target_s,
                    self.warm_epoch,
                    epoch,
                    with_shape=False,
                )
            )
        aux = torch.stack(aux_losses).mean()
        total = final_loss + self.args.aux_loss_weight * aux
        return total, {...}

    if instance_map is None:
        raise RuntimeError("ownership mode requires instance_map")

    assignment = self.scale_ownership(instance_map)
    aux_terms = []
    active_counts = []

    for s, (side_logit, stride) in enumerate(zip(masks, (1, 2, 4, 8))):
        target_s, valid_s, active_s = build_side_supervision(...)
        per_sample = self.masked_aux_loss(side_logit, target_s, valid_s)
        if self.args.empty_side_policy == "skip":
            active = active_s.to(per_sample.dtype)
            side_loss = (per_sample * active).sum() / active.sum().clamp_min(1.0)
        else:
            side_loss = per_sample.mean()
        aux_terms.append(side_loss)
        active_counts.append(active_s.float().mean().detach())

    aux = torch.stack(aux_terms).mean()
    total = final_loss + self.args.aux_loss_weight * aux
    return total, {...}
```

### 15.5 权重可比性

当前 legacy 是五个 loss 的平均。Alternative 推荐写为：

\[
L=L_f+\lambda_{aux}\frac{1}{S}\sum_sL_s.
\]

这与 legacy 的绝对尺度不同，因此必须加入两类 control：

1. `legacy_rescaled`：把 legacy 写成相同 `L_f + lambda_aux mean(L_s)`；
2. `side_no_location`：同权重但无 ownership。

否则优化步长变化会被误认为方法收益。

建议把 `legacy_rescaled` 加入 choices：

```text
legacy_exact
legacy_rescaled
```

论文主 baseline 仍使用 `legacy_exact`；机制消融用 `legacy_rescaled` 匹配总 loss 尺度。

### 15.6 Metadata 与 resume

加入 checkpoint semantic keys：

```text
deep_supervision
aux_loss_weight
ownership_preferred_cells
ownership_sigma
ownership_min_representability
ownership_temperature
ownership_fallback
ownership_ignore_dilation
empty_side_policy
```

不同 ownership 语义不得从同一个 optimizer checkpoint 续跑。

---

## 16. Phase B 实验矩阵

| ID | Final | Side supervision | Assignment | 作用 |
|---|---|---|---|---|
| M0 | legacy SLS | legacy SLS | all-to-all | canonical |
| M1 | legacy SLS | none | none | final-only |
| M2 | legacy SLS | scale-IoU | all-to-all | 去 side location control |
| M3 | legacy SLS | masked scale-IoU | hard geometry | 主候选 |
| M4 | legacy SLS | masked scale-IoU | random matched sparsity | 关键 control |
| M5 | legacy SLS | masked scale-IoU | area-only | 排除复杂 representability 无必要 |
| M6 | legacy SLS | masked scale-IoU | soft geometry | 仅 M3 GO 后 |
| M7 | legacy SLS | global scalar weight | no ownership | dynamic weighting control |

### 16.1 Random matched control

对每张图保持：

- 每个 head 获得的 component 数量；
- 每个 component 被分配的 head 数量；
- 总 auxiliary positive pixels 大致匹配；

但打乱 component–head 对应。若 M3 与 M4 相同，说明收益来自稀疏监督而非 resolution-aware ownership。

### 16.2 Area-only control

仅按 native area/diameter选择尺度，不使用 quantization 与 collision。若 M5 等同 M3，论文应简化方法，不能宣称复杂可表达性建模必要。

### 16.3 Global scalar control

根据每图 target size 给四个 side losses 标量权重，但仍让所有目标监督所有 head。它用于区分：

```text
改变 loss 权重
vs.
改变实例—head 监督拓扑
```

---

## 17. 训练调度

### Gate 0：代码 smoke

- NUAA-SIRST；
- seed `20260711`；
- 2 epoch；
- M0/M1/M2/M3/M4；
- 检查 finite、resume、metadata、instance-map collate。

### Gate 1：单 seed 跨数据集

```text
3 datasets × 1 seed × M0–M5
```

只用 internal validation。

### Gate 2：复现

只保留 M3 和最强 control：

```text
3 datasets × 3 seeds
```

### Gate 3：最终 control

若 M3 GO，再运行：

- M4 random matched；
- M5 area-only；
- M7 global scalar；
- 至少一个非 MSHNet backbone。

### 示例命令

```bash
python tools/run_scale_ownership.py \
  --batch-id scale_ownership_gate1 \
  --datasets NUAA-SIRST,NUDT-SIRST,IRSTD-1K \
  --seeds 20260711 \
  --methods legacy_exact,final_only,side_no_location,ownership_hard,ownership_random,ownership_area_only \
  --gpus 0,1 \
  --epochs 400 \
  --deterministic true
```

复现：

```bash
python tools/run_scale_ownership.py \
  --batch-id scale_ownership_repl_v1 \
  --datasets NUAA-SIRST,NUDT-SIRST,IRSTD-1K \
  --seeds 20260711,20260712,20260713 \
  --methods legacy_exact,ownership_hard,ownership_random \
  --gpus 0,1 \
  --epochs 400 \
  --deterministic true
```

---

## 18. 训练日志

每个 epoch/head 记录：

```text
final_loss
side0_loss ... side3_loss
weighted_aux_loss
aux_to_final_ratio
side_active_image_ratio
side_assigned_component_count
side_ignored_component_count
side_positive_pixel_count
side_valid_pixel_ratio
assignment_entropy（soft only）
assignment_histogram
fallback_ratio
collision_rejection_ratio
mean_representability_per_head
```

每个预注册 interval 可选记录 sentinel gradient：

```text
cos_side0_final ... cos_side3_final
negative_dot_count
```

在线 gradient audit 只作 sanity check；正式统计以 frozen offline audit 为准，避免显著拖慢训练和改变随机性。

---

## 19. 测试清单

### `tests/test_gradient_audit.py`

- `autograd.grad` 不写入 `.grad`；
- audit 前后参数 bitwise identical；
- BN buffers identical；
- identical vectors cosine=1；
- opposite vectors cosine=-1；
- zero norm finite handling；
- missing named parameter fail-closed；
- DataParallel name normalization。

### `tests/test_scale_ownership.py`

- component id permutation invariance；
- hard assignment one owner/component；
- fallback behavior；
- empty image；
- one-pixel target；
- large target assigned to coarser valid head；
- collision rejects coarse head；
- random control matches sparsity；
- area-only control ignores collision by design；
- deterministic under fixed seed。

### `tests/test_masked_aux_loss.py`

- unassigned target pixels have zero gradient；
- assigned positives remain valid after ignore dilation；
- ordinary background remains negative；
- empty-side `skip` gives no auxiliary gradient；
- target/valid shape mismatch raises；
- all-valid target matches unmasked scale-IoU numerically；
- no NaN on empty target；
- `aux_loss_weight=0` preserves final-only gradients。

### Legacy identity

- M0 forward loss exact；
- M0 logits gradients exact；
- same seed one-epoch loss sequence exact；
- full repository `pytest -q` passes。

---

## 20. 方法 GO / NO-GO

### 20.1 Scale Ownership GO

建议冻结：

1. M3 在至少 2/3 数据集上 `median ΔIoU >= 0`；
2. aggregate `ΔPd >= -0.002`；
3. FA 或 FP component count 至少 2/3 数据集改善；
4. M3 明显优于 M4 random matched；
5. M3 优于或至少不弱于 M5 area-only，若相同则采用更简单 M5；
6. 训练后 side-final negative-dot rate 相对 M2 下降；
7. 改善集中在 Phase A 预先识别的不可表达/冲突子集；
8. inference 参数量、FLOPs、延迟与 MSHNet 完全相同。

### 20.2 方法 NO-GO

- M3 与 random control 相同；
- 只减少 auxiliary loss，却不改善 final metrics；
- final-only 已经优于所有 ownership；
- 仅一个 dataset/seed 有效；
- 收益来自整体 loss 尺度变化；
- ownership 把绝大多数 component 都分给 side0，退化为删除 coarse supervision；
- Pd 下降而 FA 下降，形成普通保守化；
- 阈值 sweep 后收益消失。

### 20.3 简化原则

若 `area-only` 与完整 representability 相同：

> 保留 area-only，删除 quantization/collision 复杂度。

若 `side_no_location` 已取得全部收益：

> 结论应是 coarse side location supervision 不合适，不应继续包装成 Scale Ownership。

---

## 21. 主要风险与控制

| 风险 | 误判方式 | 对应控制 |
|---|---|---|
| 负 cosine 是小梯度噪声 | norm 极小也被计 conflict | 同时报 dot、norm；设置最小 norm 分层 |
| 只由 target area 解释 | representability 看似有效 | area-only regression/control |
| 稀疏 supervision 本身有效 | ownership 看似有效 | random matched control |
| 删除 side LLoss 有效 | ownership 看似有效 | side-no-location control |
| loss 尺度改变 | optimizer 步长不同 | legacy-rescaled control |
| unassigned target 被当背景 | 新冲突、Pd 下降 | valid/ignore mask unit test |
| all targets 退化到 side0 | 方法等同 final+side0 | assignment histogram 和 fallback gate |
| 组件标签受 augmentation 影响 | assignment 不稳定 | augmentation 后重新 label，记录 fragment policy |
| 与动态尺度加权重叠 | 新颖性不足 | 强调 instance-to-head topology 与冲突证据 |
| 只适配 MSHNet | 通用性弱 | 至少一个其他 deep-supervision backbone |

---

## 22. 不建议作为第一版的方法

### 22.1 PCGrad

PCGrad 可以作为 control，但不应先做主方法：

- 它是通用多任务梯度投影；
- 不能解释哪个目标在哪个尺度不可表达；
- 多次 backward/optimizer 包装复杂；
- 可能与 Adagrad 状态交互；
- 审稿人容易认为是直接套用。

只有在 Phase A 证明确有冲突、M3 不足时，才增加：

```text
M8 = PCGrad over final/side losses
```

### 22.2 Learned ownership/router

learned gate 容易全部选择 fine head 或 keep，历史 DEA 已出现动作坍缩。除非 deterministic ownership 已有效且有清晰 residual failure，否则不要实现。

### 22.3 修改 decoder

本线的论文命题是监督冲突。修改 decoder 会把优化来源混入 feature capacity，破坏归因。

---

## 23. Server B 提交序列

```text
B-00  pin source commit and define audit manifest
B-01  add loss decomposition without changing legacy numerics
B-02  add instance-scale representability ledger and tests
B-03  add frozen gradient audit and integrity tests
B-04  add OOF conflict/error analyzer and preregistered GO report
B-05  add training-time instance maps
B-06  add deterministic hard ownership and assignment tests
B-07  add masked auxiliary loss and ignore semantics tests
B-08  add training modes and matched controls
B-09  add runner, summarizer, paired bootstrap, final decision report
```

每个 commit 必须记录：

- source SHA；
- checkpoint SHA256；
- split SHA256；
- exact CLI；
- new tests；
- inference graph 是否改变；
- 参数量/FLOPs 是否改变。

---

## 24. 预期论文叙事

若备线成立，论文问题应写成：

> Deep supervision assumes every auxiliary resolution can faithfully represent every target. For infrared tiny targets, downsampled labels may survive but become geometrically unresolvable, producing gradients that conflict with the final objective. We diagnose this conflict and assign instance supervision only to representable scales.

候选题目：

```text
When Deep Supervision Conflicts: Resolution-Aware Scale Ownership for Infrared Small Target Detection
```

论文贡献必须包含：

1. 跨数据集、跨 seed 的梯度冲突测量；
2. 标签 survival 与 geometric representability 的区分；
3. 目标实例到 side head 的 supervision topology；
4. random matched、area-only、global scalar、final-only 等 controls；
5. inference 零开销和跨 backbone 验证。

若只发现“把 side loss 权重调小更好”，不要包装为 AAAI 方法。

---

## 25. Server B 最终交付物

```text
1. source_commit.txt
2. checkpoint_hash_manifest.json
3. gradient_records.jsonl
4. component_scale_records.jsonl
5. gradient_conflict_summary.md/json
6. OOF incremental-prediction report
7. Phase-A GO_NO_GO.md
8. M0–M7 experiment manifest
9. assignment histograms and per-image ledgers
10. paired bootstrap report
11. frozen final config.yaml
12. official-test unlock record（仅最终 GO 后）
```

### 最重要的执行原则

> **先证明“低分辨率监督对某类目标不可表达，并确实产生损害 final objective 的共享梯度”，再做 Scale Ownership；没有机制证据时，任何动态尺度模块都只是又一个不可辨识的权重器。**
