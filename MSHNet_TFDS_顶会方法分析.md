# 基于 MSHNet 的 Task-Feasible Deep Supervision 方法分析与顶会投稿重构建议

> **实现状态（2026-07-11）**：当前新增的 `tfds_projection` 路径只审计并使用
> `max-pool + nearest lift` 这一给定投影产生的任务一致性，因此运行名采用
> **TCDS-Projection**。它尚未求解辅助输出空间的最优可表达误差
> \(\epsilon^*\)，不能作为“完整 TFDS 已实现”或“Phase A 已 GO”的证据。
> 该路径只允许用于单元测试、投影审计和预注册 smoke；正式主方法训练仍需
> 等待实例级梯度 influence 与 one-step intervention 的 Phase-A 结论。

### 本轮代码落地与运行结果

已实现并验证：

- `model/task_consistent_supervision.py`：scene-level 投影恢复、binary graph、
  merge/split 全参与者拒绝、unknown 优先；
- `model/partial_sls_loss.py`：all-valid 时 bitwise 退化为 canonical SLS，partial
  区域无梯度泄漏；
- `main.py --deep-supervision tfds_projection`：保持
  \((L_f+\sum_sL_s)/5\) 的 canonical 聚合；
- `tools/audit_task_consistent_projection.py`：只审计给定投影，不宣称估计
  \(\epsilon^*\)；
- NUAA-SIRST 上完成 1 epoch、`warm_epoch=-1` 的可训练性 smoke；该结果不
  具有性能解释意义；完整仓库测试为 `118 passed`。

在 `min_iou=0.5`、定位容差 3 pixels 下，三个训练集的投影审计为：

| Dataset | stride 1 | stride 2 | stride 4 | stride 8 |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 100.00% | 67.04% | 10.74% | 0.37% |
| NUDT-SIRST | 100.00% | 92.27% | 15.58% | 0.22% |
| IRSTD-1K | 100.00% | 93.49% | 26.54% | 1.09% |

这组结果不是方法有效证据，反而暴露出一个必须先解决的设计风险：当前
`IoU>=0.5` 判据几乎删除 stride-8 的正监督。NUAA 上将阈值改为 0.25 和 0.10
时，stride-8 可行率分别变为 7.78% 和 32.96%，说明图密度对阈值高度敏感。
因此正式实验前必须：

1. 预注册阈值的任务来源，不能根据最终性能选阈值；
2. 报告每个 head 的图密度和有效正样本预算；
3. 加入 scale-drop / matched-budget control；
4. 若粗尺度长期接近空图，应将当前规则判为退化，而不是 TFDS 成功。

## 总体判断

当前研究方向具有明确价值，但现阶段的 RODS / Scale Ownership 更适合作为探索性原型，而不宜继续作为论文的核心创新。真正有机会形成顶会级贡献的，不是“尺度所有权”本身，而是下面这一更一般的学习原理：

> **深监督默认假设辅助输出空间能够表达原始任务标签；当某个实例超出辅助 head 的任务可表达空间时，对该实例施加确定性辅助监督并非普通近似误差，而是结构性错误监督。**

因此，建议把 Scale Ownership 降为该原则的一种实现结果，将论文主线重构为：

> **Task-Feasible Deep Supervision（TFDS）：辅助 head 仅监督其输出空间中任务可恢复的实例；无法可靠恢复的实例区域应被视为 unknown，而不是前景或背景。**

不过，还需要进一步区分两个问题：

1. 是辅助输出空间本身无法表达原始标签；
2. 还是当前标签下采样或投影算子设计得不够好。

如果不区分这两类误差，审稿人很容易认为该工作只是再次发现“小目标标签下采样会丢失信息”，并质疑使用概率保持标签、soft label 或 multi-label pooling 是否已经能够解决问题。

因此，最有力的论文主线应当是：

> **区分“投影方法造成的误差”和“辅助输出空间本身造成的不可恢复误差”，只在任务标签可由该输出空间充分恢复时施加确定监督。**

---

# 一、当前代码与研究流程的核心问题

## 1. Phase A 尚未完成，Phase B 已提前实现

当前文档中的研究流程要求：

1. 先完成多尺度梯度审计；
2. 验证低分辨率标签不可恢复是否真的产生有害辅助梯度；
3. 满足诊断 GO 条件后，再实现 ownership 或监督过滤机制。

但现有状态是：

- `tools/audit_multiscale_gradients.py` 不存在；
- `analyze_multiscale_gradients.py` 不存在；
- 没有形成 `gradient_records.jsonl`；
- 没有 OOF 增量预测；
- 没有正式的 Phase-A GO / NO-GO 结论；
- RODS 已经进入训练代码，并完成了多轮 smoke 训练。

因此，论文所需的因果链尚未闭合：

\[
\text{辅助输出空间不可恢复}
\Longrightarrow
\text{产生有害辅助梯度}
\Longrightarrow
\text{移除该监督可改善最终任务}
\]

当前实验最多只能说明：

> 一种改变了辅助监督分布和损失组合方式的训练策略可能有效。

但尚不能说明：

> 不可恢复实例导致了结构性错误梯度，而 TFDS 修复了这一问题。

这不是普通工程顺序问题，而是论文核心因果证据缺失。

---

## 2. 当前实现同时改变多个变量，无法归因

当前 `compute_deep_supervision_loss` 路径中，RODS 同时改变了：

- 哪些实例被分配给各个 side head；
- 哪些像素被设置为 ignore；
- side loss 从原始 SLS 变为 masked scale-IoU；
- location loss 是否存在；
- final loss 与 auxiliary loss 的相对权重；
- 各 head 的正样本数量；
- 各 head 的背景监督比例。

因此，即使 RODS 获得提升，也可能来自：

- 删除 LLoss；
- 降低辅助损失总权重；
- 减少困难正样本；
- 增加背景保守性；
- 改变优化尺度；
- 改变各 head 的监督预算。

这会使论文无法回答最关键的问题：

> 性能提升究竟来自 task-feasibility 语义，还是来自训练目标的其他变化？

主方法必须尽量只改变一个变量：**辅助标签的已知域**。

---

## 3. MSHNet baseline 必须物理隔离

当前 `MSHNet.py` 中额外加入了 `decidability_head`。即使该 head 在普通 baseline backward 中没有梯度，它仍然会改变：

- 参数量统计；
- `state_dict`；
- 模型类定义；
- checkpoint 兼容性；
- 对“完全复现原始 MSHNet”的表述。

建议将 canonical MSHNet 恢复成完全独立的 baseline：

```text
model/
├── baselines/
│   └── mshnet_canonical.py
├── adapters/
│   └── deep_supervision_adapter.py
└── experimental/
    └── archived_dea_heads.py
```

需要增加 baseline identity tests：

- `state_dict` key 与官方实现一致；
- 参数量一致；
- 固定输入的输出一致；
- 官方 checkpoint 可通过 `strict=True` 加载；
- baseline loss 计算一致；
- FLOPs 和推理时延一致。

TFDS 不需要增加网络 head，因此主方法不应包含 `decidability_head`、router 或额外 decoder。

---

## 4. 当前损失聚合改变了 final 与 side loss 的相对权重

canonical MSHNet 的五项监督可写为：

\[
L_{\mathrm{base}}
=
\frac{L_f+L_1+L_2+L_3+L_4}{5}
\]

当前替代形式近似为：

\[
L_{\mathrm{new}}
=
L_f+0.8\,\operatorname{mean}(L_1,\ldots,L_4)
=
L_f+0.2\sum_{s=1}^{4}L_s
\]

这两者并非只差一个全局常数。

在 canonical 形式中：

- 单个 side loss 与 final loss 的相对系数为 \(1:1\)。

在当前形式中：

- 单个 side loss 与 final loss 的相对系数为 \(0.2:1\)。

也就是说，final 相对于每个 side 的权重提高了 5 倍。

如果继续使用：

\[
L_f+\lambda_{\mathrm{aux}}\operatorname{mean}(L_s)
\]

要保持 canonical 的相对比例，应设置：

\[
\lambda_{\mathrm{aux}}=4
\]

而不是 0.8。

更合理的实现是直接保持原始平均方式：

\[
L_{\mathrm{TFDS}}
=
\frac{L_f+\sum_{s=1}^{4}L_s^{\mathrm{partial}}}{5}
\]

这样，TFDS 唯一改变的是辅助标签有效区域，而不是梯度总量和 final/side 比例。

---

## 5. soft responsibility 的数学语义不正确

当前 masked loss 中近似执行：

\[
v'=v\cdot w
\]

\[
p'=p\cdot v'
\]

\[
y'=y\cdot v'
\]

随后交集项为：

\[
\operatorname{intersection}
=
\sum p'y'
=
\sum p\,y\,v^2w^2
\]

因此正样本权重在 intersection 中实际成为 \(w^2\)，而非普通 weighted IoU 中预期的 \(w\)。

同时：

- 正样本梯度随 responsibility 非线性下降；
- 背景监督仍保持权重 1；
- 一个实例被分配到越多 head，每个 head 的正梯度越弱；
- 背景抑制没有同步减弱。

这会形成未声明的保守预测偏置。

第一版应完全删除：

- soft responsibility；
- interval responsibility；
- Gaussian scale preference；
- learned router。

只保留二值可行图：

\[
A_{k,s}\in\{0,1\}
\]

所有有效像素权重均为 1。

正确的 masked IoU 至少应采用：

\[
I=\sum_i p_i y_i v_i
\]

\[
U=\sum_i p_i v_i+\sum_i y_i v_i-I
\]

\[
L_{\mathrm{IoU}}
=
1-\frac{I+\epsilon}{U+\epsilon}
\]

不能把 `weight` 同时乘入 prediction 和 target 两侧。

---

## 6. hard ownership 与新的理论原则不一致

如果论文主张是：

> 每个辅助 head 监督其任务可表达的实例。

那么一个实例只要在多个 head 上均可恢复，就应在这些 head 上全部接受监督：

\[
A_{k,s}=1
\quad
\text{for every feasible }(k,s)
\]

没有理论理由要求：

\[
\sum_s A_{k,s}=1
\]

因此，主方法应使用 **binary feasibility graph**，而不是 hard owner。

例如：

- 大目标可能在四个 side head 上均可恢复，应保留四层监督；
- 中等目标可能只在前三个 head 可恢复；
- 极小目标可能只在最高分辨率 head 可恢复；
- 某些极端实例可能在所有 side head 上都不可恢复，但 final head 仍接受完整监督。

这能够解释：

- 深监督总体为何有效；
- 不可恢复实例的局部监督为何有害；
- TFDS 为什么不是简单删除深监督；
- 原始 MSHNet 中增加监督尺度整体上有效，与本文并不矛盾。

论文应强调：

> 深监督总体上有益，但其收益并不意味着所有实例—head 配对都具有正确监督语义。

---

## 7. 当前 representability score 仍是启发式模块堆叠

当前 score 由以下因素相乘：

- area；
- quantization；
- collision；
- Gaussian scale preference；
- 多个阈值和比例参数。

这很容易被审稿人概括为：

> 三到四个手工指标，加若干阈值，再乘一个尺度偏好核。

这类方法即使实验有效，也很难被视为统一创新。

此外，当前 collision 实现主要检测多个实例是否进入同一个 pooling cell，但并未完整实现：

- 投影后连接分量合并；
- 相邻 coarse cell 导致的拓扑连接；
- 一个原始实例被分裂；
- 多个实例在回升后连成同一 component；
- 匹配身份变化。

因此，必须用统一的任务恢复误差替换这些启发式模块。

---

## 8. assignment audit 的 decidable ratio 定义有误

当前 audit 中使用：

```text
decidability > 0
```

统计“可判定率”，但真正 assignment 使用：

```text
decidability >= min_decidability
```

因此，报告中的 decidable ratio 并不表示实例真正通过了训练时的决策阈值。

另外，原始 mask 审计没有经过训练时实际使用的：

- 随机 resize；
- crop；
- flip；
- 可能的 padding；
- batch 中真实输入尺寸变化。

因此，当前 assignment 分布不能代表训练期间的真实分布。

需要增加：

- augmentation-aware assignment audit；
- 每个 epoch 的在线统计；
- 各 head 实例数；
- 各 head 正像素数；
- valid ratio；
- unknown ratio；
- 目标相对 stride 的分布。

---

# 二、统一理论：Task-Feasible Deep Supervision

## 1. 定义辅助输出空间

设辅助 head \(s\) 的离散标签空间为：

\[
\mathcal Z_s
\]

回升到原始图像空间的算子为：

\[
U_s
\]

该 head 能表达的原分辨率标签集合为：

\[
\mathcal R_s
=
\left\{
U_sz\mid z\in\mathcal Z_s
\right\}
\]

对于实例 \(G_k\)，定义任务失真：

\[
D_T\left(G_k,\hat G;G_{\neg k}\right)
\]

其中 \(G_{\neg k}\) 表示场景中的其他实例。引入其他实例是必要的，因为任务失真必须能够感知：

- 目标消失；
- 中心偏移；
- 多实例合并；
- 实例分裂；
- 形状退化；
- 区域面积严重偏差。

---

## 2. 输出空间的最优可表达误差

定义：

\[
\epsilon^{*}_{k,s}
=
\inf_{z\in\mathcal Z_s}
D_T\left(
G_k,U_sz;G_{\neg k}
\right)
\]

它表示：

> 即使在该辅助输出空间中选择最优标签，该实例仍然至少产生多少任务误差。

当 \(\epsilon^{*}_{k,s}\) 很大时，说明问题不是投影算子不够好，而是该输出空间本身无法满足任务要求。

---

## 3. 当前投影算子的实际误差

设实际标签投影算子为：

\[
P_s
\]

则实际任务误差为：

\[
\epsilon^{P}_{k,s}
=
D_T\left(
G_k,
U_sP_s(Y);
G_{\neg k}
\right)
\]

由此可分解为：

\[
\epsilon^{P}_{k,s}
=
\underbrace{\epsilon^{*}_{k,s}}_{\text{representation gap}}
+
\underbrace{
\left(
\epsilon^{P}_{k,s}-\epsilon^{*}_{k,s}
\right)
}_{\text{projection gap}}
\]

这一区分非常关键。

| 情况 | 含义 | 对应解决方案 |
|---|---|---|
| \(\epsilon^*\) 小，projection gap 大 | 输出空间可以表达，但投影算子较差 | 改进 pooling、soft label、CPP、multi-label projection |
| \(\epsilon^*\) 大 | 无论如何投影，该输出空间都无法满足任务 | TFDS：将该实例区域设为 unknown |
| 两者都小 | 正常有效深监督 | 保留确定监督 |

这使论文不再停留在“标签下采样损失信息”，而是回答：

> 什么误差可以通过更好的标签投影修复，什么误差属于输出空间结构性限制？

---

## 4. 论文主张必须与实际估计方式一致

如果代码只计算：

\[
D_T\left(G_k,U_sP_s(Y)\right)
\]

那么严格能够证明的是：

> 当前 supervision operator 生成了任务不一致的辅助标签。

尚不能严格证明：

> 整个辅助输出空间均无法表达该标签。

要声称后者，必须近似求解：

\[
\epsilon^{*}_{k,s}
\]

例如在目标周围的有限 coarse cells 中搜索任务最优离散表示。

因此可分两个版本：

### 稳健第一版

方法表述：

> Task-Consistent Label Projection for Deep Supervision

使用实际投影后的任务恢复质量判断是否施加监督。

优点：

- 实现简单；
- 容易验证；
- 论文主张与代码一致；
- 适合作为第一轮 GO / NO-GO 实验。

### 更强完整版

方法表述：

> Task-Feasible Deep Supervision under Resolution-Limited Output Spaces

近似搜索 coarse output space 中的最优表达，估计 \(\epsilon^*\)，明确区分 projection defect 与 representation infeasibility。

建议先完成稳健第一版，确认机制成立后再实现局部最优表示搜索，避免再次演变成复杂模块堆叠。

---

# 三、用统一任务误差替代所有启发式指标

## 1. 直接在完整投影场景上评估任务恢复性

对完整标签图 \(Y\)：

1. 应用辅助 head 的真实投影算子：

\[
Y_s=P_s(Y)
\]

2. 回升到原始空间：

\[
\widetilde Y_s=U_s(Y_s)
\]

3. 对 \(\widetilde Y_s\) 重新提取连接分量；
4. 将原始实例与投影后实例进行一对一匹配；
5. 根据任务评价协议计算实例恢复质量。

可定义：

\[
r_{k,s}
=
\operatorname{IoU}
\left(
G_k,C_{\pi_s(k)}
\right)
\cdot
\mathbf 1
\left[
\left\|
 c(G_k)-c(C_{\pi_s(k)})
\right\|_2<d_T
\right]
\]

其中：

- \(C_{\pi_s(k)}\) 为与原始实例 \(G_k\) 匹配的投影实例；
- 未匹配实例的 \(r_{k,s}=0\)；
- \(d_T\) 使用任务评价协议中的定位容差；
- 匹配应采用一对一分配，而非独立最近邻。

最终定义可行图：

\[
A_{k,s}
=
\mathbf 1[r_{k,s}\ge\tau_T]
\]

---

## 2. 四类问题自然统一

采用任务恢复质量后：

- **目标消失**：没有可匹配 component，\(r=0\)；
- **中心量化严重**：中心距离不满足阈值；
- **多实例合并**：一对一匹配后至少一个实例无法匹配或 IoU 显著下降；
- **形状退化**：区域 IoU 下降；
- **目标分裂**：单个实例无法与多个 fragment 同时形成有效一对一匹配；
- **面积异常**：自然反映到 IoU 或区域误差中。

因此不再需要：

- `q_area`；
- `q_quant`；
- `q_merge`；
- Gaussian scale preference；
- preferred diameter；
- interval ratio；
- responsibility weight；
- learned router。

方法只保留一个统一对象：

\[
\text{任务投影后的实例恢复误差}
\]

这比手工指标乘积更具有可解释性和可推广性。

---

# 四、TFDS 的最小训练实现

本节分为两个层次：当前可执行的是给定投影算子的 **TCDS-Projection**；只有
近似求解 \(\epsilon^*\) 后，才升级命名为完整 **TFDS**。二者共享 partial-label
训练接口，但论文主张不能混用。

## 1. 构造辅助 head 的部分标签

对于辅助 head \(s\)，将可行实例集合定义为：

\[
\mathcal F_s
=
\{k\mid A_{k,s}=1\}
\]

不可行实例集合定义为：

\[
\mathcal U_s
=
\{k\mid A_{k,s}=0\}
\]

可行正标签：

\[
Y_s^{+}
=
P_s\left(
\bigcup_{k\in\mathcal F_s}G_k
\right)
\]

unknown 区域：

\[
Y_s^{?}
=
P_s\left(
\bigcup_{k\in\mathcal U_s}G_k
\right)
\]

有效监督 mask：

\[
V_s=1-Y_s^{?}
\]

其中：

- \(Y_s^+\) 表示可恢复实例的正标签；
- \(Y_s^?\) 表示任务上无法可靠确定的区域；
- \(V_s\) 表示已知监督域；
- 未被 unknown 覆盖的真实背景仍然是有效负样本。

---

## 2. unknown 必须优先于 positive

如果可行实例和不可行实例在同一个 coarse cell 中产生冲突，应采用：

\[
V_s(i)=0
\]

也就是 unknown 优先。

不能执行：

1. 先将该 cell 标记为 ignore；
2. 后因存在 positive target 又将其恢复为 valid。

一个 coarse cell 同时承载多个任务身份时，其实例级语义本身就是不可判定的。

第一版也不建议对 unknown 额外 dilation，因为 dilation 会引入：

- 新超参数；
- 额外监督预算变化；
- 更大的归因困难。

应先使用投影算子本身产生的精确 unknown 支持区域。

---

## 3. 保持原始损失族不变

主方法应实现：

\[
L_{\mathrm{PartialSLS}}(p,y,V)
\]

并满足严格退化性质：

\[
L_{\mathrm{PartialSLS}}(p,y,\mathbf 1)
=
L_{\mathrm{SLS}}(p,y)
\]

这必须成为单元测试，而不是经验近似。

### Scale-IoU 部分

只在 \(V=1\) 的像素上计算。

### Location loss 部分

同样只使用有效预测区域和有效标签区域。

### 没有有效正实例时

- 对纯 IoU/SLS，只有背景而没有有效正实例时不存在有意义的交集项，因此该
  side-image 的预测梯度应为零；不能声称它仍提供了有效背景监督；
- location term 必须置零；
- 不额外加入 BCE 或 hard-negative loss，否则会改变原始损失族并破坏归因；
- 该 side term 保留为固定零梯度项，五项损失的平均分母仍保持为 5，从而不
  改变 final 与各 side objective 的名义聚合比例；
- 当至少存在一个有效正实例时，已知背景仍通过 IoU union 参与监督。

为同时满足 baseline identity，存在一个明确例外：若 `valid` 全为 1，则无论
target 是否为空，都直接调用 canonical SLS，保留基线的原始数值行为；上述
“零梯度 side term”只适用于 partial/unknown 域已经实际出现的情况。

这样能够确保：

\[
A_{k,s}\equiv1
\Rightarrow
L_{\mathrm{TFDS}}=L_{\mathrm{canonical\ MSHNet}}
\]

TFDS 相对 baseline 只改变：

> 哪些辅助标签位置被视为确定标签。

不改变：

- 网络结构；
- 推理路径；
- 损失族；
- final/side 相对权重；
- head 数量；
- decoder。

这正是避免模块堆叠的关键。

---

# 五、梯度证据的完整设计

## 1. 同图像 side-final cosine 不能作为主证据

MSHNet 的 final output 直接融合多个 side predictions，因此：

\[
\cos(g_s,g_f)
\]

会受到计算图结构耦合影响。

它可以作为描述性统计，但不足以证明某个实例的 side supervision 对最终任务有害。

同时，动态辅助权重、梯度冲突、PCGrad、模块级辅助影响学习已有较多先例。因此本文不能把以下内容作为主要创新：

- cosine 分析；
- PCGrad；
- 动态 loss weight；
- auxiliary gradient projection。

论文差异必须落在：

> 分辨率受限辅助输出空间产生的任务标签不可行性。

---

## 2. 单实例图像直接审计

对于只包含一个目标实例的图像，可直接计算该实例在 head \(s\) 上的辅助梯度：

\[
g_{k,s}
=
\nabla_{\theta_b}L_s(G_k)
\]

其中 \(\theta_b\) 只选择共享 backbone 或共享 decoder 参数，避免使用 side head 私有层。

再在独立 probe batch 上计算：

\[
g_f^{\mathrm{probe}}
=
\nabla_{\theta_b}L_f(B_{\mathrm{probe}})
\]

分析：

\[
g_{k,s}^{\top}g_f^{\mathrm{probe}}
\]

---

## 3. 多实例图像使用 leave-one-component-out

对多实例图像中的实例 \(G_k\)，定义：

\[
\Delta g_{k,s}
=
\nabla_{\theta_b}L_s(Y)
-
\nabla_{\theta_b}L_s(Y\setminus G_k)
\]

它近似提取实例 \(G_k\) 对 head \(s\) 辅助梯度的边际贡献。

然后定义跨 batch influence：

\[
I_{k,s}
=
\Delta g_{k,s}^{\top}g_f^{\mathrm{probe}}
\]

若执行一步辅助梯度下降：

\[
\theta'
=
\theta-\eta\Delta g_{k,s}
\]

则 probe final loss 的一阶变化近似为：

\[
L_f^{\mathrm{probe}}(\theta')
-
L_f^{\mathrm{probe}}(\theta)
\approx
-\eta I_{k,s}
\]

因此：

- \(I_{k,s}>0\)：该辅助更新预计改善 probe final loss；
- \(I_{k,s}<0\)：该辅助更新预计恶化 probe final loss。

这比同 batch cosine 更接近论文所需的“有害辅助监督”定义。

---

## 4. 使用有限差分验证一阶 influence

对部分代表性实例—head 对，复制模型参数并真实执行一步 isolated auxiliary update：

\[
\theta'
=
\theta-\eta\Delta g_{k,s}
\]

再计算：

\[
\Delta L_f^{\mathrm{probe}}
=
L_f^{\mathrm{probe}}(\theta')
-
L_f^{\mathrm{probe}}(\theta)
\]

比较：

\[
\operatorname{sign}(\Delta L_f^{\mathrm{probe}})
\]

与：

\[
\operatorname{sign}(-I_{k,s})
\]

是否一致。

这能够验证一阶梯度影响分析是否真的对应优化后的 final objective 变化。

---

## 5. 多阶段审计

至少在以下阶段执行审计：

- 初始化；
- warm-up 结束；
- 训练中期；
- best checkpoint；
- final checkpoint。

只分析最终模型会产生幸存者偏差，因为模型可能已经学会降低对错误辅助标签的敏感度。

需要报告：

- 可行边和不可行边的 influence 分布；
- 负 influence 比例；
- influence 均值和中位数；
- 不同 head 的差异；
- 不同目标相对尺度的差异；
- 不同训练阶段的变化。

---

## 6. 受控尺度干预

仅按数据集中的目标大小分组可能混入：

- 目标对比度；
- 背景复杂度；
- 数据集偏差；
- 实例数量；
- 标注噪声。

建议构造同场景尺度干预：

1. 从同一图像提取目标和局部背景；
2. 保持背景、对比度和噪声机制一致；
3. 只连续改变目标尺寸；
4. 计算目标相对 head stride 的比例；
5. 观察 recoverability、gradient influence 和 TFDS 收益是否在同一边界附近发生变化。

核心变量应是：

\[
\rho_{k,s}
=
\frac{\text{object size}_k}{\text{stride}_s}
\]

论文应证明：

\[
\text{harmfulness follows relative resolution}
\]

而不仅是：

\[
\text{small objects are harder}
\]

---

## 7. Phase-A GO 条件

建议在正式实现 TFDS 训练前，提前固定 GO 条件：

1. 不可行边的负 influence 比例显著高于可行边；
2. 至少两个数据集方向一致；
3. 多个随机种子结果一致；
4. 控制目标面积、SCR、实例数量后仍显著；
5. 现象存在于共享 backbone / decoder 参数；
6. one-step finite difference 支持一阶 influence；
7. matched random graph 无法复现同样收益。

如果这些条件不成立，应 pivot 到：

> 改进辅助标签投影算子。

而不是继续扩展 ownership、router、soft responsibility 或复杂 loss weighting。

---

# 六、matched random graph 的正确设计

## 1. 当前 random control 不 matched

当前 random assignment 只是对每个实例从候选 head 中均匀随机选择一个，因此不能保持：

- 每个实例拥有的 head 数；
- 每个 head 的实例数量；
- 每个 head 的正像素数量；
- 每个 head 的目标尺度分布；
- valid / unknown 像素比例。

因此，hard-vs-random 差异仍然混入监督预算差异。

---

## 2. 使用二部图 edge swap

对于 binary feasibility graph，可使用 2×2 edge swap：

\[
(k_1,s_1),(k_2,s_2)
\rightarrow
(k_1,s_2),(k_2,s_1)
\]

该操作天然保持：

- 每个实例的 degree；
- 每个 head 的实例数量。

为了进一步保持正像素预算，可加入：

- 只在同一目标面积 bin 内交换；
- 只在相近 perimeter / diameter bin 内交换；
- 拒绝导致某 head 正像素总量超过容差的交换；
- 使用迭代优化最小化各 head 正像素差异。

最终需要报告每个 head 的：

- 实例数；
- 正像素数；
- valid pixel ratio；
- unknown pixel ratio；
- 目标面积分布；
- 目标相对 stride 分布。

M4 必须满足：

\[
\text{supervision budget}_{\mathrm{random}}
\approx
\text{supervision budget}_{\mathrm{TFDS}}
\]

两者唯一差别应是：

> 图中的边是否具有 task-feasibility 语义。

---

# 七、最小顶会实验矩阵

第一阶段建议只保留以下实验：

| 编号 | 方法 | 作用 |
|---|---|---|
| M0 | canonical MSHNet | 原始可复现 baseline |
| M1 | final-only | 判断深监督总体价值 |
| M2 | side-no-location | 隔离 LLoss 影响 |
| M3 | TFDS：binary feasible graph + Partial-SLS + canonical averaging | 唯一主方法 |
| M4 | degree/pixel/size-bin matched random graph | 验证可行图语义 |
| M5 | 改进标签投影 control，如 CPP / multi-label / soft projection | 区分 projection gap 与 representation gap |

M5 很重要。如果缺少这一项，审稿人会质疑：

> 为什么不改善标签投影，而要将区域设为 unknown？

第一轮不要同时加入：

- area-only；
- global scalar；
- PCGrad；
- hard owner；
- interval owner；
- soft responsibility；
- learned router；
- decoder modification。

只有在 M3 相对 M0、M2、M4、M5 明确成立后，再增加少量机制 ablation。

---

# 八、跨模型与跨任务验证

## 1. 必须解除只允许 MSHNet 的限制

如果只在 MSHNet 和三个 IRSTD 数据集上成立，论文容易被评价为：

> 针对 CVPR 2024 baseline 的训练修补。

至少应增加一个原生具有多个 auxiliary outputs 的非 MSHNet 网络。

建议通过统一 adapter 暴露：

```python
from dataclasses import dataclass
from typing import Callable, Tuple

@dataclass
class SupervisionHead:
    logits: object
    stride: Tuple[int, int]
    projector: Callable
    name: str
```

TFDS 只依赖：

- side logits；
- stride；
- 标签投影算子；
- final logits。

不依赖特定 backbone 内部结构。

---

## 2. 最好增加一个非 IRSTD 小结构任务

可考虑：

- 小病灶分割；
- 细胞实例或微小目标分割；
- 细长结构分割；
- 小器官或小缺陷检测。

如果资源不足，则论文主张必须收窄为：

> resolution-limited deep supervision for infrared small-target segmentation

不能直接泛化为所有 dense prediction 任务。

---

# 九、建议的代码目录重构

```text
model/
├── baselines/
│   └── mshnet_canonical.py
├── adapters/
│   └── deep_supervision_adapter.py
└── experimental/
    └── archived_dea_heads.py

supervision/
├── label_projection.py
├── task_matching.py
├── task_recoverability.py
├── feasibility_graph.py
├── partial_target.py
└── matched_random_graph.py

losses/
└── partial_sls.py

tools/
├── audit_instance_gradients.py
├── audit_projection_recoverability.py
├── analyze_gradient_influence.py
├── run_one_step_interventions.py
└── audit_supervision_budget.py

tests/
├── test_mshnet_identity.py
├── test_partial_sls_identity.py
├── test_scene_level_merge.py
├── test_feasibility_graph.py
├── test_unknown_priority.py
├── test_matched_random_marginals.py
└── test_projection_gap.py
```

---

# 十、建议的实施顺序

当前仓库落地状态：P0 已执行；P3 的 projection-consistent 子集已实现为
`tfds_projection`，但仅用于验证训练接口，不能越过 P2 的 GO 门控。P1 的
canonical baseline 物理隔离和 P2 的因果审计仍是正式实验前置条件。

## P0：冻结当前 RODS

保留：

- 当前代码；
- smoke 结果；
- assignment audit；
- 现有测试。

但停止继续调整：

- Gaussian preference；
- threshold；
- interval；
- responsibility；
- router。

当前 RODS 应作为 exploratory archive，而不是继续扩展的主分支。

---

## P1：恢复 canonical baseline

完成：

- 官方 MSHNet 代码隔离；
- strict checkpoint load；
- 参数量一致；
- 输出一致；
- loss 一致；
- 训练配置一致。

在这一阶段前，不进行主方法对比实验。

---

## P2：完成 Phase A 机制审计

实现：

- 单实例梯度审计；
- leave-one-component-out；
- independent probe influence；
- 多阶段 checkpoint audit；
- one-step finite-difference intervention；
- scene-level projection recoverability；
- augmentation-aware assignment audit。

形成明确 GO / NO-GO 报告。

---

## P3：实现最小 TFDS

只实现：

- binary feasibility graph；
- partial label / unknown mask；
- Partial-SLS；
- canonical five-loss averaging。

不增加任何网络参数。

---

## P4：实现 matched controls

完成：

- degree-preserving random graph；
- pixel-budget matching；
- size-bin matching；
- improved label projection baseline。

---

## P5：扩展模型与受控实验

完成：

- 非 MSHNet backbone；
- 受控目标尺度实验；
- 相对 stride 分析；
- 至少一个额外任务或明确收窄论文主张。

---

# 十一、论文贡献的推荐表述

## 贡献 1：现象与机制

揭示：

> 深监督虽然整体有益，但当辅助标签投影在任务意义上不可恢复时，特定实例—head 配对会产生对独立最终目标有害的共享梯度。

---

## 贡献 2：统一理论定义

将辅助监督误差区分为：

- projection gap；
- representation gap。

由此明确：

- 哪些问题应通过更好的标签投影修复；
- 哪些问题应通过 unknown partial supervision 处理。

---

## 贡献 3：参数自由的训练原则

提出 TFDS：

- final head 监督全部实例；
- auxiliary head 监督所有任务可行实例；
- 不可行实例区域设为 unknown；
- 不改变网络；
- 不改变推理；
- 不增加 learned router；
- 不使用 soft responsibility；
- 不修改 decoder。

---

## 贡献 4：因果与泛化证据

通过以下证据验证：

- 实例级 leave-one-out influence；
- 独立 probe batch；
- one-step parameter intervention；
- 多训练阶段审计；
- 受控相对尺度实验；
- matched random graph；
- 非 MSHNet backbone。

---

# 十二、标题方向

可考虑：

> **When Deep Supervision Lies: Task-Feasible Auxiliary Labels for Tiny-Target Segmentation**

或：

> **Task-Feasible Deep Supervision under Resolution-Limited Output Spaces**

或更稳健地表述为：

> **Task-Consistent Deep Supervision for Resolution-Limited Tiny-Target Segmentation**

如果尚未严格估计输出空间最优误差 \(\epsilon^*\)，建议优先使用 `Task-Consistent`，避免过度声称完整的 representation infeasibility。

---

# 十三、最终结论

当前项目可以判断为：

> **salvageable + needs-mechanism + needs-evidence + needs-literature-positioning**

当前定向测试主要证明：

- 代码能够运行；
- 基础逻辑未发生明显崩溃；
- 已实现模块可以通过 smoke test。

但尚未证明：

- baseline identity；
- matched random；
- 完整 collision / component merge；
- 标签不可恢复与有害梯度之间的因果链；
- 性能收益来自 task-feasibility 语义；
- 方法可以泛化到非 MSHNet 网络。

最值得保留的不是：

- ownership；
- Gaussian preference；
- decidability head；
- soft responsibility；
- 多个 handcrafted quality term。

而是下面这一单一原则：

\[
\boxed{
\text{辅助监督只能在辅助输出空间具有任务充分性的区域被视为确定标签；}
\\
\text{其余区域应为 unknown，而不能被错误编码为前景或背景。}
}
\]

工程上的首要动作不应继续训练更多 RODS 版本，而应依次完成：

\[
\text{canonical baseline isolation}
\rightarrow
\text{Phase-A causal audit}
\rightarrow
\text{binary TFDS}
\rightarrow
\text{matched controls}
\rightarrow
\text{cross-backbone validation}
\]

只有按照这一顺序，才能将当前的启发式实现收敛为一个具有：

- 明确问题定义；
- 统一理论；
- 最小方法；
- 可证伪假设；
- 因果证据链；
- 跨模型推广性；

的顶会级研究工作。
