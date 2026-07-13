# MSHNet–SDRR 第三轮代码、机制与投稿复核

> 复核对象：`Arialliy/dea3090` 默认分支  
> 复核日期：2026-07-12  
> 当前可见最新提交：`9e8d8e9`  
> 原始复核说明：本轮最初基于仓库公开代码、测试与实验记录进行静态复核；当时没有
> 在本地重新执行仓库所报告的 `187 passed` 或重新训练模型。文末“本地执行响应”记录
> 了随后实际完成的代码修改、测试和 checkpoint 审计；两者应按时间区分，不能把最初
> 的静态引用误当成独立复现。

---

## 1. 总体结论

当前研究主线已经不再是 TCDS/TFDS，而是：

# Scale-Deletion Responsibility Refinement（SDRR）

这条线比此前的 Scale Ownership、TCDS 和多种深监督筛选分支更接近一个可以投稿的统一方法，原因是：

1. 不新增推理模块；
2. 不改变 MSHNet 的部署图；
3. 直接利用原生四尺度线性融合；
4. 方法事件可以被明确写成一个可证伪的删除翻转条件；
5. 没有 Gaussian preference、soft responsibility、learned router 等模块堆叠；
6. 训练前半段可以严格保持 canonical objective；
7. 无责任事件时正则项严格为零。

但当前状态仍应定义为：

> **promising mechanism candidate，而不是已经完成归因的顶会方法。**

论文成败取决于一个问题：

> SDRR 的收益究竟来自“删除某尺度后决策翻转”这一特有语义，还是来自普通安全背景硬负样本挖掘、稀疏晚期正则、梯度脉冲、尺度贡献幅值筛选或训练轨迹稳定化？

在 matched controls 完成以前，不能把当前结果解释为责任机制已被验证。

---

## 2. 当前方法的真实数学语义

MSHNet 将四个 side logits 回升到同一分辨率，并经过最终线性卷积融合。记：

\[
z(x)=b+\sum_{i=0}^{3}c_i(x),
\]

其中：

\[
c_i=W_i * s_i
\]

是最终融合卷积中第 \(i\) 个输入尺度的真实卷积贡献。

代码通过 grouped convolution 计算四个 \(c_i\)，然后构造：

\[
z^{-i}(x)=z(x)-c_i(x).
\]

在固定全部中间 logits、只将最终融合输入的第 \(i\) 个通道置零的条件下，这一删除是代数精确的。

### 2.1 安全背景

代码使用 ground-truth mask 的方形 max-pool dilation：

\[
B_{\mathrm{safe}}
=
1-
\operatorname{Dilate}_{K}(Y),
\]

默认 `safe_kernel=15`，即以 Chebyshev 距离形成约 7 像素保护半径。

### 2.2 删除翻转责任事件

对安全背景像素，定义：

\[
r_i(x)=
\mathbf 1[x\in B_{\mathrm{safe}}]
\mathbf 1[z(x)>0]
\mathbf 1[z(x)-c_i(x)\le 0].
\]

它表示：

- 完整模型在该像素给出正决策；
- 删除尺度 \(i\) 的最终融合贡献后，决策变成非正；
- 该像素不位于目标及其保护邻域。

### 2.3 正则项

当前实现为：

\[
L_R
=
\frac{
\sum_{x,i}r_i(x)\operatorname{softplus}(c_i(x))
}{
\max(1,\sum_{x,i}r_i(x))
}.
\]

责任掩码停止梯度，因此正则只作用于已被选中的 \(c_i\)。

总目标近似为：

\[
L
=
L_{\mathrm{MSHNet}}
+
\lambda(t)L_R,
\]

其中正式 NUAA 配置记录为：

\[
\lambda=0.05,\qquad
t_{\mathrm{start}}=250,\qquad
t_{\mathrm{ramp}}=50,\qquad
K=15.
\]

---

## 3. 该公式实际施加了什么梯度

责任事件满足：

\[
z>0,\qquad z-c_i\le0.
\]

因此必有：

\[
c_i\ge z>0.
\]

对一个责任事件：

\[
\frac{\partial L_R}{\partial c_i}
=
\frac{\sigma(c_i)}{N_R},
\]

其中 \(N_R=\sum r_i\)。因为 \(c_i>0\)，所以：

\[
\sigma(c_i)>0.5.
\]

梯度下降会单调降低该尺度的正贡献。

这说明 SDRR 的最准确解释不是宽泛的“因果学习”，而是：

> **利用最终加性融合上的单尺度 but-for pivotality，选择并抑制安全背景上的阈值关键正贡献。**

或者更直接地说：

> **SDRR 是由尺度删除关键性选择的稀疏 hard-negative contribution regularization。**

这一定义本身是清晰的，也具有研究价值；但必须通过 matched controls 证明“pivotality”比普通 hard-negative selection 更有信息。

---

## 4. “精确删除”的可声明边界

论文可以写：

> 对固定的多尺度 logits，在线性最终融合层中，将某个尺度通道置零的干预可以由 \(z-c_i\) 精确计算。

论文不能写：

> 删除了网络中的某个尺度分支，并得到了完整模型的真实反事实输出。

原因是当前方法没有：

- 重新执行 encoder/decoder；
- 删除共享特征路径；
- 允许其他尺度对删除发生响应；
- 干预数据生成机制。

因此，“exact”只成立于：

\[
\text{fixed-intermediate-logit final-fusion intervention}.
\]

它是模型内部代数干预，不是数据因果识别。

### 4.1 浮点阈值风险

虽然公式在线性层上代数成立，但代码使用：

\[
z-c_i
\]

而不是实际构造“第 \(i\) 个通道置零后再次调用 final convolution”。两种计算在浮点加法顺序上可能存在微小差异。

由于责任条件在 0 处硬阈值化，极小误差也可能改变事件集合。

正式版本应增加：

\[
z>\delta,\qquad z^{-i}<-\delta,
\]

并至少审计：

\[
\delta\in
\{0,10^{-6},10^{-4},10^{-3},10^{-2}\}.
\]

同时比较：

1. `z - contribution_i`；
2. 真实 zero-channel final-convolution；

两者的：

- 最大绝对误差；
- 责任 mask mismatch rate；
- margin-conditioned mismatch rate。

“重构误差小于某个容差”不足以保证阈值事件完全一致。

---

## 5. 当前实现最严重的隐式优化偏差：按事件数归一化

当前：

\[
L_R
=
\frac1{N_R}
\sum_{e=1}^{N_R}
\operatorname{softplus}(c_e).
\]

责任梯度的 \(L_1\) 总量约为：

\[
\left\|\nabla_cL_R\right\|_1
=
\frac1{N_R}
\sum_e\sigma(c_e)
\in(0.5,1).
\]

这意味着：

> 一个事件和一万个事件都可能获得近似相同的总 \(L_1\) 梯度预算。

其 \(L_2\) 范数近似为：

\[
\left\|\nabla_cL_R\right\|_2
=
O(N_R^{-1/2}).
\]

因此事件越少，单个事件的梯度越集中。一个极少出现的责任事件也可能对优化器产生完整强度的“kick”。

仓库记录本身已经发现，早期设置中责任事件很稀疏，但单个事件仍获得完整权重；正式三种子中，有责任事件的 batch 比例也只有约：

\[
5.4\%,\quad 4.4\%,\quad 4.0\%.
\]

这使下面两种解释都成立：

1. SDRR 找到了少量但真正关键的尺度错误；
2. SDRR 只是通过晚期稀疏梯度脉冲改变了优化轨迹。

当前证据无法区分二者。

### 5.1 必做归一化控制

至少比较：

#### Event mean（当前）

\[
L_R^{\mathrm{event}}
=
\frac{\sum r_i\ell_i}{\sum r_i}.
\]

#### Safe-pixel density

\[
L_R^{\mathrm{density}}
=
\frac{\sum r_i\ell_i}{|B_{\mathrm{safe}}|}.
\]

#### Unique responsible pixel mean

设：

\[
P_R=\{x:\sum_i r_i(x)>0\},
\]

则：

\[
L_R^{\mathrm{pixel}}
=
\frac1{|P_R|}
\sum_{x\in P_R}
\frac1{d(x)}
\sum_i r_i(x)\ell_i(x),
\]

其中：

\[
d(x)=\sum_i r_i(x).
\]

第三种同时消除一个像素被多个尺度重复计权的问题。

这些应作为归因控制，而不是并列扩展为多个主方法。

---

## 6. 多尺度责任度数仍未被充分分析

一个像素可能同时满足多个：

\[
z-c_i\le0.
\]

因此：

\[
d(x)=\sum_i r_i(x)
\]

可能大于 1。

当前 event-wise mean 会让 degree 更高的像素重复进入损失。于是方法不仅在选择“哪些像素”，也隐式选择“一个像素应被处罚几次”。

现有 audit 已记录：

- 每尺度 event count；
- event share；
- mean contribution；
- full/deleted margin；
- `events_per_responsible_pixel`；
- active images；
- reconstruction error。

但还缺少：

- 完整 \(d(x)\) 直方图；
- 每个 degree 的梯度贡献；
- 各尺度组合频率；
- event-wise 与 pixel-wise normalization 的性能对照。

如果绝大多数 \(d(x)=1\)，问题较小；如果大量 \(d(x)>1\)，则当前增益可能来自隐式 degree weighting。

---

## 7. `safe_kernel=15` 的研究风险

当前安全区来自方形 max-pool，而不是任务评价中的欧氏距离或连接分量匹配语义。

因此 K=15 实际定义的是：

> 目标周围 Chebyshev 半径约 7 像素内的所有预测均不进入 SDRR。

该设计有三个问题：

1. K=7 已经实验失败，K=15 被保留，因此 K 带有 NUAA 选择痕迹；
2. 它可能保护真实目标周围的 halo，也可能放过评价指标会计为 FA 的近目标错误；
3. 不同数据集目标尺寸和标注形态不同，固定 15 的语义并不恒定。

应报告：

\[
K\in\{7,15,31\}
\]

的敏感性，并在所有外部数据集冻结 K。

更原则化的实现可以使用距离变换：

\[
B_{\mathrm{safe}}(x)
=
\mathbf 1[d(x,Y)>d_0],
\]

其中 \(d_0\) 与任务的 component matching tolerance 对齐，而不是根据某个数据集的性能选择。但在主归因未完成前，不建议立即再改主方法；先将其作为 sensitivity/control。

---

## 8. baseline 尚未物理隔离

当前 `model/MSHNet.py` 仍包含：

- `decidability_head`；
- `build_dea_lite_outputs`；
- `return_dea`；
- `fusion_alpha`；
- DEA 相关 forward 分支。

即使普通 baseline forward 不使用这些参数，也会污染：

- 参数量；
- `state_dict`；
- checkpoint strict identity；
- 模型类定义；
- 对“官方 MSHNet 完全一致”的表述。

此外，`AdaptiveMaxPool2d(1)` 被替换为：

```python
torch.amax(x, dim=(-2, -1), keepdim=True)
```

两者前向通常等价，但在最大值出现并列时，梯度分配规则未必等价。因此这不是严格的 backward identity。

### 8.1 推荐结构

```text
model/
├── baselines/
│   ├── mshnet_official.py
│   └── mshnet_deterministic.py
├── evidence/
│   └── additive_scale_view.py
└── regularizers/
    └── sdrr.py
```

其中：

- `mshnet_official.py` 尽可能逐行保持官方实现；
- `mshnet_deterministic.py` 明确标注 deterministic-backward variant；
- SDRR 通过外部 evidence adapter 读取 side logits 和 final conv；
- canonical class 中不再出现实验 head。

### 8.2 必须增加的 identity tests

1. 官方 checkpoint 能够 `strict=True` 加载；
2. `state_dict` key 集合一致；
3. 参数量一致；
4. 固定输入前向输出一致；
5. canonical 五项损失一致；
6. official 与 deterministic 版本分别报告；
7. 对并列最大值样本显式测试 backward 差异。

在完成前，论文应写：

> deterministic MSHNet implementation used consistently for all paired comparisons

而不是：

> exact official MSHNet baseline。

---

## 9. 正式 NUAA 三种子结果的重新解读

仓库记录的 400-epoch 配对结果为：

| seed | baseline IoU / PD / FA | SDRR IoU / PD / FA | paired IoU Δ |
|---|---|---|---:|
| 20260711 | 0.7369 / 0.9630 / 9.2262 | 0.7324 / 0.9630 / 6.0325 | -0.0045 |
| 20260712 | 0.6934 / 0.9444 / 1.7743 | 0.7350 / 0.9630 / 6.7423 | +0.0416 |
| 20260713 | 0.7250 / 0.9630 / 10.2908 | 0.7359 / 0.9815 / 5.3228 | +0.0109 |
| mean | 0.7184 / 0.9568 / 7.0971 | 0.7344 / 0.9692 / 6.0325 | +0.0160 |

### 9.1 可以确认的积极信号

- 三种子平均 IoU 提升 0.0160；
- 三种子平均 PD 提升；
- 三种子平均 FA 降低；
- 2/3 种子 IoU 为正；
- SDRR IoU 样本标准差记录为 0.0018，baseline 为 0.0225；
- 去掉任意一个种子后的 paired mean 仍为正：
  - 去掉 seed 11：+0.02625；
  - 去掉 seed 12：+0.00320；
  - 去掉 seed 13：+0.01855。

所以结果不是“去掉某一个种子后立即变成负均值”。

### 9.2 仍不能视为统计闭环

三个 paired delta：

\[
[-0.0045,\ 0.0416,\ 0.0109]
\]

的样本标准差约为：

\[
0.02347.
\]

探索性 paired t 统计为：

\[
t\approx1.18,\qquad df=2,
\]

双侧 \(p\approx0.359\)，95% t 区间约为：

\[
[-0.0423,\ 0.0743].
\]

在 \(n=3\) 时，t 分布假设本身也非常不稳定，因此这不是正式显著性检验，只说明不确定性仍然很大。

### 9.3 最大增益种子与直接机制不一致

seed 20260712 的 IoU 提升最大：

\[
+0.0416,
\]

但 FA 从：

\[
1.7743\rightarrow6.7423
\]

反而恶化。

这意味着最大的收益不能被简单解释为：

> SDRR 直接删除了安全背景 false alarms，所以 IoU 提升。

更可能的解释包括：

- 晚期稀疏梯度改变了训练轨迹；
- SDRR 提高了目标响应或形状质量；
- SDRR 避免了某个坏 basin；
- best-epoch selection 捕获了不同轨迹的峰值；
- 责任事件具有间接优化作用，而非只作用于最终 FA。

这并不否定方法，但迫使论文把“直接错误抑制”和“优化轨迹效应”分开验证。

### 9.4 当前文档存在 Pareto 表述冲突

同一阶段文档后部仍保留：

> 不能说已经获得 Pareto 改善，平均 FA 从 9.70 增至 24.72。

这对应早期 80-epoch 候选，而正式 400-epoch 三种子均值实际上在 IoU、PD、FA 三项上都改善。

应改为：

> 正式 NUAA 三种子均值形成 aggregate Pareto improvement，但并非每个种子都形成 Pareto improvement，也尚未跨数据集验证。

历史实验日志和当前状态不能继续混在同一段落中。

---

## 10. 当前阈值曲线不能证明责任事件阈值稳健

仓库记录在最终预测概率阈值 0.3–0.7 上，SDRR 平均曲线优于 baseline。这能支持：

> 增益不只是默认 0.5 推理阈值校准造成。

但它不能支持：

> 训练时的责任事件对阈值选择稳健。

因为训练责任始终使用：

\[
z>0,\qquad z-c_i\le0.
\]

需要另外审计：

\[
r_i^{(t)}
=
\mathbf 1[z>t]
\mathbf 1[z-c_i\le t].
\]

至少离线报告：

- \(t\) 对 event count 的影响；
- event set Jaccard；
- per-scale share；
- robust margin；
- 与最终 false-alarm component 的关联。

---

## 11. 最关键的 matched control 矩阵

主实验不应继续扩展新模块。下一轮只需要归因控制。

| 编号 | 方法 | 选择位置 | 被处罚量 | 回答的问题 |
|---|---|---|---|---|
| M0 | canonical MSHNet | 无 | 无 | 基线 |
| M1 | all safe-FP | 所有安全背景正像素 | `softplus(z)` | 普通安全背景 hard-negative 是否足够 |
| M2 | same responsible pixels | SDRR 的 unique pixels | `softplus(z)` | 是否只需找到这些困难像素 |
| M3 | magnitude-matched non-pivotal | 匹配 \(z,c_i\) 的正贡献，但删除不翻转 | `softplus(c_i)` | pivotality 是否比贡献幅值本身有效 |
| M4 | same-pixel random scale | 相同责任像素，打乱尺度身份 | `softplus(c_j)` | 正确尺度归责是否重要 |
| M5 | sparse random-event / gradient-matched | 相同活动 batch、事件数和梯度范数 | 同形正则 | 是否只是稀疏优化扰动 |
| M6 | SDRR | 删除翻转事件 | `softplus(c_i)` | 主方法 |

### 11.1 匹配必须同时保持

不能只匹配 event count。至少应匹配：

- event pair 数量 \(|E|\)；
- unique responsible pixel 数；
- responsibility degree \(d(x)\)；
- 每尺度 event 数；
- \(z\) margin 分布；
- \(z-c_i\) margin 分布；
- \(c_i\) 幅值分布；
- 距离最近 GT 的分布；
- active batch ratio；
- active image ratio；
- 正则 loss 数值；
- 对共享参数的梯度范数；
- schedule 与 ramp。

M3 是最关键的对照：

> 在完全相似的安全正像素、相似 final margin 和相似正贡献幅值下，只有“删除是否翻转”不同。

若 M3 与 SDRR 相当，则论文不能声称 pivotality 有独立价值，只能声称正尺度贡献 hard mining 有效。

---

## 12. 机制证据必须升级到 component 与 optimizer 层面

### 12.1 事件 anatomy

按 epoch、数据集和 checkpoint 报告：

- 每尺度 event share；
- \(d(x)\) 直方图；
- full margin \(z\)；
- deleted margin \(z-c_i\)；
- contribution \(c_i\)；
- 到 GT 的距离；
- FP component 大小；
- event 是否落在最终 FA component；
- event temporal persistence；
- 各尺度组合。

如果事件几乎全部来自 side0/fine scale，则论文应收窄为：

> fine-scale pivotal false-positive regularization

而不能声称发现了一般多尺度责任结构。

### 12.2 component-level linkage

当前训练事件是像素级，但 FA 指标是连接分量级。

应计算：

\[
\Pr(
\text{FP component contains responsible event}
).
\]

并比较：

- 有责任事件的 FP component；
- 无责任事件但 margin/大小匹配的 FP component；

在后续 checkpoint 中的：

- 消失率；
- 面积变化；
- 最大概率变化；
- 是否重新出现。

这能直接验证 SDRR 是否针对真正持久的 false-alarm components。

### 12.3 optimizer-aware one-step intervention

从完全相同的模型参数和 Adagrad state 出发，构造：

1. canonical step；
2. canonical + SDRR step；
3. canonical + matched control step。

在独立 probe batch 上评价：

\[
\Delta L_f^{\mathrm{probe}}
=
L_f(\theta')-L_f(\theta).
\]

至少在以下阶段审计：

- SDRR 启动前；
- 启动后第一轮；
- ramp 中期；
- ramp 完成；
- 训练后期。

仓库已有 optimizer counterfactual 基础设施，应把它用于 SDRR，而不是只保留为旧 TCDS 诊断工具。

### 12.4 轨迹而非单点 best epoch

同时报告：

- epoch 400 固定点；
- 介入后最后 20/50 epoch 均值；
- 介入后 validation AUC；
- best epoch；
- best-epoch gap；
- checkpoint bootstrap。

分别为 baseline 和 candidate 选择最佳 epoch，容易把路径噪声当成方法增益。

---

## 13. 数据与统计协议

### 13.1 模型种子不等于数据不确定性

目前主要使用一个固定 holdout split，再跑多个模型种子。这只能估计优化随机性，不能估计 split uncertainty。

至少增加：

- 5–10 个 paired model seeds；或
- 5 个 paired seeds + image/component bootstrap；
- 多个 split seeds；
- 两个以上数据集的冻结超参数验证。

### 13.2 正式跨数据集条件

正式主张至少要求：

1. NUAA-SIRST；
2. NUDT-SIRST；
3. IRSTD-1K；

均使用冻结：

\[
\lambda=0.05,\quad
K=15,\quad
\text{相对训练进度 schedule}.
\]

如果 NUDT canonical baseline 仍出现全背景塌缩，应先修复/复现 baseline，而不是让 SDRR承担恢复塌缩的任务。SDRR要求已有正决策，本身不能从零正预测恢复。

### 13.3 schedule 应按相对训练进度定义

当前：

\[
250/400=62.5\%
\]

是晚期介入。

论文实现最好写成：

\[
t_{\mathrm{start}}=\rho_sT,\qquad
t_{\mathrm{ramp}}=\rho_rT,
\]

例如：

\[
\rho_s=0.625,\qquad
\rho_r=0.125.
\]

这样跨训练长度才有明确语义。

仍需控制：

- 从 epoch 0 启动；
- 中期启动；
- 当前晚期启动；
- 同活动 batch 比例的随机稀疏正则。

如果只有晚期启动有效，论文应把方法解释为 late-stage decision refinement，而不是一般训练原则。

---

## 14. 相关工作与新颖性边界

近期相关工作已经覆盖：

- MSHNet：多尺度 side supervision 与 SLS loss；
- LoMix：多尺度 decoder logits 混合与可学习权重；
- PConv + scale-based dynamic loss：目标尺度驱动的动态损失；
- AC-SLSIoU：logit-domain target/hard-negative contrast、边界抑制、false-alarm focal loss；
- 对 MSHNet scale loss 非单调性和训练不稳定性的重新分析。

因此，以下表述不再足够新：

- “我们利用多尺度 logits”；
- “我们增加一个训练期 loss”；
- “我们抑制 false positives”；
- “我们在 logit 域处理 hard negatives”；
- “我们动态选择困难区域”；
- “我们不增加推理开销”。

真正可以建立差异的只有：

> **利用原生加性融合的精确单通道删除，定义阈值 pivotal scale contribution，并只抑制导致安全背景决策翻转的贡献。**

论文必须用 M3/M4 matched controls 证明这句话中的：

- deletion；
- flip；
- correct scale identity；

三者都具有独立价值。

---

## 15. 推荐的论文定位

### 15.1 方法名

当前 SDRR 可以保留，但正文应明确：

> responsibility means model-internal but-for pivotality under a fixed-logit final-fusion intervention.

更数学化的备选名是：

> Scale-Deletion Pivotal Refinement（SDPR）

“pivotal”比“responsibility”更不容易被理解为完整因果识别。

### 15.2 推荐标题

> **Which Scale Caused This False Alarm? Pivotal Scale Deletion for Infrared Small-Target Detection**

或：

> **Scale-Deletion Pivotal Refinement for Additive Multi-Scale Fusion**

### 15.3 推荐贡献写法

1. 发现加性多尺度融合中的一类错误：安全背景正决策可能由单个尺度贡献在阈值处起 but-for pivotal 作用。
2. 给出不增加参数的最终融合尺度分解和删除翻转定义。
3. 提出训练期 SDRR，只抑制阈值关键的错误尺度贡献，推理阶段完全移除。
4. 通过 contribution/margin/degree/gradient-budget matched controls，证明删除翻转语义优于普通 hard-negative mining。
5. 在多个数据集、多个种子和另一个加性融合 backbone 上验证。

### 15.4 不应使用的表述

- causal identification；
- true causal scale；
- exact network counterfactual；
- universally improves every seed；
- general deep-supervision principle；
- parameter-identical official MSHNet baseline，在 baseline 未物理隔离前；
- current evidence proves false-alarm suppression causes the IoU gain。

---

## 16. 代码仓库必须收敛

当前 README 仍以 “DEA-lite MSHNet” 为标题；`main.py` 同时暴露大量旧分支：

- RODS；
- TFDS；
- TGDS；
- coalition；
- filtration；
- continuation；
- DEA/full-DEA；
- SDRR/CRS。

这会让审稿人看到一个大规模 trial-and-error workbench，而不是清晰的方法实现。

### 16.1 paper branch 建议

```text
.
├── model/
│   ├── baselines/
│   │   └── mshnet_official.py
│   ├── evidence/
│   │   └── additive_scale_view.py
│   └── regularizers/
│       └── sdrr.py
├── experiments/
│   ├── controls/
│   └── legacy_archive/
├── tools/
│   ├── audit_sdrr_events.py
│   ├── audit_sdrr_components.py
│   └── audit_sdrr_influence.py
├── tests/
│   ├── test_mshnet_identity.py
│   ├── test_sdrr_deletion.py
│   ├── test_sdrr_gradient.py
│   └── test_sdrr_controls.py
├── configs/
│   ├── mshnet_baseline.yaml
│   └── mshnet_sdrr.yaml
├── CURRENT_STATUS.md
└── README.md
```

### 16.2 参数命名

将：

```text
--deep-supervision crs_flip_suppression
--crs-lambda
--crs-start-epoch
--crs-ramp-epochs
--crs-safe-kernel
```

改为：

```text
--fusion-regularizer sdrr
--sdrr-lambda
--sdrr-start-ratio
--sdrr-ramp-ratio
--sdrr-safe-kernel
```

旧名称只保留兼容 alias。

SDRR 不是一种 deep-supervision topology，把它放在 `--deep-supervision` 下会模糊论文概念。

### 16.3 状态文档

分开：

- `LAB_NOTES.md`：所有历史失败和筛选过程；
- `CURRENT_STATUS.md`：仅保留当前冻结方法、结果、未完成 gate；
- `README.md`：论文方法和复现命令。

---

## 17. 最小执行顺序

### Gate 0：baseline identity

- 移除 canonical MSHNet 中的 decidability head；
- 明确 official 与 deterministic variant；
- 通过 strict checkpoint/key/parameter/output tests；
- 复现 canonical baseline。

### Gate 1：数值和事件稳定性

- direct zero-channel deletion；
- \(\delta\)-margin sweep；
- responsibility degree；
- per-scale anatomy；
- component linkage；
- event persistence。

### Gate 2：matched attribution

只跑 M0–M6，不新增模块。

最关键结论：

\[
\text{SDRR} >
\text{magnitude-matched non-pivotal}
\]

且：

\[
\text{SDRR} >
\text{same-pixel random-scale}.
\]

### Gate 3：统计稳定性

- 更多 paired seeds；
- fixed epoch、last-K 和 AUC；
- split uncertainty；
- paired bootstrap/CI。

### Gate 4：跨数据集和跨 backbone

- NUDT-SIRST；
- IRSTD-1K；
- 至少一个具有原生加性多尺度融合的非 MSHNet 网络。

### Gate 5：冻结后读取 official test

在配置、阈值、schedule 和 controls 全部冻结后，才打开最终 test。

---

## 18. GO / NO-GO 标准

### GO

只有同时满足以下条件，SDRR 才具有顶会主方法潜力：

1. 在至少两个数据集上，以冻结配置取得稳定正 paired mean；
2. 增益不由一个种子主导；
3. 优于 safe-FP、same-pixel、magnitude-matched non-pivotal 和 random-scale controls；
4. optimizer-aware probe influence 优于 matched controls；
5. 事件与真实 FP component 的消失存在可重复联系；
6. canonical baseline 物理隔离；
7. 另一个加性融合 backbone 上成立；
8. 不需要针对每个数据集重新调 K、lambda 和 schedule。

### NO-GO / 必须降级表述

出现以下任一情况，应停止把 SDRR作为“责任机制”投稿：

1. magnitude-matched non-pivotal 与 SDRR相当；
2. random event / gradient-matched control 与 SDRR相当；
3. 增益只来自一个种子；
4. 跨数据集消失；
5. 事件几乎全部来自单一尺度；
6. direct zero-channel 与代数事件高度不一致；
7. event mean 有效但 density/pixel-normalized 失效，且增益可由稀疏梯度 kick 解释。

此时仍可将其降级为：

> late-stage sparse contribution regularization

但创新等级会明显下降。

---

## 19. 最终判断

SDRR 是当前仓库中最值得继续的研究线，且明显优于此前多启发式拼接的方法。它满足“不堆模块”的基本要求，并具备：

- 原生结构依赖；
- 单一事件定义；
- 训练期-only；
- 可解释梯度；
- 可证伪机制；
- 初步多种子正结果。

但当前顶会风险依次为：

1. **责任语义尚未与 matched hard-negative controls 区分；**
2. **按事件均值产生稀疏全强度梯度 kick；**
3. **最大增益种子的 FA 反而恶化，直接机制无法解释主要收益；**
4. **三种子统计不确定性过大；**
5. **baseline 尚未物理隔离，且 deterministic `amax` 改变潜在并列最大值 backward；**
6. **只在一个正式数据集、一个 backbone 上成立；**
7. **README、参数命名和历史分支尚未收敛。**

当前最正确的研究动作不是设计 SDRR v2，也不是引入 soft pivotality、pairwise coalition 或 learned router，而是完成：

\[
\boxed{
\text{baseline identity}
\rightarrow
\text{robust deletion audit}
\rightarrow
\text{matched attribution}
\rightarrow
\text{optimizer influence}
\rightarrow
\text{cross-dataset validation}
}
\]

只有当删除翻转事件在严格匹配贡献幅值、像素难度、尺度预算和梯度预算后仍显著优于对照，SDRR 才能从“有效的稀疏正则”升级为“具有独立方法学贡献的尺度责任学习原则”。

---

## 20. 2026-07-12 本地执行响应（Codex）

### 20.1 权威状态修正

- 本地当前 HEAD 实测为 `ff0ea51`，不是原复核开头记录的 `9e8d8e9`；后者仍存在于
  `model/MSHNet.py` 的历史中。
- 完整仓库测试已在本地实际执行，当前为 **`228 passed in 19.82s`**，不再沿用
  静态记录的 `187 passed`。
- NUDT-SIRST 的三条剩余 400-epoch 正式任务仍在运行；在它们全部写满 epoch
  `0..399` 前，fail-closed 汇总器拒绝生成最终跨数据集结论。

### 20.2 Gate 0：baseline identity 已完成的部分

已从仓库历史 revision `46cdfd4` 恢复物理隔离的 canonical MSHNet，并新增：

```text
model/baselines/mshnet_official.py
model/baselines/mshnet_deterministic.py
tests/test_mshnet_baseline_identity.py
```

实测结论：

1. official-forward 与 deterministic-backward 版本的 `state_dict` key 集、参数量完全
   相同，可 `strict=True` 互载；clean 模型参数量为 **4,065,513**。
2. 当前 workbench MSHNet 只比 clean 版本多四个 `decidability_head.*` tensor，共
   **521** 个参数；过滤这些 dormant keys 后，固定输入的四个 side logits 和最终
   输出逐元素完全一致。
3. `AdaptiveMaxPool2d(1)` 与 `torch.amax` 前向一致，但在并列最大值输入上 backward
   确实不同，已用显式单元测试证明。因此论文只能写“parameter-identical
   deterministic-backward MSHNet variant”，不能写 exact official backward。
4. CLI 已新增 `--mshnet-variant {workbench,official,deterministic}`，clean variant
   物理上不包含 DEA-lite head 或 `fusion_alpha`。为兼容历史 checkpoint，默认仍为
   `workbench`；paper 命令必须显式选择 `deterministic`。正式 checkpoint 已通过下述
   带 SHA 等价迁移物理去除 dormant head，Gate 0 不再依赖重新训练。

### 20.3 Gate 1：direct zero-channel 与事件稳定性实测

新增 `tools/audit_sdrr_deletion_stability.py`，在 NUAA seed 20260713 的正式 best-IoU
baseline checkpoint 上实际运行。审计使用过滤 dormant keys 后的 clean deterministic
MSHNet，并比较：

1. 训练实现的 `z-c_i`；
2. 将第 `i` 个 fusion input channel 真正置零后重新调用 final convolution。

结果：

| 项目 | baseline seed 20260713 | SDRR seed 20260713 |
|---|---:|---:|
| direct/algebraic max abs logit error | `3.1948e-5` | `3.1471e-5` |
| mean abs logit error | `2.6691e-6` | `2.4792e-6` |
| delta=0 responsibility events | 4 | 0 |
| direct/algebraic event mismatch | 0 | 0 |
| delta `0,1e-6,1e-4,1e-3,1e-2` mismatch | 全部 0 | 全部 0 |

baseline 的 4 个事件在所有上述 margin 下均保留，最小 robust margin 为 `0.1027`，
因此它们不是浮点阈值伪影。4 个事件全部为 degree 1，且全部来自 scale 0；这只是
一个 checkpoint 的证据，提示 fine-scale 偏置风险，不能外推为一般多尺度结构。

训练责任阈值仍表现出语义敏感性：相对 logit threshold 0 的 event-set Jaccard 在
约 `-0.405/+0.405` 时分别为 `0.50/0.667`。这与 direct deletion 的数值稳定性是
两个不同问题，论文必须分别报告。

### 20.4 Gate 2：归因控制已实现但尚无性能结论

新增三类 training-only 控制，均不增加参数或推理路径：

- `safe_density` 与 `unique_pixel` normalization：检验 event-mean 是否因稀疏事件
  获得近似固定总梯度 kick；默认 `event` 保持正式 SDRR 数值不变。
- M3 `crs_magnitude_nonpivotal`：在同一 image×scale 中匹配 full logit `z` 与贡献
  `c_i`，但强制 `z-c_i>0`，即删除不翻转。
- M4 `crs_same_pixel_random_scale`：保持责任像素与 degree，只更换被处罚的尺度身份。

此前实现的随机控制只匹配 image×scale event count，没有匹配 unique pixel、degree、
margin 或梯度范数，已明确降级命名为：

```text
SDRR-ScaleBudgetRandomControl-Unmatched
```

它不得再被称为完整 matched-random control，也不得用于证明 pivotality。M3/M4 当前
只有 mechanics/gradient/metadata 测试通过，尚未产生训练性能，因此原复核的 GO/NO-GO
判据保持不变。

### 20.5 当前执行判定

本轮修改已经消除了两类容易导致顶会拒稿的表述风险：

1. 不再把 polluted workbench MSHNet 称为 exact official baseline；
2. 不再把只匹配事件数的随机控制包装为 matched attribution evidence。

但当前仍是 **NO-GO pending**：需要完成 NUDT 三种子、clean deterministic baseline
等价迁移，并实际运行 M3/M4/normalization controls。只有性能和 optimizer/component 证据
通过，才能把 SDRR 从 sparse contribution regularization 提升为 pivotal scale
attribution 方法。

### 20.6 optimizer-aware one-step 初步结果：当前不支持强责任主张

新增 `tools/audit_sdrr_optimizer_influence.py`。工具从同一个 checkpoint 的模型参数和
Adagrad accumulator 深拷贝分支，使用同一个 train-mode batch 分别执行 canonical、
SDRR、M3、M4 一步更新，再在独立 validation probe batch 上计算 canonical 五项损失。
它包含 canonical-vs-canonical 自检；严格 CUDA 确定性下，该自检的参数更新差、probe
差和一阶差均精确为 0。

同时修复了一个重要不变量：无责任/无控制事件时，训练器现在直接返回 canonical loss
对象，而不是把数值为 0 的辅助计算图相加。完整模型测试确认此时所有参数梯度逐 tensor
bitwise 相同。

在 NUAA baseline best checkpoints 上，每个可用种子审计两个真正 train-mode active
batches，得到相对 canonical step 的 probe loss 变化（负值更好）：

M4 结果使用 control-only 的共享参数梯度 L2 匹配：在同一 forward 图上分别计算真实
SDRR 与 M4 正则对所有 MSHNet 参数的梯度，并以停止梯度比例缩放 M4。需要注意，
Adagrad accumulator 仍会使 raw gradient norm 相同但实际 update norm 不同。

| seed | active batch | SDRR probe Δ | M4 probe Δ | SDRR marginal update norm | M4 norm |
|---:|---:|---:|---:|---:|---:|
| 20260711 | 16 | -0.000489 | **-0.002048** | 2.3645 | 0.3575 |
| 20260711 | 31 | +0.020432 | **-0.000781** | 3.3119 | 0.2951 |
| 20260713 | 17 | +0.011166 | **+0.001056** | 2.5611 | 1.0722 |
| 20260713 | 39 | +0.157911 | **+0.100463** | 7.8045 | 6.3634 |

seed 20260712 best checkpoint 的全部固定训练 batches 均无责任事件，因此没有可定义的
one-step SDRR influence，工具按 fail-closed 退出。

上述样本量仍小，不能构成最终统计；但方向已经足够明确，不能被隐藏：当前 event-mean
SDRR 在 3/4 个 active batches 上使独立 probe loss 变差，且单事件可产生很大的 Adagrad
边际更新。在共享参数 raw-gradient L2 已匹配后，M4 在 4/4 个 batch 上的 probe 影响仍
优于 SDRR，并在 seed 11 两个 batch 上均改善 probe。当前证据支持“稀疏强梯度改变
轨迹”的风险，不支持“正确 pivotal scale identity 产生更优的一步任务影响”。除非
正式 M3/M4、normalization 和多 batch/stage 审计给出相反证据，论文只能保守定位为
late-stage sparse contribution refinement。

进一步加入 M1（all safe-FP fused penalty）和 M2（same pivotal pixels、fused-logit
penalty、去除尺度身份），并对 M2/M4 匹配 SDRR 的共享参数 raw-gradient L2。首个
active batch 结果如下：

| seed | SDRR | M1 all-safe FP | M2 pivotal-pixel fused | M4 random scale |
|---:|---:|---:|---:|---:|
| 20260711 | -0.000489 | **-0.002403** | -0.001266 | -0.002048 |
| 20260713 | +0.011166 | +0.036599 | +0.010660 | **+0.001056** |

这里的数值仍是相对 canonical one-step 的 probe loss 变化。M1 跨种子不稳定；M2 与
SDRR 非常接近且两次均略优；M4 两次均优于 SDRR。初步解释是：scale deletion 找到的
pivotal pixel 可能有信息，但“处罚正确责任尺度”尚未显示独立价值。该结论必须由 M1/M2/
M4 正式训练验证；若成立，研究主线应收窄为 deletion-pivotal region refinement，而
不是继续声称 scale responsibility learning。

### 20.7 component-level linkage：局部命中真实 FA，但覆盖有限

`tools/audit_counterfactual_responsibility.py` 已扩展为使用仓库正式 PD/FA 连接分量
匹配规则。NUAA seed 20260713 baseline best checkpoint 的 4 个责任事件全部位于图像
`Misc_36` 的同一个 unmatched false-alarm component：

- component area：9 pixels；
- centroid：`(71.78, 240.89)`；
- responsible pixels/events：4/4；
- responsible-pixel FP-component precision：100%；
- 但只覆盖全部 6 个 FP components 中的 1 个，即 16.7%。

在配对 SDRR best checkpoint 中，该图像的 FP component 数从 1 降到 0；原 component
的 9-pixel support 上没有任何正预测，最大 logit 从 `+15.4569` 降到 `-17.8351`。
整个验证集上，FP components 从 6 降到 2、matched components 从 52 增到 53，责任
事件从 4 降到 0。

这支持“责任事件确实可能落在真实 FA component，并随 SDRR 轨迹消失”的局部机制，
但不能证明直接因果：两个 best checkpoints 来自不同训练轨迹，且事件只覆盖少数 FP
components。必须用相同 checkpoint 的 one-step/component persistence 或更多事件重复
验证，不能把全部 component 改善归因于这 4 个像素。

### 20.8 NUDT 正式分支共享前缀证明

NUDT seed 20260712 的 baseline 与 SDRR 从同一个 strict epoch-79 checkpoint 和
Adagrad state 恢复。两条任务在 epoch `0..250` 的全部 251 行 IoU/PD/FA 逐项完全
一致，首次 mismatch 精确出现在 epoch 251；配置中 `start=250`，因此 epoch 250 的
ramp 仍为 0，epoch 251 首次非零。该结果排除了 resume、数据顺序或随机数差异导致的
提前分叉，正式配对差异只在 SDRR 训练目标实际介入后出现。

### 20.9 paper-facing CLI 已与方法概念对齐

旧 `--deep-supervision crs_*` 与 `--crs-*` 参数仍作为历史 checkpoint 兼容入口；新增
论文命令可以写为：

```bash
--fusion-regularizer sdrr \
--sdrr-lambda 0.05 \
--sdrr-start-ratio 0.625 \
--sdrr-ramp-ratio 0.125 \
--sdrr-safe-kernel 15
```

验证器会把相对 schedule 映射为 400-epoch 协议的 `start=250/ramp=50`，并同时持久化
绝对 epoch 与相对比例。M1–M4 也通过 `--fusion-regularizer` 的显式 control 名称进入，
不再在论文代码中伪装成 deep-supervision topology；冲突的新旧开关会 fail-closed。

### 20.10 NUAA 轨迹级结果：主要作用是晚期稳定化，而非 best-only 峰值

正式汇总器现强制报告 epoch 399、last-20、last-50 和介入后全段均值。NUAA 三种子
结果如下（均为 paired SDRR−baseline）：

| 统计方式 | mean IoU Δ | 正/负种子 | mean PD Δ | mean FA/M Δ |
|---|---:|---:|---:|---:|
| per-run best IoU | +0.0160 | 2/1 | +0.0124 | -1.0646 |
| epoch 399 | **+0.0557** | **3/0** | -0.0123 | -68.2505 |
| last-20 mean | **+0.0667** | **3/0** | -0.0392 | -124.5011 |
| last-50 mean | **+0.1015** | **3/0** | -0.0309 | -196.0057 |
| epochs 250–399 mean | **+0.0572** | **3/0** | -0.0191 | -118.5722 |

epoch 399 的逐种子结果：

| seed | baseline IoU/PD/FA | SDRR IoU/PD/FA | IoU Δ |
|---:|---:|---:|---:|
| 20260711 | 0.6400 / 0.9444 / 94.0367 | 0.7266 / 0.9815 / 17.0331 | +0.0866 |
| 20260712 | 0.5905 / 0.9444 / 185.9443 | 0.6689 / 0.9259 / 80.5522 | +0.0784 |
| 20260713 | 0.6671 / 0.9815 / 82.6813 | 0.6691 / 0.9259 / 60.3254 | +0.0020 |

该证据比 best-IoU 单点更稳定：所有种子在固定终点和轨迹均值上均提高 IoU，并显著
降低 late-stage FA。但 PD 在 seed 12/13 固定终点下降，不能称作逐种子 Pareto。
最合理的当前解释是：event-mean 稀疏介入改变并稳定了后期优化轨迹，阻止 canonical
MSHNet 的 IoU/FA 退化；这与 one-step audit 发现的大边际 update 一致，却弱化了“直接
处罚正确责任尺度导致提升”的主张。M2/M4/normalization 正式训练必须判断稳定化是否
依赖 pivotal pixel、正确尺度身份，还是任何匹配的晚期梯度脉冲都能实现。

### 20.11 epoch-399 paired image/component bootstrap

对每个种子的同一 43 张 validation images 做 10,000 次 paired bootstrap；每次同时
重采样 baseline/SDRR，并重新聚合 pixel IoU、component PD 和 FA area。该分析只估计
固定模型下的图像采样不确定性，不代替 model-seed/split uncertainty。

| seed | point IoU Δ | IoU 95% percentile CI | bootstrap ΔIoU>0 | point PD Δ | point FA/M Δ |
|---:|---:|---:|---:|---:|---:|
| 20260711 | +0.0866 | `[+0.0107,+0.1878]` | 99.85% | +0.0370 | -77.00 |
| 20260712 | +0.0783 | `[-0.0056,+0.2005]` | 93.94% | -0.0185 | -105.39 |
| 20260713 | +0.0020 | `[-0.1265,+0.1090]` | 51.90% | -0.0556 | -22.36 |

seed 11 的固定终点改善对图像重采样稳健；seed 12 方向大多为正但区间跨 0；seed 13
几乎完全不确定。seed 12/13 的 PD 下降也不能忽略。因此可以说“三种子固定终点 IoU
方向均为正、平均轨迹稳定化明显”，不能说“三个种子在数据层面都显著改善”。

### 20.12 Gate 0 完成：正式 checkpoint 的物理 clean 等价迁移

真实 epoch-79 checkpoint 中，workbench MSHNet 有 224 个 parameter tensors，clean
deterministic 版本有 220 个；额外 optimizer IDs 220–223 对应四个
`decidability_head.*` tensors，其 Adagrad step 和 accumulator 均严格为 0。前 220 个
参数的名称与顺序完全一致。

`tools/migrate_mshnet_run_to_clean.py` 只在以下条件全部满足时迁移：

1. clean 参数序列是 workbench 参数序列删除 `decidability_head.*` 后的精确子序列；
2. 被删 optimizer state 全零；
3. 其余 model/buffer keys 无 missing/unexpected；
4. 迁移后 clean model 与 Adagrad 均 strict load；
5. 每个父文件与迁移文件记录 SHA-256。

真实 seed 20260712 epoch-79 checkpoint 迁移后，从 workbench 与 clean 分支运行完全相同
的 epoch 80，得到：

- 公共 340 个 state tensors：0 mismatch，最大绝对差 0；
- 前 220 个 Adagrad states：0 mismatch；
- IoU/PD/FA：两边均为 `0.5497035/0.9444444/216.8167`；
- 唯一结构差异：workbench 仍有四个 dormant head keys，clean 无这些 keys。

NUAA 三个 baseline 与三个 SDRR 正式 400-epoch runs 均已生成独立
`clean_formal_*` artifacts，checkpoint epoch=399、340 state keys、方法/variant 元数据
正确，并保留原始结果目录。因此无需重跑来证明一个从未参与 forward/backward 的零状态
head；paper artifact 可以物理报告 4,065,513 个 MSHNet 参数和 0 个 SDRR 新参数。

### 20.13 随机种子协议：三种子报告，最终模型选择 seed 13

投稿稳健性协议沿用 `20260711 / 20260712 / 20260713`。baseline、SDRR 和所有 matched
controls 必须在每个 seed 内共享同一初始化、dataloader 顺序、optimizer state 和介入
前 checkpoint；三种子全部报告。同时，最终发布 checkpoint 允许按内部 validation IoU
选择 seed。

正式数据不再用划分 seed 生成第三个 validation：每个数据集只读取用户指定 `img_idx`
内的 `train_*.txt/test_*.txt`，完整 train 用于拟合、official test 用于固定 epoch 评测，
并保存两份 SHA-256。所有方法复用。现有内部 holdout runs 继续作为不删减的先导证据；
当前 SDRR best-IoU 在 SIRST-v1 为 `0.7324/0.7350/0.7359`，在 NUDT-SIRST 为
`0.7639/0.7140/0.7656`，两个数据集的最高值均来自 `20260713`。因此最终模型预选 seed
13，并只用该 seed 在 full-train/official-test 协议重训，避免观察多个 test 结果后再挑选。
`seed=42` 的 1-epoch run 只验证 213/214 loader 与 artifact，不进入性能结论。

正式入口 `scripts/official/run_sdrr_formal_nuaa_seed.sh` 使用
`--evaluation-protocol official_train_test`，最终命令使用 `SEED=20260713`；脚本仍保留
显式 `SEED/GPU_ID` 便于复现实验，且先训练唯一
canonical parent 至 epoch 250，
再通过带父 checkpoint SHA-256
记录的 `baseline`/`sdrr` 两类分支继续至 epoch 399；正式 loader 使用 `num_workers=0`。
脚本已通过 `bash -n`，baseline 分支的 method/lambda/logging 语义有单元测试。

### 20.14 第二骨干可迁移性与 SIRST-v1 数据完整性核验

UIUNet 的原生最终层是 `Conv2d(6,1,1)`，其输入正好是六个 full-resolution side
logits。因此无需新增 head，可精确写成 `z=b+Σ_i w_i s_i`。新增
`model/additive_fusion.py` 只在训练/审计期捕获已有 `outconv` 的输入和原生输出：不注册
参数、不改变 forward 输出，且 SDRR 核心已从固定四尺度推广为任意 `S>=1` 的原生可加
贡献。六尺度代数、hook 清除、参数不变和梯度等价均有单元测试。

在既有 UIUNet SIRST-v1 best checkpoint（epoch 804）上，严格使用用户指定的
`/home/md0/ly/DEA/datasets/img_idx` test manifest 逐图审计：214 张图中 3 张存在责任
事件，共 21 events / 20 unique pixels，六尺度分布为 `[16,3,2,0,0,0]`。原生卷积与
显式求和的 FP32 最大差为 `4.58e-5`，事件的最小 full-positive margin 为 `0.0495`、
最小 deleted-nonpositive margin 为 `0.0569`，仍比数值误差高约三个数量级；
决策事件在当前 checkpoint 上数值稳定。结果已保存至
`repro_runs/uiunet_nuaa_best_sdrr_eligibility.json`。这只证明第二骨干存在原生可归责事件，
不等于 SDRR 已在 UIUNet 上取得性能提升；仍需正式 paired training。

审计同时发现 SIRST-v1 `Misc_111` 的 image 为 `325×220`、mask 为 `592×400`。两文件
与数据集作者官方 `images.zip/masks.zip` 的 SHA-256 分别逐字节一致，因此不是本地损坏；
二者宽高比一致且 mask 非空。MSHNet 会将 image/mask 分别 resize 到统一输入尺寸，原路径
不会 shape crash；BasicIRSTD UIUNet 测试路径只 pad，原实现会在该样本失配。现已加入
显式规则：仅当宽高比误差不超过 1% 时，用 nearest-neighbour 将 mask 对齐至 image；
真正的宽高比不一致则 fail-closed，禁止静默裁剪或删除测试样本。
`tools/audit_dataset_pair_integrity.py` 已对固定 manifests 中的 2,755 个 unique IDs 完成
全量检查：SIRST-v1 427、NUDT-SIRST 1,327、IRSTD-1K 1,001，均无 missing pair、
manifest 内重复或非法宽高比；唯一特殊项就是同宽高比可重采样的 `Misc_111`。机器可读
记录为 `repro_runs/dataset_pair_integrity.json`。

### 20.15 UIUNet paired runner smoke：共享数据流与前缀逐位一致

新增 `tools/train_uiunet_sdrr_paired.py`：同一 GPU 内同时维护 baseline/candidate 两份
UIUNet，但每个 step 只从 dataloader 读取一次增强后的 batch，再依次更新两模型；SDRR
直接读取 candidate 原生 `outconv`，不改模型 forward。runner 保存两套 model、Adam、
scheduler、Python/NumPy/Torch/CUDA RNG 以及 dataloader generator state，支持 epoch 边界
严格恢复。

两 epoch seed-42 mechanics smoke（旧 BasicIRSTD 255/85 split，仅用于代码通路）中，
epoch 0 的两模型 942 个 state tensors 逐位一致，最大绝对
差为 0，IoU/PD/FA 也完全相同，证明共享前缀路径成立；人为将 SDRR 提前至 epoch 1 后，
责任计算、反向传播、评估与 1.2GB 完整 checkpoint 均成功执行。该提前介入产生 3,116,107
events，远离冻结的 `start=250/ramp=50` 协议，因此 smoke 的性能数值不可作为实验结果。
一次 `batch=32` smoke 明确 OOM；`batch=4` 稳定占用约 6.6GB（同时还有 MSHNet 进程），
正式 UIUNet 使用原协议 batch=8，并在独占/空闲 GPU 上运行。

### 20.16 NUDT-SIRST 三种子完成：best-IoU 全正，但存在 PD trade-off

400-epoch paired runs 已全部完成；baseline/SDRR 在每个 seed 内共享介入前轨迹。按每条
run 的内部验证 best-IoU 选择：

数值日志复核显示三个 seeds 的 epoch `0..250` 全部逐项相同，3/3 的首次 mismatch 都
精确出现在 epoch 251，与 ramp 的首次非零位置一致。

| seed | baseline IoU / PD / FA | SDRR IoU / PD / FA | paired IoU Δ |
|---:|---:|---:|---:|
| 20260711 | 0.7575 / 0.9572 / 13.6526 | 0.7639 / 0.9465 / 4.0155 | +0.0064 |
| 20260712 | 0.7116 / 0.9519 / 34.7625 | 0.7140 / 0.9465 / 27.0757 | +0.0024 |
| 20260713 | 0.7496 / 0.9626 / 25.6990 | 0.7656 / 0.9519 / 19.8479 | +0.0160 |
| **Mean** | **0.7396 / 0.9572 / 24.7047** | **0.7478 / 0.9483 / 16.9797** | **+0.0083** |

即 best-IoU 在 3/3 seeds 为正，FA 在 3/3 seeds 下降；但 PD 也在 3/3 seeds 下降，
平均 `-0.0089`，不能写成 Pareto improvement。轨迹统计进一步显示：

| 统计 | mean IoU Δ | 正/负 seeds | mean PD Δ | mean FA/M Δ |
|---|---:|---:|---:|---:|
| epoch 399 | +0.0128 | 2/1 | -0.0036 | -9.4842 |
| last-20 mean | +0.0110 | 2/1 | -0.0046 | -16.7656 |
| last-50 mean | +0.0047 | 2/1 | -0.0080 | -15.9701 |
| epochs 250–399 mean | +0.0021 | 2/1 | -0.0077 | -16.2511 |

seed 12 的 fixed-end IoU 为 `-0.0119`，必须保留；另一方面，三个 SDRR best epochs
为 389/398/399，而 baseline 为 341/382/346，继续支持“晚期 IoU/FA 稳定化”解释。
结果保存在 `repro_runs/formal_sdrr_nudt_paired_summary.json`。六条 run 已全部迁移为
`clean_formal_*_nudt_*`，clean summary 与原 summary 逐字节一致。跨 SIRST-v1 与 NUDT
的正结果已建立，但 mechanism claim 仍取决于正在运行的 M2/M4/normalization controls。

### 20.17 NUDT epoch-399 paired bootstrap

对每个 seed 的同一 133 张 validation images 做 10,000 次 paired bootstrap，使用 clean
checkpoint 重新推理并聚合 pixel IoU/component PD/FA：

| seed | point IoU Δ | IoU 95% CI | bootstrap ΔIoU>0 | point PD Δ | point FA/M Δ |
|---:|---:|---:|---:|---:|---:|
| 20260711 | +0.0159 | `[-0.0174,+0.0480]` | 82.71% | -0.0267 | -27.19 |
| 20260712 | -0.0118 | `[-0.0496,+0.0182]` | 25.17% | +0.0160 | +4.70 |
| 20260713 | +0.0346 | `[+0.0082,+0.0650]` | 99.64% | 0.0000 | -6.31 |

只有 seed 13 的 fixed-end IoU 95% CI 完全高于 0；seed 11 方向大多为正但不显著，seed
12 更偏向负。seed 11 的 FA CI 完全低于 0，但同时 PD 明显下降。这再次说明三种子
best-IoU 正结果不能扩张为“所有固定模型均显著改善”，也不能隐藏 IoU/PD/FA 之间的
trade-off。原始报告为 `clean_formal_sdrr_nudt_seed*_epoch399_bootstrap.json`。
