# 基于 MSHNet 的 TCDS-Projection / TFDS 第二轮阶段复核

> **复核日期：2026-07-11**  
> **当前状态：工程路径成立，科学假设尚未 GO。**  
> 当前仓库已从启发式 RODS 推进到一个可归因得多的 **TCDS-Projection** 原型：网络和 canonical 五项损失聚合保持不变，只根据给定标签投影的任务恢复结果改变辅助标签的确定域。但它仍然没有证明“投影不一致会产生有害辅助更新”，也没有估计完整辅助输出空间的最优可表达误差 \(\epsilon^*\)，因此不能称为完整 TFDS。

> **本轮执行边界**：新增 `tfds_projection_active_renorm` 仅作为梯度预算诊断，
> 不作为第二个主方法；optimizer counterfactual 必须采用“完整训练目标含边/不含边”
> 双分支，并复制同一 Adagrad 状态。当前只建立了 Phase-A 的正确计算基础，尚未
> 产出数据集级 harmful-influence 统计，因此科学状态仍为 NO-GO pending。

> **本轮验证**：完整仓库测试为 `141 passed`。在 `min_iou=0.5` 下重跑三个
> official-train 静态审计，stride `1→2`、`2→4`、`4→8` 的 graph nestedness
> violation 均为 0。该结果说明当前图在这三个数据集上完全满足“粗尺度可行则更细
> 尺度也可行”的单调关系，进一步强化了 occupancy/area gate 解释；它不是方法
> 有效证据。

> NUAA-SIRST 上还完成了同 seed、1 epoch、`warm_epoch=-1` 的 natural-drop 与
> positive-active-renorm 两条 smoke。两者均正常完成且日志语义符合预期；当
> `positive_active_ratio=0` 时 active-renorm side term 为 0。两次验证集 IoU/PD
> 均为 0，符合单 epoch mechanics smoke 的无性能解释边界，禁止用于方法比较。

## 2026-07-12 续验：从 TCDS/TFDS 转向原生融合归责

### 当前结论

TCDS/TFDS 及后续若干辅助监督变体没有通过真实三随机种子门槛，不能继续包装为
主方法。当前唯一值得继续验证的方向是 **训练期尺度删除归责**：它不增加网络模块、
不改变 MSHNet 推理图，而是利用 MSHNet 最终线性卷积可精确分解这一结构事实，询问
“删除某一尺度的原生贡献后，当前背景正决策是否翻转”。只有造成翻转的尺度贡献才
进入额外约束。

该方向目前仍是候选，不是最终成功模型。NUAA-SIRST 内部验证出现了三种子平均
提升，但最佳 IoU 工作点的 FA 尚未形成 Pareto 改善；NUDT-SIRST 独立验证正在运行。

### 严格确定性修复

原 `--deterministic true` 并不满足严格确定性：`ChannelAttention` 中
`AdaptiveMaxPool2d(1)` 的 CUDA backward 在当前 PyTorch 环境没有确定性实现。
本轮把该操作替换为前向完全等价的：

```python
torch.amax(x, dim=(-2, -1), keepdim=True)
```

并启用 `torch.use_deterministic_algorithms(True)` 与固定
`CUBLAS_WORKSPACE_CONFIG`。两个独立的一轮训练在同 GPU、同 seed、
`num_workers=0` 下得到逐张量 bitwise 相同的 checkpoint，最大参数差为 0。
因此下表的 strict baseline 才是后续正式对照；此前非严格结果只保留为探索记录。

### NUAA-SIRST 严格三种子基线

固定配置：官方 train manifest 内部 80/20 holdout、`split_seed=20260711`、
80 epochs、batch size 4、Adagrad、`lr=0.05`、canonical MSHNet 五项损失。

| seed | best epoch | IoU | PD | FA/M |
|---:|---:|---:|---:|---:|
| 20260711 | 73 | 0.6719 | 0.9630 | 25.5496 |
| 20260712 | 71 | 0.6773 | 0.9074 | 1.7743 |
| 20260713 | 61 | 0.6754 | 0.9630 | 1.7743 |
| mean | — | **0.6749** | **0.9444** | **9.6994** |

IoU 样本标准差为 0.0027。

### 训练期尺度删除归责定义

记最终 MSHNet logit 为：

\[
z=b+\sum_{i=0}^{3}c_i,
\]

其中 \(c_i\) 是使用最终融合卷积真实权重得到的第 \(i\) 个尺度贡献，而非另建
代理头。对安全背景像素，定义离散责任事件：

\[
r_i=
\mathbf 1[z>0]\,
\mathbf 1[z-c_i\le 0]\,
\mathbf 1[\text{safe background}].
\]

责任掩码停止梯度，只对造成决策翻转的 \(c_i\) 施加单调 softplus 抑制。没有责任
事件时该项严格为零；推理阶段完全删除该训练项，因此参数量、FLOPs 和预测路径与
MSHNet 相同。

### 已拒绝设置与当前候选

1. `start=20, ramp=20, lambda=0.05, kernel=15`：三种子 IoU 变化为
   `+0.0023/-0.0145/+0.0104`，平均 `-0.0006`，拒绝。离线审计显示责任事件过于
   稀疏，但按事件数归一化会让单个早期瞬时事件仍获得完整权重。
2. `start=50, ramp=10, lambda=0.05, kernel=7`：三种子 IoU 变化为
   `+0.0130/-0.0038/-0.0121`，平均 `-0.0010`，拒绝。虽然 seed 11 的 FA/M
   从 25.55 降到 8.87，但更小保护区损害了另外两个种子。
3. `start=50, ramp=10, lambda=0.05, kernel=15`：当前保留候选。前三种子
   介入前 51 个 epoch 与各自 baseline 指标逐项完全一致；最终结果如下。

| seed | baseline IoU | candidate IoU | paired delta | candidate PD | candidate FA/M |
|---:|---:|---:|---:|---:|---:|
| 20260711 | 0.6719 | 0.6866 | +0.0147 | 0.9630 | 47.5506 |
| 20260712 | 0.6773 | 0.6769 | -0.0004 | 0.9259 | 14.9039 |
| 20260713 | 0.6754 | 0.6884 | +0.0130 | 0.9259 | 11.7102 |
| mean | **0.6749** | **0.6840** | **+0.0091** | **0.9383** | **24.7216** |

候选 IoU 样本标准差为 0.0062。在线抽样日志中，有责任事件的 batch 比例分别为
12.4%、3.4%、10.3%；几乎没有责任事件的 seed 12 基本不变，而事件更丰富的
seed 11/13 获益，这与提出的机制方向一致，但目前只是相关性证据。

### NUDT-SIRST 80-epoch 三种子筛选

| seed | baseline IoU / PD / FA | candidate IoU / PD / FA | paired IoU delta |
|---:|---:|---:|---:|
| 20260711 | 0.6697 / 0.9305 / 57.5933 | 0.6837 / 0.9198 / 19.5037 | +0.0140 |
| 20260712 | 0.0009 / 0 / 22712.6502 | 0.0009 / 0 / 22712.6502 | 0.0000 |
| 20260713 | 0.6644 / 0.9305 / 61.9530 | 0.6654 / 0.9305 / 55.0693 | +0.0010 |
| mean | 0.4450 / 0.6203 / 7610.7322 | 0.4500 / 0.6168 / 7595.7411 | **+0.0050** |

三个种子均无负 IoU 配对变化。seed 20260711 的 IoU 和 FA 改善、PD 下降 0.0107；
seed 20260713 在 IoU、PD、FA 三项上形成严格 Pareto 改善；seed 20260712 的
baseline/candidate 都全背景塌缩。责任机制要求已有正决策，因此不会恶化该塌缩，也
无法从零正预测中恢复。该失败必须计入均值和稳定性分析，不能删种子。

NUAA 平均 `+0.0091`、NUDT 平均 `+0.0050` 使候选通过 80-epoch 跨数据集筛选，
但 NUDT 的巨大方差意味着它仍不是正式论文结果。

另一个“只让责任项更新最终融合卷积”的成熟-checkpoint 微调对照已在 NUAA seed
20260711 上完成。40-epoch 期间最高 IoU 仅 0.6573，低于来源 400-epoch checkpoint
的 0.6742，因此该分支直接淘汰；限制梯度路径并没有自动解决性能/FA 权衡。

需要强调：以上 80-epoch 实验只是机制筛选。仓库标准训练协议为 400 epochs，且
已有旧环境 clean baseline 的 NUDT 三种子 IoU 为 0.7532/0.7483/0.7309。由于旧
`deterministic=true` 未真正启用严格确定性，这些 checkpoint 可用于诊断和初始化，
不能与新严格候选直接组成正式配对结果。只有筛选 gate 通过后，才值得承担严格
400-epoch 重跑成本。

该 gate 已通过。400-epoch 正式 NUAA 实验从各自严格 80-epoch checkpoint 分支，
baseline 与 candidate 共享前 80 轮状态和 optimizer；候选调度按训练长度等比例缩放
为 `start=250, ramp=50`。分支 checkpoint 显式改写方法语义并记录父 checkpoint
SHA-256，元数据审计不允许把 legacy checkpoint 静默当作 candidate 恢复。

正式 seed 20260711 已完成：strict baseline 最佳为 epoch 312、IoU 0.7369、
PD 0.9630、FA/M 9.2262；candidate 最佳为 epoch 355、IoU 0.7324、PD 0.9630、
FA/M 6.0325。候选改善 FA、保持 PD，但 IoU 下降 0.0045，因此单种子不通过；
seed 20260712/13 仍按原设置补齐，不根据该结果中途调参。

三个正式种子现已全部完成：

| seed | baseline IoU / PD / FA | SDRR IoU / PD / FA | paired IoU delta |
|---:|---:|---:|---:|
| 20260711 | 0.7369 / 0.9630 / 9.2262 | 0.7324 / 0.9630 / 6.0325 | -0.0045 |
| 20260712 | 0.6934 / 0.9444 / 1.7743 | 0.7350 / 0.9630 / 6.7423 | +0.0416 |
| 20260713 | 0.7250 / 0.9630 / 10.2908 | 0.7359 / 0.9815 / 5.3228 | +0.0109 |
| mean | **0.7184 / 0.9568 / 7.0971** | **0.7344 / 0.9692 / 6.0325** | **+0.0160** |

baseline/candidate 的 IoU 样本标准差分别为 0.0225/0.0018。SDRR 不仅提高平均
IoU 1.60 percentage points，也提高平均 PD、降低平均 FA，并显著降低种子方差。
这构成当前首个满足正式 NUAA 三种子成功门槛的模型。seed 20260711 的小幅负增益
必须保留报告；不能写成“每个种子都改善”。

论文名冻结为 **Scale-Deletion Responsibility Refinement (SDRR)**。其核心贡献
仍是原生融合贡献上的精确删除干预与决策翻转归责，不是新网络模块。正式超参数冻结
为 `lambda=0.05, start=250, ramp=50, safe_kernel=15`；NUDT 验证不再调参。

冻结 checkpoint 审计显示，seed 20260713 baseline 仍有 9 个远目标安全背景正像素，
其中 4 个由同一个尺度的删除直接翻转；SDRR checkpoint 将两者都降为 0。seed
20260711/12 的 baseline/candidate 最佳 checkpoint 本身均无此类像素。训练在线抽样
中，三个 SDRR 种子有责任事件的 batch 比例仅为 5.4%/4.4%/4.0%，说明约束是稀疏
事件驱动，而非持续重加权所有像素。不过 seed 20260712 的强增益不能仅由最终责任
事件消除解释，仍需 matched sparse/random-event control 排除优化扰动效应。

该对照现已实现为 `crs_matched_random`，但尚未产生性能结果。它在每个
image×scale 上先计算真实删除翻转事件数，再从同一安全背景正决策域内的非责任正
贡献中，以不消耗训练 RNG 的无状态伪随机排序抽取相同数量；惩罚函数、归一化、
权重和调度与 SDRR 完全相同。没有真实事件时，对照损失及梯度严格为零。该设计匹配
“何时激活、每尺度多少事件、梯度形式和预算”，只替换事件身份，因此可用于检验收益
是否来自删除翻转归责本身。当前只能说对照 mechanics 通过测试，不能提前声称 SDRR
已胜过该对照。

概率阈值 0.3–0.7 的均值曲线中，SDRR 的 IoU 均高于 baseline；在默认 0.5
阈值处为 0.7344 vs 0.7186，PD 为 0.9691 vs 0.9568，FA/M 为 6.03 vs 7.10。
因此正式增益不是只由默认阈值校准造成。

### 不能越界的论文表述

- 可以说：利用原生线性尺度融合的精确删除干预构造训练期归责，不增加推理模块。
- 不能说：已经证明因果识别；删除尺度仍是模型内部干预，不是数据生成机制干预。
- 不能说：已经获得 Pareto 改善；当前最佳 IoU 点的平均 FA/M 由 9.70 增至 24.72。
- 不能把混合尺度辅助监督本身作为新颖性。NeurIPS 2025 LoMix 已系统组合多尺度
  decoder logits 并监督其混合输出；本方法的差异必须落在“精确删除导致的决策翻转
  归责”，而不是“又一种 multi-scale loss”。
- `CRS` 缩写与 2026 年 Counterfactual Segmentation Reasoning 重名风险较高，
  投稿命名需改为不冲突的 Scale-Deletion Responsibility 类名称。

### 下一道 GO gate

1. 完成 NUDT-SIRST 的严格 400-epoch 三种子配对，不根据中途结果改参。
2. 正式配置冻结为 `start=250/ramp=50/lambda=0.05/kernel=15`；早期
   `start=50/ramp=10` 只属于 80-epoch 筛选协议。
3. 至少在两个数据集达到正的三种子配对 IoU 均值，并完整报告 PD/FA 与负种子。
4. 在 NUAA 正式设置运行 matched sparse/random-event control；若随机对照复现同等
   增益，则必须削弱或否定“删除翻转身份关键”的机制主张。
5. 若跨数据集只提升 IoU 而持续恶化 FA，则该候选只能作为分析分支，不能宣布成功。
6. 最终模型和机制表述通过上述 gate 后才允许读取 official test manifest。

本轮完整测试：`192 passed`。

---

## 一、总体判断

这轮修改方向正确，而且解决了上一版最严重的四个方法学问题：

| 问题 | 当前状态 | 判断 |
|---|---|---|
| 多个手工质量项相乘 | 已由 scene-level 投影恢复替代 | **实质性改进** |
| hard owner / soft responsibility | 已改为 binary feasibility graph | **实质性改进** |
| merge 只检查同一 pooling cell | 已按投影后连接分量检查，并拒绝 merge/split 全体参与者 | **实质性改进** |
| final/side 相对损失比例被改变 | `tfds_projection` 保持 canonical 五项平均 | **归因问题已部分解决** |
| unknown 与 positive 冲突 | unknown 优先 | **语义正确** |
| all-valid 时偏离 canonical SLS | 已设置 bitwise canonical 分支 | **工程基线更可靠** |
| RODS audit 阈值统计错误 | 已修正 | **工程问题已解决** |
| `rods_random` 被误称 matched control | 已明确为 unmatched | **表述边界已修正** |
| Phase-A 梯度 influence | 尚未完成 | **核心科学阻塞项** |
| matched-budget / scale-drop control | 尚未完成 | **核心归因阻塞项** |
| canonical MSHNet 物理隔离 | 尚未完成 | **投稿复现阻塞项** |
| \(\epsilon^*\) 或 projection gap / representation gap 分解 | 尚未实现 | **完整 TFDS 阻塞项** |

当前最准确的研究定位是：

> **TCDS-Projection 是一个用于检验“给定辅助标签投影是否在任务空间中一致”的最小训练原型，而不是方法有效性的证明。**

本轮完整仓库 `141 passed` 和 mechanics smoke 证明代码路径可运行、关键不变量得到覆盖，但不构成性能证据或机制证据。

---

# 二、当前实现中已经成立的部分

## 1. Scene-level 投影恢复比原 RODS 更接近统一原理

`model/task_consistent_supervision.py` 当前执行：

1. 对完整场景二值标签进行 max-pool 投影；
2. 将投影结果回升到原空间；
3. 在回升结果上重新提取连接分量；
4. 检查原实例与投影实例之间是否保持一对一关系；
5. 对 merge、split、消失、低 IoU 和定位失败进行统一记录；
6. 构造二值实例—head 可行图。

这比此前的

\[
q_{\mathrm{area}}\times q_{\mathrm{quant}}\times q_{\mathrm{merge}}\times q_{\mathrm{preference}}
\]

更统一，因为消失、合并、分裂、定位和区域退化都来自同一个投影后场景，而不是多个独立模块的乘积。

尤其重要的是：merge/split 的所有参与者都被拒绝，assignment 不再依赖实例遍历顺序。unknown 优先于 positive 也避免了一个 coarse cell 同时承载可行与不可行身份时被错误恢复为确定正标签。

## 2. 当前命名边界是正确的

当前代码只评估固定投影：

\[
P_s=\text{max-pool},\qquad U_s=\text{nearest lift}.
\]

因此它估计的是：

\[
\epsilon^P_{k,s}
=
D_T\!\left(G_k,U_sP_s(Y);G_{\neg k}\right),
\]

而不是：

\[
\epsilon^*_{k,s}
=
\inf_{z\in\mathcal Z_s}
D_T\!\left(G_k,U_sz;G_{\neg k}\right).
\]

所以运行名采用 **TCDS-Projection** 是合理的。正式论文在没有估计 \(\epsilon^*\) 前，不应写成：

- auxiliary output space is intrinsically infeasible；
- complete TFDS；
- representation gap has been solved。

当前只能写：

> the configured auxiliary label projection is task-inconsistent for some instance–head pairs.

## 3. canonical 五项平均恢复了显式损失比例

当前 `tfds_projection` 保持：

\[
L=
\frac{L_f+L_1+L_2+L_3+L_4}{5}.
\]

这消除了旧 RODS 中

\[
L_f+0.8\operatorname{mean}(L_s)
\]

造成的 final/side 显式权重改变。主方法不再通过 `aux_loss_weight` 人为提高 final head 的相对权重。

但这只解决了**公式系数层面**的比例问题，尚未解决实际有效梯度预算问题，后文会说明。

---

# 三、当前最重要的新风险：阈值实际上形成隐式尺度裁剪

## 1. 三个训练集的投影审计

在 `min_iou=0.5`、定位容差 3 pixels 时：

| Dataset | stride 1 | stride 2 | stride 4 | stride 8 |
|---|---:|---:|---:|---:|
| NUAA-SIRST | 100.00% | 67.04% | 10.74% | 0.37% |
| NUDT-SIRST | 100.00% | 92.27% | 15.58% | 0.22% |
| IRSTD-1K | 100.00% | 93.49% | 26.54% | 1.09% |

NUAA-SIRST 上：

| `min_iou` | stride-8 可行率 |
|---:|---:|
| 0.50 | 0.37% |
| 0.25 | 7.78% |
| 0.10 | 32.96% |

这说明当前图密度高度依赖单一阈值。它不是一个小型超参数敏感性问题，而是会改变方法的实际训练拓扑。

## 2. 为什么 `IoU >= 0.5` 天然压空粗尺度

在图像尺寸可被 stride 整除、单个实例完全位于一个 coarse cell、且没有 merge/split 的理想情形下，max-pool 后再 nearest lift 会把该实例恢复成一个 \(s\times s\) block。

若原实例面积为 \(A\)，则近似有：

\[
\operatorname{IoU}_{k,s}
=
\frac{A}{s^2}.
\]

因此：

\[
\operatorname{IoU}_{k,s}\ge \tau
\quad\Longrightarrow\quad
A\ge \tau s^2.
\]

当 \(\tau=0.5\) 时：

| stride | 单 cell 情形下所需最小面积 |
|---:|---:|
| 2 | 2 pixels |
| 4 | 8 pixels |
| 8 | 32 pixels |

如果目标跨越 \(n\) 个 coarse cells，则回升区域约为 \(ns^2\)，判据进一步近似为：

\[
A\ge \tau ns^2.
\]

所以当前的实例 IoU 阈值在很大程度上等价于一个 **cell occupancy / area gate**。stride-8 几乎为空并非偶然，而是该几何定义在 IRSTD 小目标分布上的直接结果。

## 3. 由此产生的论文风险

在 `min_iou=0.5` 下，当前方法近似执行：

- stride-1：canonical supervision；
- stride-2：部分保留；
- stride-4：大部分删除；
- stride-8：几乎删除。

因此任何性能提升都可能被解释为：

> coarse-side supervision 太强，删除 stride-4/8 后训练更好。

而不是：

> task-consistent graph 识别了语义上有害的实例—head 边。

正式实验前必须首先排除 **scale pruning explanation**。

---

# 四、Partial-SLS 中仍存在一个隐式梯度权重混杂

## 1. all-valid exact degeneration 是正确的

当整个输入张量满足：

\[
V\equiv 1,
\]

`PartialSLSIoULoss` 直接调用 canonical `SLSIoULoss`，能够保证全有效情形下的 bitwise identity。这是必要且正确的 baseline safeguard。

unknown 区域通过 `pred * valid`、`target * valid` 和带 valid 的 intersection 被移除，unknown 像素梯度为零，这一实现方向也正确。

## 2. “无已知正实例时零梯度”会产生动态 auxiliary weight

当前 partial 分支中，当某个 sample–head pair 没有已知正实例时：

- IoU 项被置为常数；
- location 项关闭；
- 该 sample–head pair 对参数梯度为零。

设一个 batch 中 head \(s\) 的 active sample 比例为：

\[
q_s=
\frac{1}{B}\sum_{b=1}^{B}a_{b,s},
\qquad
 a_{b,s}\in\{0,1\}.
\]

当前 batch mean 下：

\[
\nabla L_s^{\mathrm{partial}}
=
\frac{1}{B}
\sum_{b:a_{b,s}=1}
\nabla \ell_{b,s}
=
q_s\cdot
\frac{1}{Bq_s}
\sum_{b:a_{b,s}=1}
\nabla \ell_{b,s}.
\]

也就是说，即使总损失仍为 canonical 五项平均，head \(s\) 的实际有效权重仍会被 active ratio \(q_s\) 自动缩小。

在 stride-8 可行率接近零时，side-3 的梯度预算也会接近零。于是当前 TCDS-Projection 同时改变了：

1. 哪些实例区域被标为 unknown；
2. 哪些 sample–head pair 完全无梯度；
3. 每个 head 的实际梯度范数；
4. coarse head 相对于 final/fine heads 的有效优化权重。

因此，“保持 canonical 五项平均”解决了显式 loss coefficient 混杂，但尚未解决**隐式梯度预算混杂**。

## 3. 正式实验必须报告的量

每个 epoch、每个 head 至少记录：

- feasible edge ratio；
- active sample–head ratio；
- positive coarse-pixel count；
- valid pixel ratio；
- unknown pixel ratio；
- side loss 的参数梯度范数；
- side loss 对共享层的梯度范数；
- final/side 梯度范数比。

只报告实例可行率不足以判断实际训练预算。

## 4. 一个必要的诊断变体

建议只作为诊断，而不是新增主方法，加入 active-renormalized 版本：

\[
L_s^{\mathrm{active\text{-}renorm}}
=
\frac{
\sum_b a_{b,s}\ell_{b,s}
}{
\max(1,\sum_b a_{b,s})
}.
\]

比较：

- 当前 natural-drop；
- active-renorm；
- matched random natural-drop。

若只有 natural-drop 有收益，说明收益更可能来自降低 coarse auxiliary weight，而不是可行图语义。

## 5. mixed-batch identity 已补充

当前 bitwise canonical 分支通常以整个 tensor 是否 all-valid 为条件。需要增加混合 batch 测试：

- 样本 A 全有效；
- 样本 B 含 unknown；
- 验证样本 A 的 loss/gradient 是否与 canonical 路径一致或在预设数值容差内一致；
- 验证样本 B 的 unknown 区域梯度严格为零；
- 验证不同样本之间不存在 reduction 引起的非预期梯度泄漏。

当前代码已增加 `reduction="none"` 和 mixed-batch identity 测试，使 natural-drop
与 active-renorm 可以基于同一组 per-sample loss terms 比较。全有效但 target
为空的 crop 会保留 canonical SLS 的原始行为，因此日志必须区分：

- `positive_active_ratio`：存在已知正实例；
- `gradient_active_ratio`：该 sample–head 对实际可能产生梯度。

`tfds_projection_active_renorm` 的分母严格使用 `positive_active_ratio`；空 crop 的
canonical LLoss 可能出现在 `gradient_active_ratio` 中，但不计入正监督预算。该
变体仅诊断隐式权重，不参与主方法命名或最终性能主张。

---

# 五、投影算子还需要与真实训练几何严格对齐

当前 graph 使用固定 stride 和 `max_pool2d + nearest lift`。正式实验前需要证明它与 MSHNet 的真实辅助监督路径完全对应。

## 1. 不应只依赖固定 stride 假设

建议由实际 side logit shape 定义投影：

\[
P_s:\mathbb R^{H\times W}
\rightarrow
\mathbb R^{H_s\times W_s},
\]

而不是仅依赖 `stride=(1,2,4,8)`。

对当前 MSHNet，先 fail-closed 地验证输入满足网络下采样整除约束，再测试：

- 边界目标；
- resize/crop 后尺寸；
- repeated 2× pooling 与一次 stride-4/8 pooling 是否完全一致；
- target shape 与 side logit shape 是否始终一致。

奇数尺寸或不能被 stride 整除的尺寸只有在目标 backbone 本身支持时才纳入；不能
把 MSHNet 原生不支持的输入几何误记为 TCDS projector failure。跨 backbone 版本
应最终由实际 side-logit shape 定义 projector，而不是硬编码 stride。

否则，边界目标的 disappearance 可能来自 pooling floor/truncation，而不是辅助输出空间的真实分辨率限制。

## 2. 区分“标签投影恢复”与“网络输出空间恢复”

如果论文主张是 fixed projector consistency，nearest lift 可以作为标签投影的回升定义。

如果论文主张升级为 output-space feasibility，则 \(U_s\) 必须反映该 head 输出在任务空间中的真实解释方式，例如：

- side prediction 的实际上采样方式；
- final fusion 中的 bilinear geometry；
- threshold 后的 component extraction；
- 或辅助输出空间中任务最优的离散表示。

二者不能混用。

## 3. 必须做 augmentation-aware 在线审计

离线原始 mask 审计不能代表训练期间的真实 graph，因为 resize、crop 和 padding 会改变：

- 目标相对 stride；
- pooling grid phase；
- 边界截断；
- merge probability；
- occupancy ratio。

正式实验应同时报告：

1. 原始训练集静态审计；
2. 使用真实 train transform 的 Monte Carlo 审计；
3. 训练期间在线 graph/budget 统计。

---

# 六、当前最小而充分的控制实验

在 Phase-A GO 前，不建议直接进行大规模 full training。先运行以下机制门控矩阵。

| 编号 | 方法 | 必须匹配的内容 | 回答的问题 |
|---|---|---|---|
| G0 | canonical MSHNet | 原始配置 | 基线 |
| G1 | fixed scale-drop | 固定删除 stride-8，必要时再测 stride-4/8 | TCDS 是否只是删粗尺度 |
| G2 | stochastic scale-drop | 每 head 匹配 TCDS active sample ratio | 收益是否来自动态 loss attenuation |
| G3 | TCDS-Projection | 当前 binary graph + Partial-SLS | 语义选择是否有效 |
| G4 | matched random graph | 匹配 head degree、active pairs、positive pixels、valid ratio | 可行边语义是否优于随机边 |
| G5 | area/occupancy-matched graph | 匹配图预算，仅依据面积或 cell occupancy | 当前 graph 是否只是 size gate |

在 G3 对 G1、G2、G4、G5 明确占优后，再进入完整论文矩阵：

- final-only；
- side-no-location；
- improved projection，如 CPP / soft / multi-label projection；
- 非 MSHNet backbone。

## matched random 至少要保持四类边际量

一个可信 matched random graph 至少保持：

1. 每个实例的 head degree；
2. 每个 head 的实例数量；
3. 每个 head 的 active sample–head 数量；
4. 每个 head 的正 coarse-pixel 总量近似一致。

最好进一步保持：

- valid/unknown pixel ratio；
- object area bins；
- object-to-stride ratio bins；
- 每 head 的 target-size histogram。

可采用同 bin 内的二部图 2×2 edge swap，并对正像素预算设置拒绝采样容差。

---

# 七、Phase-A 梯度 influence 应如何实现

## 1. 主分析使用连续恢复质量，不要先固定阈值

阈值敏感性已经表明，直接用 \(\tau=0.5\) 划分可行/不可行可能把结论锁定为尺度裁剪。

Phase A 应先把以下量作为连续变量：

- recovery IoU；
- centroid distance；
- merge/split/disappearance indicator；
- cell occupancy；
- object-to-stride ratio。

分析恢复质量与有害更新之间是否存在稳定单调关系。只有在关系成立后，才定义二值边界。

## 2. 使用实例边际辅助梯度

单实例图像：

\[
g_{k,s}
=
\nabla_{\theta_b}L_s(G_k).
\]

多实例图像：

\[
\Delta g_{k,s}
=
\nabla_{\theta_b}L_s(Y)
-
\nabla_{\theta_b}L_s(Y\setminus G_k).
\]

其中 \(\theta_b\) 只选择共享 encoder/decoder 参数，不选择 side output conv 和 final fusion 参数，避免 final 与 side 的直接结构耦合主导结论。

由于 SLS 的 IoU、scale factor、target mass 和 LLoss 都是图像级非加性函数，
该差分是 **contextual marginal gradient**，不是严格可分离的单实例梯度。单实例
图像应作为最干净的主分析；多实例 LOCO 需要按 instance count、merge 状态和场景
组成分层报告。

## 3. probe batch 必须独立

对独立 probe batch：

\[
g_f^{\mathrm{probe}}
=
\nabla_{\theta_b}L_f(B_{\mathrm{probe}}).
\]

基础一阶 influence 为：

\[
I_{k,s}
=
\Delta g_{k,s}^{\top}g_f^{\mathrm{probe}}.
\]

但训练优化器是 Adagrad，因此正式主指标应使用**优化器感知的更新方向**。

同一个 probe gradient 若被大量 source instances 复用，会引入共同相关性。因此
需要多个固定 probe batches，并对 source image 与 probe batch 做双重聚类或两维
bootstrap。用于阈值诊断的 probe、checkpoint-selection validation 和最终 sealed
test 不能是同一份数据；优先采用 official-train manifest 内 cross-fitting。

## 4. 使用 optimizer-aware influence

不能把旧 Adagrad 状态近似成固定预条件器后孤立更新 \(\Delta g_{k,s}\)。Adagrad
会使用本次完整联合梯度的平方更新 accumulator，实例边还会与 final 和其他 side
objectives 相互作用。定义不含该边的完整训练梯度为 \(g_0\)，含边梯度为：

\[
g_1=g_0+\Delta g_{k,s}.
\]

从相同参数和 optimizer state \(S_t\) 出发：

\[
u_0=\operatorname{AdagradStep}(g_0;S_t),\qquad
u_1=\operatorname{AdagradStep}(g_1;S_t),
\]

\[
\Delta u_{k,s}=u_1-u_0.
\]

统一定义 optimizer-aware harm score：

\[
H_{k,s}
=
\left(g_f^{\mathrm{probe}}\right)^\top\Delta u_{k,s}.
\]

- \(H_{k,s}<0\)：加入该边预计改善 probe final objective；
- \(H_{k,s}>0\)：加入该边预计损害 probe final objective。

原始 gradient dot product 和 cosine 可以作为补充，但不能替代实际优化器更新空间中的 influence。

## 5. one-step intervention 必须复制模型和优化器状态

对代表性实例—head 对：

1. 复制模型参数；
2. 复制 Adagrad accumulator；
3. 分支 A 执行包含该边的完整训练目标；
4. 分支 B 执行完全相同但移除该边的完整训练目标；
5. 测量两个更新后模型在独立 probe 上的 loss 差：

\[
H^{\mathrm{actual}}_{k,s}
=L_f^{\mathrm{probe}}(\theta_{\mathrm{with}}')
-L_f^{\mathrm{probe}}(\theta_{\mathrm{without}}').
\]

再检验 \(H^{\mathrm{actual}}\) 与一阶 \(H\) 的符号一致性。当前
`tools/optimizer_counterfactual.py` 提供复制模型和 optimizer state 的双分支基础
工具；正式审计仍需将实例—head edge 构造成两份仅差一条边的完整 loss closure。

## 6. 审计阶段

至少覆盖：

- 初始化；
- warm-up 结束；
- 训练中期；
- best checkpoint；
- final checkpoint。

单元测试还暴露出一个必须报告的边界：当 Adagrad accumulator 为零且两个分支的
标量梯度同号时，首次更新可能被归一化成相同步长，使真实 marginal update 为零。
这不是“该边无梯度”，而是 optimizer state 的特殊效应；因此初始化审计不能替代
warm-up 后和中期 checkpoint 审计。

## 7. 统计方法

实例来自同一图像、同一 checkpoint 和多个 heads，不是独立样本。建议：

- 对 source image 与 probe batch 做双重聚类或两维 bootstrap；
- 报告 effect size 与 95% CI；
- 使用分层回归控制 object area、SCR、instance count、head 和 checkpoint；
- 报告 recovery score 预测 harmful update 的 AUROC/AUPRC；
- 同时报告 raw influence、cosine 和 finite-difference result。

---

# 八、阈值应如何确定

不应根据最终验证集性能遍历 `min_iou` 后选最佳值，否则方法会退化为一个可调尺度门控器。

推荐优先级如下。

## 方案 A：由任务协议给出

若任务成功条件本身包含实例 IoU 阈值，则直接采用该阈值。当前 IRSTD 常用指标主要是全局 IoU 和中心匹配 PD/FA，因此必须解释为什么实例级 `IoU >= 0.5` 是任务容差，而不是借用通用实例分割阈值。

## 方案 B：由 Phase-A influence 零交叉确定

在预注册诊断 split 上估计：

\[
\tau^*
=
\inf\left\{
 r:\mathbb E[H\mid r]\le 0
\right\},
\]

然后：

- 冻结该阈值；
- 不再根据最终性能调节；
- 跨数据集和跨 backbone 使用同一个阈值；
- 报告阈值迁移是否成立。

## 方案 C：不把低 IoU 直接作为 hard gate

如果 influence 审计显示真正稳定有害的主要是：

- merge；
- split；
- disappearance；
- 超出任务定位容差；

而 recovery IoU 与 influence 没有清晰零交叉，则第一版只对这些结构事件设 unknown，把 IoU 保留为连续诊断变量。这样比任意硬阈值更稳健，也更接近“结构性错误监督”的论文主张。

## 必须绘制预算曲线

无论最终选择何种阈值，都应绘制：

- x 轴：retained edge / positive-pixel / active-pair budget；
- y 轴：probe influence 或最终任务指标；
- 曲线：TCDS、matched random、area/occupancy control。

只有 TCDS 在相同预算上稳定优于 controls，才能说明语义选择有效。

---

# 九、投影审计还需新增的诊断

## 1. rejection reason composition

静态审计工具已经按数据集和 stride 输出：

- disappeared；
- split；
- merged；
- low IoU；
- localization；
- feasible。

下一步不是重复实现该统计，而是扩展为多阈值、真实 augmentation、面积/occupancy
分层，并分析每类 rejection reason 对 harm score \(H\) 的独立解释能力。

如果 stride-4/8 的拒绝几乎全部来自 low IoU，而 merge/split 极少，则当前方法主要是 occupancy/area gating，论文不能突出拓扑不可恢复性。

## 2. graph nestedness

对于嵌套分辨率，检查：

\[
A_{k,s=8}=1
\Rightarrow
A_{k,s=4}=1
\Rightarrow
A_{k,s=2}=1.
\]

报告违反比例并可视化样例。大量非单调边通常意味着：

- grid phase；
- 尺寸不整除；
- projector geometry；
- merge interaction；
- 阈值边界

正在主导 assignment。

当前 `min_iou=0.5` 的三个训练集静态审计在全部相邻尺度上均为 0 nestedness
violations。这个结果排除了“大量非单调 grid-phase 决策”作为当前主要现象，但
同时表明 graph 极接近单调尺度/面积筛选；必须依靠 area/occupancy-matched control
而不是 nestedness 本身证明 scene-level 语义价值。

## 3. area-predictability

训练一个只使用：

- object area；
- width/height；
- cell occupancy；
- object-to-stride ratio

预测当前 feasibility 的简单模型。

若其 AUROC 接近 1，说明 TCDS graph 基本可以被尺度统计复现，必须依赖 matched area/occupancy control 证明 scene-level task semantics 的附加价值。

## 4. augmentation and grid-phase sensitivity

对同一实例在不同 crop offset / resize 下统计 graph 翻转率：

\[
\Pr\bigl(A_{k,s}^{(t_1)}\ne A_{k,s}^{(t_2)}\bigr).
\]

如果翻转率很高，阈值决策主要受 pooling 网格相位影响，需要在论文中解释或重新定义投影恢复度量。

---

# 十、仍未解决的 baseline 问题

canonical MSHNet 仍需与额外 `decidability_head` 物理隔离。即使该 head 不参与普通 forward/backward，它仍然污染：

- parameter count；
- `state_dict`；
- checkpoint strict identity；
- 模型结构表述；
- 论文中“使用原始 MSHNet”的可验证性。

正式主实验前必须完成：

1. canonical class 与官方实现结构一致；
2. 官方 checkpoint `strict=True` 加载；
3. 参数量、state keys、固定输入输出一致；
4. canonical loss 和训练拓扑一致；
5. TCDS 仅作为训练监督 wrapper，不修改模型类。

这是工程问题，但对顶会复现和审稿可信度属于硬门槛。

---

# 十一、建议补充的测试

现有 126 个测试说明工程基础较好，下一轮重点应从“能运行”转向“研究不变量”。

本轮已完成：

- mixed-batch canonical identity；
- supported-shape border target；
- sequential pooling vs direct stride pooling；
- instance-ID permutation invariance；
- graph nestedness diagnostics；
- exact two-branch optimizer-state preservation。

仍建议新增：

1. **actual-side-shape target alignment**；
2. **merge/split participant order invariance**；
3. **unknown gradient zero for IoU and location branches**；
4. **empty image / no feasible edge behavior**；
5. **matched random marginal preservation**；
6. **CPU/GPU and deterministic-repeat consistency**；
7. **epoch-level supervision-budget aggregation correctness**；
8. **real instance-edge full-objective counterfactual closure**。

---

# 十二、Phase-A GO / NO-GO 门槛

建议在正式训练前冻结以下判断规则。

## GO

至少满足：

1. recovery distortion 与 optimizer-aware harmful influence 存在稳定单调关系；
2. 不可恢复或低恢复边的 harmful-update 比例显著高于高恢复边；
3. image-cluster bootstrap 的 95% CI 不跨零；
4. one-step intervention 与一阶预测具有足够高的符号一致性；
5. 结果在至少两个数据集、两个训练阶段上方向一致；
6. 控制 area、SCR、instance count 后，recovery distortion 仍有独立解释力；
7. TCDS 在相同 active/positive-pixel budget 下优于 matched random；
8. TCDS 优于 fixed/stochastic scale-drop，而不只是等价于删除 coarse heads。

## NO-GO / Pivot

出现以下任一主情形，应停止宣称 task-feasible supervision：

- harmful influence 与 recovery distortion 无稳定关系；
- 图决策几乎完全可由 area/occupancy 预测；
- TCDS 与 matched scale-drop 表现相同；
- 只有通过数据集特定阈值调优才能获得收益；
- stride-4/8 长期接近空图且收益等价于移除这些 heads；
- improved projection 能完全替代 unknown gating。

对应 pivot：

- 若 improved projection 更有效：转向 **task-consistent label projection**；
- 若只需删 coarse supervision：转向 **deep-supervision topology/scale scheduling**，不要包装成 feasibility；
- 若只有 merge/split 产生稳定伤害：收敛为 **topology-preserving deep supervision**；
- 若 \(\epsilon^*\) 与 fixed projection 差距大：研究 projection gap，而不是 representation gap。

---

# 十三、论文当前可以与不可以主张的内容

## 当前可以主张

- 给定 max-pool 辅助标签投影在不同 head 上产生显著不同的 scene-level task recovery；
- 该恢复度可以通过统一的 component matching 进行审计；
- binary partial-label 训练可以在不改变网络和显式 loss ratio 的情况下实现；
- unknown 冲突、merge/split 和 all-valid degeneration 已有严格工程处理。

## 当前不可以主张

- TCDS/TFDS 提高了性能；
- 不可恢复标签已经被证明产生有害梯度；
- stride-8 应被删除；
- `IoU >= 0.5` 是正确或普适的可行边界；
- 辅助输出空间本身不可表达，而不是当前 projector 不好；
- 当前 smoke 或可行率表证明方法有效；
- `rods_random` 是 matched control；
- canonical MSHNet 已与扩展模型完全相同。

## 若 Phase A 与 controls 成立，可形成的核心贡献

> Deep supervision is globally beneficial yet locally harmful when its auxiliary label operator maps an instance to a task-inconsistent target. TCDS identifies such instance–head pairs in the task space and converts only their ambiguous support to unknown, without changing the inference network.

若进一步估计 \(\epsilon^*\)，才升级为：

> TFDS separates projection error from intrinsic output-space infeasibility.

---

# 十四、下一步执行顺序

## 数据与划分协议

数据根目录固定为 `datasets/`，只读取各数据集 `img_idx/*.txt`。manifest 读取时
必须 `.strip()` 或显式移除 CRLF 中的 `\r`。当前数量为：

| Dataset | official train | official test |
|---|---:|---:|
| NUAA-SIRST | 213 | 214 |
| NUDT-SIRST | 663 | 664 |
| IRSTD-1K | 800 | 201 |

train/test 在换行归一化后均无重叠。`hcval_NUDT-SIRST.txt` 的 6 个样本全部属于
official test，禁止用于 probe、阈值确定、checkpoint selection 或独立验证。
Phase-A 使用 official train 内部 disjoint/cross-fitted diagnostic 与 probe folds，
official test 保持 sealed。

## P1：先完成 baseline isolation

不要带着额外 `decidability_head` 进入正式 baseline 实验。

## P2：补齐投影与损失预算审计

优先增加：

- reason composition；
- active sample ratio；
- positive/valid/unknown pixel budget；
- gradient norm；
- graph nestedness；
- augmentation/grid-phase sensitivity；
- odd-size projector identity。

## P3：完成 Phase-A influence

先连续分析 recovery score，不先锁定 `min_iou=0.5`。

## P4：实现三个关键 controls

1. fixed/stochastic scale-drop；
2. active/degree/pixel-matched random graph；
3. area/occupancy-matched graph。

## P5：再进行有限正式训练

只有 Phase-A GO 后，才运行：

- canonical；
- final-only；
- side-no-location；
- TCDS；
- matched controls；
- improved projection。

## P6：最后扩展 backbone 与完整 TFDS

TCDS 在 MSHNet 上机制成立后，再：

- 加入一个非 MSHNet 深监督网络；
- 检验阈值或 influence boundary 的跨模型迁移；
- 评估是否值得求解 \(\epsilon^*\)；
- 决定论文最终命名为 TCDS 还是 TFDS。

---

# 十五、最终结论

这轮修改已经把项目从：

> 多个尺度启发式、责任权重和 loss 变化混合在一起的 RODS 原型

推进到了：

> 一个网络不变、显式损失比例不变、基于完整场景投影恢复构造 binary partial labels 的 TCDS-Projection 原型。

这是实质进展。

但当前最主要的风险也更清晰了：

\[
\boxed{
\text{当前 IoU 阈值可能把 task consistency 退化为 occupancy/area gate，}
\\
\text{而无正实例零梯度又把 graph density 转化为隐式 auxiliary weight。}
}
\]

因此，下一步的核心不是继续训练并寻找一个“最好阈值”，而是依次证明：

\[
\text{projection distortion}
\rightarrow
\text{harmful optimizer update}
\rightarrow
\text{semantic filtering beats matched budget controls}.
\]

只有这三步成立，当前实现才从可靠的工程原型升级为顶会方法；如果其中任何一步失败，也能明确 pivot 到标签投影、拓扑保持或深监督尺度调度，而不是继续增加模块。
