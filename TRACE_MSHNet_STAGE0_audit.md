# TRACE-MSHNet Stage 0：状态空间、数据合同与 baseline 锁定审计

## 结论

**当前结论为 `NO-GO`，不能进入 TRACE solver 实现阶段。**

这不是性能未达标，而是监督标签不属于预设结构变量。三个数据集都包含多目标图像，且训练增强后仍有大量多组件标签；另有真实组件不属于“每个占用行只有一个连续 run”的最小可解组件族。于是

\[
Y\in\{\varnothing\}\cup\mathcal C
\]

无法为这些 ground truth 定义结构化 NLL。把输出扩成真实 maximal-8CC set 后，当前唯一语义达标的直接 row-frontier 构造需要显式跟踪 frontier occupancy、未闭合组件的 connectivity partition，并在路径中恢复 canonical root；仅 occupancy 在宽度 256 时就有 \(2^{256}\) 种，因而该构造不能在 256×256 上作为可运行 exact layer。这里的状态计数是对已审计显式 transfer family 的下界，不冒充对所有可能代数算法的无条件复杂度定理。

固定切片的边界现已进一步收紧。对任意固定、非重叠的坐标 partition \(q:[256]^2\to\mathcal S\)，internal-fit 增广支持中的正概率事件覆盖了输出网格全部 130,560 条四邻边；“组件不可被切开”迫使每条边两端拥有相同 \(q\)，所以连通网格上的 \(q\) 必为常数。每套 fit 又都有正概率多组件事件，要求至少两个不同 slice，形成严格矛盾。该证明不限矩形、tile 数量或 phase；但它不外推到 overlapping crops、sample-adaptive slicing，也不允许把 duplicate/drop/merge 偷换成无损切片。

另一个纯代数 loophole 是对小 block 或像素建立 bounded-treewidth mask CRF，再把每个二值 mask 事后解释为其 maximal-8CC set。它确实可以 exact normalize，但 component、root 和完整 support 不在同一 TRACE state/path 中，本质仍是用户明确排除的 dense-mask structured surrogate，不能作为 TRACE 或顶会创新。

按照预注册规则，Stage 0 未通过时不得实现缩水版本，不得取最大组件、做 hull、合并标签、跳过非法样本，也不得用 NMS、连通域筛选或形态学后处理补救。因此本轮没有新增 TRACE 网络或近似 solver。

## 1. Canonical baseline 合同

唯一 baseline 是从 commit

```text
46cdfd46802629da51f70124662af7335be74b56
```

恢复的 canonical MSHNet。当前隔离实现位于：

- `model/baselines/mshnet_official.py`
- `model/baselines/mshnet_deterministic.py`

deterministic 版本与 canonical 版本参数 schema 和前向完全一致，仅把 channel-attention 的 global max reduction 换成确定性 tie-backward 语义。这个替换在并列最大值处改变 backward 子梯度，所以它是**参数/前向等价的 deterministic-backward baseline**，不能写成与 source commit 训练算子完全相同。provenance 测试会直接从上述 commit 读取历史 `model/MSHNet.py`，逐参数、逐输出核对，并另测 tie-backward 差异，而不是相信文件名或事后 metadata。

这里的 commit provenance **只锁定模型定义**，不把当前仓库的整套训练/评价代码冒充为该历史 commit。当前 `model/loss.py`、`utils/data.py` 和 `utils/metric.py` 已有后续修订；尤其当前 PD/FA 使用“每个 GT 匹配最近的未匹配预测组件”，历史实现按 region 顺序接受第一个距离小于 3 的组件。新 baseline 因而只能与使用当前冻结协议重跑的 control 做配对比较，不能和历史 commit 的旧 PD/FA 数值直接作论文结论。运行期依赖、环境和实际 fit/val image/mask bytes 另由 runtime attestation 记录；该补充是在进程运行中捕获，不伪称 launch-time manifest。

冻结的开发协议：

| 字段 | 固定值 |
|---|---|
| model | `mshnet` |
| physical variant | `deterministic` |
| supervision | `legacy_exact` |
| fusion regularizer | `none` |
| split | official train 内固定 80/20 holdout |
| split seed | `20260711` |
| model seeds | `20260711, 20260712, 20260713`，全部报告 |
| optimizer / LR | Adagrad / 0.05 |
| epochs | 400 |
| validation cadence | 每 10 epoch |
| selection | 每个 seed 仅按 internal-val IoU 选择 |
| workers | 0 |
| resume | 禁止；当前 checkpoint 不是 exact-resumable |
| official test（baseline 训练流程） | 只读 ID manifest 做 hash/overlap 审计；不迭代 image/mask、不评价、不选 checkpoint |

新三种子运行位于：

```text
weight/clean/trace_stage0_canonical_mshnet_nuaa_holdout_v1/
repro_runs/clean/trace_stage0_canonical_mshnet_nuaa_holdout_v1/
```

三个 fresh 400-epoch 运行均已自然结束，runner return code 全为 0；strict finalizer 对 source/split/command、40 个固定 validation epoch、best/latest checkpoint、canonical state schema、finite weights、运行中 attestation 与实际数据 bytes 全部 fail-closed 核验通过：

| Seed | internal-val best epoch | IoU | PD | FA/Mpixel |
|---:|---:|---:|---:|---:|
| 20260711 | 359 | 0.680293 | 0.981481 | 61.745 |
| 20260712 | 129 | 0.674387 | 0.944444 | 39.744 |
| 20260713 | 389 | 0.707264 | 0.981481 | 50.744 |
| mean ± sample SD | 292.33 ± 142.24 | 0.687315 ± 0.017527 | 0.969136 ± 0.021383 | 50.744 ± 11.001 |

这些数值只属于 NUAA official-train 内部固定 holdout，不是 official-test 结果，也不能进入论文主表。summary JSON SHA256 为 `fca52b6ad4444d23ae65a7bd7a8860f7c6576f65929a38ce7853e22799168e33`，对应 Markdown SHA256 为 `f85b11c0feb6ee5a19f700e7000ec372ed532eb51ad1b2b564758e8ba9a0dbfa`。历史迁移结果和 test-selected 结果不进入正式证据。

## 2. MSHNet 中允许修改和必须删除的边界

MSHNet 至 `d0` 的结构语义被冻结：

```text
input -> conv_init -> encoder_0..3 -> middle -> decoder_3..0 -> d0
```

原模型共 `4,065,513` 个参数；其中必须被 TRACE 整体替换的 dense prediction path 只有 `281` 个参数：

| 层 | 参数量 |
|---|---:|
| `output_0` | 17 |
| `output_1` | 33 |
| `output_2` | 65 |
| `output_3` | 129 |
| `final` 4→1 conv | 37 |
| 合计 | 281 |

`d0` 及其之前共有 `4,065,232` 个参数。若 Stage 0 将来能够解除，TRACE 和 dense-head control 都必须使用同一个 `d0`，不能保留 side head、辅助损失或第二条预测路径。

## 3. 原始/验证标签的组件审计

统计口径与仓库评价一致：mask 阈值为 0.5，8-connect；验证/测试标签使用 NEAREST resize 到 256×256。这里必须区分两个流程：baseline 训练没有迭代 official-test image/mask；独立的 Stage-0 任务定义审计读取了 test mask 并在下表披露描述性统计。因此 official test **不是全局 sealed**。下面所有导致 `NO-GO` 的设计判据只使用 official-train、internal-fit/val 与其真实增强流；test 行不用于选择组件族、solver restriction、超参数或模型。

| Dataset / split | Empty | Single | Multi | Components | 最大 K |
|---|---:|---:|---:|---:|---:|
| NUAA train (213) | 0 | 178 | 35 | 270 | 7 |
| NUAA test (214) | 0 | 180 | 34 | 263 | 6 |
| NUDT train (663) | 0 | 431 | 232 | 918 | 4 |
| NUDT test (664) | 0 | 430 | 234 | 945 | 4 |
| IRSTD train raw (800) | 0 | 492 | 308 | 1195 | 6 |
| IRSTD train eval-256 (800) | 4 | 488 | 308 | 1192 | 6 |
| IRSTD test raw (201) | 0 | 131 | 70 | 297 | 8 |
| IRSTD test eval-256 (201) | 1 | 131 | 69 | 294 | 7 |

固定 `split_seed=20260711` 后，开发期的 fit/val 细分为：

| Dataset / split | Empty | Single | Multi | Components | 最大 K |
|---|---:|---:|---:|---:|---:|
| NUAA fit 170 | 0 | 142 | 28 | 216 | 7 |
| NUAA val 43 | 0 | 36 | 7 | 54 | 4 |
| NUDT fit 530 | 0 | 350 | 180 | 731 | 4 |
| NUDT val 133 | 0 | 81 | 52 | 187 | 4 |
| IRSTD fit eval-256 640 | 2 | 401 | 237 | 940 | 5 |
| IRSTD val eval-256 160 | 2 | 87 | 71 | 252 | 6 |

若每图只保留一个组件，即使总是保留最大组件，official-train 的 eval-256 标签至少会删除：

| Dataset | 被删除组件 / 总组件 | 被删除前景像素 / 总前景像素 |
|---|---:|---:|
| NUAA | 57 / 270 | 1,163 / 8,714 |
| NUDT | 255 / 918 | 5,918 / 29,337 |
| IRSTD eval-256 | 396 / 1,192 | 3,272 / 15,105 |

这不是可忽略的 annotation noise；它会系统性改变任务。

官方 train 的完整组件数分布为：

```text
NUAA:  {1:178, 2:24, 3:5, 4:4, 6:1, 7:1}
NUDT:  {1:431, 2:215, 3:11, 4:6}
IRSTD eval-256: {0:4, 1:488, 2:235, 3:61, 4:10, 5:1, 6:1}
```

任何 train empirical maximum 都只是有限样本统计；没有采集/标注协议给出的先验上界证明时，不能把它升级成任务级严格状态上界。上表中的 test `Kmax` 只作为这一风险的事后披露，不参与状态空间选择；即使完全删除所有 test 行，official-train 与增强流也已经使当前 single-component / single-row-run 定义失败。

## 4. 实际 internal-fit 增强标签流

以下统计复现 paired protocol 的真实 DataLoader 行为：相同 internal-fit、batch size 4、`drop_last=True`、`num_workers=0`，三个预注册 seed 各走一个完整 epoch。增强与 `utils/data.py` 完全相同：随机镜像、0.5–2.0 随机缩放、NEAREST mask resize、pad、随机 256 crop。

`SSR` 表示每个 connected component 的每个占用行恰有一个连续 run。

| Dataset | 增强样本 | Empty | Single | Multi | Kmax | 非 SSR 组件 | 不属于 set-of-SSR 的样本 | 不属于 ∅/single-SSR 的样本 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NUAA | 504 | 71 | 374 | 59 | 7 | 35 / 528 (6.63%) | 32 / 504 (6.35%) | 83 / 504 (16.47%) |
| NUDT | 1584 | 332 | 911 | 341 | 4 | 523 / 1633 (32.03%) | 468 / 1584 (29.55%) | 738 / 1584 (46.59%) |
| IRSTD | 1920 | 351 | 1037 | 532 | 5 | 214 / 2246 (9.53%) | 212 / 1920 (11.04%) | 657 / 1920 (34.22%) |

训练增强不是对最小组件族的闭包：crop 会得到空标签、保留多个目标、截断组件，NEAREST resize 还可能删除细目标或连接、拆分原组件。对非法标签做 projection 或跳过样本都会改变 paired training protocol。

原始 official-train eval-256 本身已出现以下最大结构复杂度：

| Dataset | 单组件单行最大 run 数 | 单图同一行最大并发组件 | 单图同一行最大并发 run |
|---|---:|---:|---:|
| NUAA | 2 | 3 | 3 |
| NUDT | 5 | 2 | 5 |
| IRSTD | 3 | 4 | 4 |

不属于 single-row-run 的组件比例为 NUAA `16/270=5.93%`、NUDT `311/918=33.88%`、IRSTD eval-256 `124/1192=10.40%`。eval-256 official train 中，NUDT 有 35 个带孔组件，IRSTD 有 3 个带孔组件；因此“每行一个 run + hole-free”不是完整标注空间。

## 5. 为什么最小 single-component TRACE 数学上可解但数据不完备

条件可解的最小 8-connected 组件族为

\[
\mathcal C_1=\left\{
\bigcup_{r=s}^{t}\{r\}\times[l_r,u_r]:
l_r\le u_r,\ [l_r,u_r]\sim_1[l_{r+1},u_{r+1}]
\right\},
\]

其中

\[
[a,b]\sim_1[l,u]\iff a\le u+1,\quad b\ge l-1.
\]

canonical root 是 support 的确定函数

\[
\rho(C)=(s,l_s),
\]

即 row-major 最小像素。它保证 path 与 support 一一对应，但不是额外的自由 latent variable。

同一个 interval-transition 图可以在 log-sum-exp semiring 下求 partition/NLL/marginal，在 max semiring 下求 MAP。单行有

\[
N=W(W+1)/2=32,896
\]

个 interval；256 行共有 8,421,376 个 row-state。利用二维 prefix/suffix LSE 或 max range query，可把朴素 \(O(HW^4)\) 转移降到 \(O(HW^2)\)。

但该多项式复杂度来自“每行只保留一个 run”的强限制。它不能表示：

- 多个 connected components；
- 一个组件在同一行出现多个 run；
- U 形、带孔或其他非 row-convex support；
- 增强造成的 component split。

因此它不能用于当前监督流的 exact NLL。

零势时 256×256 单组件族已经极大：

```text
log |C_4-connect| = 2581.948
log |C_8-connect| = 2583.279
```

若空集 score 为 0、非空 start score 未做组合基数校准，则零初始化时 `P(exist)≈1`。任何未来合法版本都必须把 `-log|C|` 作为一次性 start offset（或给出等价、预注册的校准），否则 empty 分支在训练开始前就数值死亡。

## 6. 多目标路线的可行性

| 路线 | 数据完备 | 256×256 exact | 输出是真实组件集合 | 结论 |
|---|---:|---:|---:|---|
| Full frontier set-DP | 是 | 否 | 是 | 不可算 |
| 固定非重叠坐标 partition | 否 | 单片可算 | 否 | **被增广支持严格否证** |
| sample-adaptive / overlapping slicing | 未建立 | 取决于全局组合 | 未建立 | 无合法冻结协议；不作全称否定 |
| Ordered-root RFS，禁止 overlap/邻接 | 可完整 | 否 | 是 | hard-core set packing |
| Ordered-root RFS，允许 overlap | 表面完整 | 可能 | 否 | 同一 mask 多重分解 |
| 垂直分离/fixed slots | 否 | 是 | 是 | 标签空间不完备 |

### 6.1 Full frontier DP

对任意 2D 连通 support，直接的逐行 frontier DP 用 occupancy（至多 \(2^W\) 种 pattern）以及 frontier runs 在已处理区域中的 connectivity partition（Catalan/Motzkin 级）保持组件语义。这个显式 transfer construction 在 `W=256` 不可计算；本节不把它外推成对任意非 frontier、允许代数 cancellation 的算法下界。

### 6.2 固定切片

合法切片必须在查看 validation/test mask 之前固定，并同时保证：

1. 每个真实组件完整落入唯一切片；
2. 每片至多一个组件；
3. 不同切片输出不会在边界邻接后并成一个组件；
4. 不使用 overlap-window merge、NMS 或 target-centered test crop。

对 official-train 穷举 uniform tile 的高、宽 `1..256` 以及全局 phase 后，三个数据集均没有一个配置同时满足前两项。整图 tile 不截断目标，但违反每片至多一个目标。

更强的结论不再局限于 uniform rectangles。令 \(q(v)\) 是输出坐标 \(v\) 的固定 slice label。符号增广支持审计只使用 deterministic internal-fit 成员，并验证 image/mask 尺寸与仓库按 `image.size` 决定 resize geometry 的语义一致：

1. NUAA 的 `Misc_11`/`Misc_119`、NUDT 的 `000848`、IRSTD 的 `XDU730` 在合法 `long_size=512`、flip 与全部 crop offset 的一个子支持中，使每条水平/垂直输出边都能成为同一 8CC 内的一对相邻前景像素；
2. no-split 因而对全部 130,560 条四邻边施加 \(q(u)=q(v)\)；
3. 256×256 四邻网格连通，所以 65,536 个坐标只能有一个 slice label；
4. fit 成员 NUAA `Misc_119`、NUDT `000848`、IRSTD `XDU907` 又各有一个合法 `long_size=256` 增强事件保留两个 8CC，要求两个不同 label，矛盾。

每个用于证明的 transform 都有正概率；结论不依赖三 seed 是否恰好抽到这些 crop。它覆盖**任意形状、任意数量 cell 的固定非重叠坐标 partition**。

独立的 union-find 审计把每个观测组件内的坐标作为 equality，同一 mask 的不同组件作为 inequality。固定 partition 存在，当且仅当 equality quotient 中没有 inequality self-loop；这个判据已在 1×3 网格的全部 256 个 mask corpus 与 5 个 set partition 上穷举验证，零不一致。真实结果为：

| 数据集 | official-train self-loop | 三 seed 一 epoch aggregate self-loop | seed 20260711 / 12 / 13 |
|---|---:|---:|---:|
| NUAA | 4 | 6 | 1 / 1 / 2 |
| NUDT | 2 | 282 | 3 / 5 / 1 |
| IRSTD | 9 | 87 | 1 / 4 / 7 |

这一有限流审计只是符号支持定理的独立实现证据，不外推成“观察了 400 epoch”。两者均不读取 official-test 文件。

对 sample-adaptive 的 full-axis guillotine，即使先给 oracle mask，IRSTD 的 internal-validation 成员 `XDU526` 仍有两个组件在 x、y projection 上同时重叠：一个组件 bbox 为 `x=50..54,y=124..129`，另一个是 `(54,128)` singleton；任何完整水平或垂直 separating cut 都会切开前者。它只能用于拒绝预先定义的 guillotine family，不能反过来参与新协议设计。允许任意曲线 separator 会重新引入 separator/frontier connectivity 状态。本文仍不声称否定所有可能的 input-adaptive 或 overlapping protocol；但当前架构约束下没有可冻结、无第二预测路径且无需 merge/drop 的合法实例。

### 6.3 Ordered-root random finite set

把组件按 root 排序只能消除 `K!` 排列重复，不能消除 overlap/adjacency conflict。禁止冲突时，normalizer 是 component-conflict graph 上的 hard-core partition；仅一行两个不交 interval 就有

\[
\binom{257}{4}=177,556,160
\]

种状态，尚未计跨行 connectivity。

若允许 overlap，多个 latent component sets 会 rasterize 成同一个 binary mask；相邻或重叠实例的 union 也不再是原来的组件集合。merge/去重后的输出不是原分布的 MAP，故不满足 TRACE 定义。

更直接地，一行恰有 `r` 个非空 run 的 support 数为

\[
\binom{W+1}{2r}=\binom{257}{2r}.
\]

允许最多 `R` 个 run 时，仅单行显式 pattern（尚未加入 frontier connectivity partition）就有：

| R | row patterns |
|---:|---:|
| 1 | 32,897 |
| 2 | 177,589,057 |
| 3 | 377,519,940,289 |
| 4 | 423,203,101,008,289 |
| 5 | 290,537,928,457,798,689 |

NUDT official-train 已观察到的样本要求 `R≥5`；这是经验下界，不是任务级真实上界。这一规模也不是通过 CUDA vectorization 可以消除的常数开销。

### 6.4 Exact set-valued TRACE 的系统路线审计

对预注册的 row-run/component 语义，本次审计先写出一个语义忠实的直接充分状态。处理第 \(r\) 行后可取

\[
q_r=(B_r,\pi_r,\rho_r,e_r),
\]

其中 \(B_r\) 是 frontier occupancy/runs，\(\pi_r\) 是它们在已处理区域中的 8-connectivity partition，\(\rho_r\) 是各未闭合组件可恢复的 row-major canonical root 信息，\(e_r\) 记录 empty/闭合事件。具体实现可以在 birth/closure 时汇总部分字段，所以这里不声称该 tuple 是所有算法的唯一最小状态；关键是这个显式 component-path transfer graph 确实可在 log-sum-exp/max semiring 下复用同一转移来计算 logZ、NLL、存在概率、条件 set 分布、marginal 与 MAP，但它在宽度 256 时不可运行：

- 任意 frontier occupancy 有 \(2^{256}\approx1.158\times10^{77}\) 种；
- 即使强制所有 frontier 前景像素互不相邻，仍有 \(F_{258}\approx3.710\times10^{53}\) 种；
- 限制每行至多 5 个 run 时，仅 row pattern 就有 \(\sum_{j=0}^{5}\binom{257}{2j}=290{,}537{,}928{,}457{,}798{,}689\)，尚未乘 connectivity partition。

在这一显式 transfer family 中，若两个不同 frontier pattern 被合并，取它们首次不同的列，并只允许下一行该列及受控延伸为前景，就会改变“延续哪个组件、何时闭合、root 属于谁以及 component score”。因此，该 component-aware frontier realization 不能用一个与 mask factorization 等价的常数小状态消掉这些区别；本文仍只主张“没有找到满足全部合同的可运行构造”，不主张已经证明所有计算模型均不可能。

| 候选 | exact 性 | 完备/唯一性障碍 | 结论 |
|---|---|---|---|
| full frontier-connectivity DP | 语义正确 | 宽度指数状态 | exact 但不可运行 |
| bounded-run frontier | 可在固定极小 R 下 exact | `R=5` 已不可算，且无增强闭包 | 拒绝 |
| guillotine grammar | chart 可定义 | 真实反例、parse 重复；完整 rectangle chart 逾 10 亿状态 | 拒绝 |
| ordered-root / root-slot RFS | 排序只消除 `K!` | overlap/8-adjacency 形成 hard-core set packing | 拒绝 |
| interval-graph RFS | 特例可 DP | 要求单轴可分，真实标签不完备 | 拒绝 |
| DPP | logZ/marginal 可算 | 一般 MAP 非同一 max recurrence；冲突族不匹配 | 拒绝 |
| orientation mixture | latent Z 可算 | 同一 set 多重表示；latent MAP 不等于 set MAP | 拒绝 |
| spanning forest | forest Z/MAP 可算 | 一个 support 有多棵树，变量不是 component set | 拒绝 |
| small-block/tree mask CRF | 全 mask 空间 exact | component/root 只在输出后解释，退化为 mask CRF | **不符合 TRACE** |

所以当前不是“还差一个 CUDA 优化”，而是完备标签空间、显式 component/root 语义和 256² exact inference 三者不能同时由已审计路线满足。不得把最后一行的 mask-CRF loophole 包装成 TRACE。

## 7. 文献边界与 novelty 风险

没有检索到与完整 TRACE 叙述完全相同的工作，但各组成部分都有成熟先例：

- 一般 connected-region segmentation 的困难性以及 x-monotone 特例的 polynomial exact MAP：[Asano et al., SODA 1996, *Polynomial-time solutions to image segmentation*](https://research.ibm.com/publications/polynomial-time-solutions-to-image-segmentation)。
- connectivity prior 很早已进入分割模型：[Vicente et al., CVPR 2008, *Graph Cut Based Image Segmentation with Connectivity Priors*](https://www.microsoft.com/en-us/research/publication/graph-cut-based-image-segmentation-with-connectivity-priors/)。
- segment-level exact normalized prediction：[Sarawagi and Cohen, NeurIPS 2004, *Semi-Markov CRFs*](https://proceedings.neurips.cc/paper_files/paper/2004/hash/eb06b9db06012a7a4179b8f3cb5384d3-Abstract.html)。
- semiring 与 differentiable DP：[Goodman, 1999, *Semiring Parsing*](https://aclanthology.org/J99-4004/)；[Mensch and Blondel, ICML 2018, *Differentiable Dynamic Programming for Structured Prediction and Attention*](https://proceedings.mlr.press/v80/mensch18a.html)。
- existence/cardinality + element distribution 的 set prediction/RFS：[DeepSetNet (ICCV 2017)](https://openaccess.thecvf.com/content_iccv_2017/html/Rezatofighi_DeepSetNet_Predicting_Sets_ICCV_2017_paper.html) 与 [probabilistic set prediction (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700545.pdf)。
- DPP 的 determinant normalizer 并不提供所需的同递推 exact MAP；一般 DPP MAP 已知为 NP-hard：[Han et al., ICML 2017, *Faster Greedy MAP Inference for Determinantal Point Processes*](https://proceedings.mlr.press/v70/han17a.html)。

因此不能把以下内容分别包装成新贡献：connectivity、row-convex MAP、semiring 切换、对 logZ 求导得到 marginal、Bernoulli `empty vs singleton` 或 canonical root。若数据合同能够合法化，最多可主张它们在 ISTD 中形成了一个统一的 exact normalized connected-support output layer；仍需通过强机制证据避免被评为 obvious composition。

近期 [MSHNet (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html)、[DEFANet (AAAI 2026)](https://ojs.aaai.org/index.php/AAAI/article/view/37368) 等仍主要输出 dense masks，但 [PConv-SD (AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/32996) 和 [InvDet (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html) 也包含 box/detection 输出，故也不能宣称“首个非 dense ISTD”。

## 8. Stage gate

| Gate | 预注册证据 | 当前状态 |
|---|---|---|
| B0 | source-commit 参数/前向 provenance；明确 deterministic tie-backward 差异 | PASS（测试通过） |
| B1 | 三种子 fresh canonical baseline | **PASS**（400 epochs × 3；strict finalizer 通过） |
| D0 | 所有监督标签属于冻结结构空间 | **FAIL** |
| D1 | 已冻结且证明完备的无泄漏 deterministic slicing 或 exact set definition | **FAIL**（任意 fixed-disjoint partition 已被严格否证；完整 set frontier 不可运行） |
| S0 | tiny-grid path/support 双射 | 未运行；被 D0/D1 阻断 |
| S1 | brute-force logZ/MAP/marginal/gradient | 未运行；被 D0/D1 阻断 |
| M0 | 删除 dense heads 后唯一 TRACE path | 未实现；被 D0/D1 阻断 |
| E0 | micro-overfit / paired short gate | 未运行；被 D0/D1 阻断 |

## 9. 数据完整性与可复核哈希

- train/test ID 均无重复、无交集。
- `NUDT hcval` 的 6 个 ID 全部来自 official test，不能作为无泄漏 validation。
- 唯一图像/mask 尺寸不一致为 NUAA `Misc_111`：image `(325,220)`，mask `(592,400)`。
- NUAA、NUDT mask 只有 `{0,255}`；IRSTD 有 85 张 mask 包含中间灰度，统计严格使用 `>=128`。
- official-train mask corpus SHA256（定义见 slicing artifact）：NUAA `77276b8d...93c4`、NUDT `f46afc98...534b`、IRSTD `30195373...69c9`。这些 hash 和全部几何/切片判据均不读取 official test。

internal split ID hashes：

```text
NUAA  fit 2bc2eaae...8d6   val ffea8743...534
NUDT  fit c1c74245...2b3   val 175eafa3...b0d
IRSTD fit 15e4a0d4...227   val 5a997f56...423
```

## 10. 可执行证据与复现入口

组件/增强流审计：

```bash
/home/md0/ly/BasicIRSTD/infrarenet/bin/python \
  tools/audit_trace_component_space.py \
  --datasets NUAA-SIRST,NUDT-SIRST,IRSTD-1K \
  --seeds 20260711,20260712,20260713 \
  --output repro_runs/clean/trace_stage0_component_space_audit_v1.json
```

- 工具：`tools/audit_trace_component_space.py`
- 自包含测试：`tests/test_audit_trace_component_space.py`
- 当前 JSON SHA256：`0831acd64146d400621f25fee48033c4a84abf596560b0e600943c11b86c2c4a`
- 该 artifact 含 test 的描述性任务审计，guardrail 明确禁止把 test 用于 family/model/hyperparameter selection。

official-train-only 几何与切片族审计：

```bash
/home/md0/ly/BasicIRSTD/infrarenet/bin/python \
  tools/audit_trace_slicing_families.py \
  --datasets NUAA-SIRST,NUDT-SIRST,IRSTD-1K \
  --output repro_runs/clean/trace_stage0_slicing_families_v1.json
```

- 工具：`tools/audit_trace_slicing_families.py`
- 自包含测试：`tests/test_audit_trace_slicing_families.py`
- 当前 JSON SHA256：`b1605ab19e8e04ddd067da5062cbbdb56331ad9502c92ba4df5d94bbf22da3af`
- 该工具不读取 official-test split、ID、image 或 mask；结论只覆盖精确定义的 uniform-grid 与 bbox-clear recursive-guillotine 两族。

任意 fixed-coordinate partition 的有限观测必要充分判据：

```bash
/home/md0/ly/BasicIRSTD/infrarenet/bin/python \
  tools/audit_trace_fixed_partition.py \
  --output repro_runs/clean/trace_stage0_fixed_partition_feasibility_v1.json
```

- 工具：`tools/audit_trace_fixed_partition.py`（SHA256 `f52bfce5534ef9d1e95254250d921feefbba0b232195e34c50434ab25916059e`）
- 自包含测试：`tests/test_audit_trace_fixed_partition.py`
- JSON SHA256：`a4004d3eccbe89a85399b2fceae7ff7afbe8108c5f19de9ea2c67c649581db13`
- 该工具证明 equality-quotient / inequality-self-loop 判据的必要充分性，输出可重放冲突链，并分别审计 official train、三个 seed 的真实 one-epoch internal-fit loader 流；它明确不声称覆盖全部 400 epoch 或增广分布闭包。
- 与 component-space、slicing-family 测试联跑：`17 passed`；仅有外部 PyTorch `pin_memory(device)` deprecation warning。

internal-fit 增广支持上的 arbitrary-partition 严格矛盾证书：

```bash
/home/md0/ly/BasicIRSTD/infrarenet/bin/python \
  tools/audit_augmentation_partition_closure.py \
  --output repro_runs/clean/trace_stage0_augmentation_partition_closure_v1.json
```

- 工具：`tools/audit_augmentation_partition_closure.py`（SHA256 `cdf9a2bdc23b018925a75eae753d789e6f35a79ce4491f0580ec84c685e456d3`）
- 自包含测试：`tests/test_audit_augmentation_partition_closure.py`（`5 passed`）
- JSON SHA256：`463ea4121bb3c7940ecc67944f2a27bb6a412c5ba317c5790846cbc513e9d0e3`
- 证书冻结 `split_seed=20260711,val_fraction=0.2`，所有 witness 都是 fit 成员；逐项记录 image/mask path、bytes SHA、尺寸相等检查与 `repository_resize_geometry_source=image.size`。crop 支持是符号覆盖，不以有限 seed 作为 closure 证明。

baseline provenance：

- source-commit 测试：`tests/test_canonical_mshnet_provenance.py`
- runner：`tools/run_clean_baselines.py`
- strict finalizer：`tools/finalize_trace_stage0_baseline.py`
- runtime attester：`tools/capture_trace_stage0_runtime_attestation.py`
- 运行中 attestation SHA256：`eae6f4af093893dffba4fdd5295dbb115fd93e80e9a5e6b951791ed4ecdca726`
- attestation 锁定 34 个 local-import dependency、fit/val 全部 213 对 image/mask bytes、三进程命令/环境/GPU 映射；它明确是 capture-time observation，不是 launch-time snapshot。
- strict baseline summary：`repro_runs/clean/trace_stage0_canonical_mshnet_nuaa_holdout_v1/trace_stage0_canonical_baseline_summary.json`（SHA256 `fca52b6a...68e33`）及同名 Markdown（SHA256 `f85b11c0...dbfa`）。
- 全仓回归：`407 passed`；24 条 warning 全部来自外部 PyTorch `pin_memory(device)` deprecation。回归期间发现并修复两个旧 mechanism-audit consumer 的合同漂移：它们现与 baseline finalizer 一致地 fail-closed 校验 `MSHNet-Deterministic`、canonical variant/protocol 以及 `evaluation_interval=10` 的 40 个固定评估点，不再错误要求 400 条逐 epoch metric；best checkpoint、source provenance、no-resume/no-init/DEA=0 校验均保留。

## 11. 解除阻断所需的任务级变更

当前已审计方案中没有合法的工程补丁。要继续，必须先给出一个新的、可证明满足现有约束的 set 定义，或由任务定义层明确放宽至少一项：

1. 提供一个从采集协议上保证每图至多一个、且所有标签属于冻结组件族的 benchmark；或
2. 保留真实 maximal-8CC set，但允许 approximate frontier / variational inference，不再声称 exact；或
3. 保留 exact 256² inference 与全部 binary labels，但接受 bounded-treewidth mask CRF，不再声称显式 component/root TRACE；或
4. 允许 sample-adaptive/overlapping instance protocol、额外定位路径和独立 set 评价，不再要求它等于原 binary-mask 的 maximal-8CC decomposition。

其中 2–4 都改变了当前用户约束，不能由实现者擅自采用。第 1 项或任何新协议仍须在只使用 training-side 证据的前提下预先冻结并证明完备。固定 tiling 已被上述 arbitrary-partition 定理排除，不能再作为“待调 tile size”的工程选项。继续堆叠 attention、refinement、辅助损失或后处理不能解决状态空间没有 ground-truth support 的问题。

## 12. 无结果伪造声明

本文只记录已经运行的代码测试、真实标签审计和已严格终结的 internal-val baseline。没有填写任何 TRACE 性能数值，也没有把历史 test-selected 数值包装成新结果。Stage 0 的失败是正式实验结论，不是待美化的负结果。
