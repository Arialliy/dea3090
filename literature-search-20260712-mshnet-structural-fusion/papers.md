# Literature Search: MSHNet Structural Multi-Scale Fusion

Date: 2026-07-12  
Search purpose: support an AAAI-27 structural redesign of MSHNet without module stacking  
Target venue/family: AAAI; AI/ML + computer vision  
Source-quality policy: primary proceedings/arXiv/OpenReview sources prioritized; policy-excluded sources omitted

## Summary

- Closest-work clusters: target-preserving encoders; adaptive/rank pooling; cross-level interaction; domain-prior operators; topology-aware losses; coarse-to-fine prediction.
- Crowded directions: generic attention, dense/nested U-Nets, wavelet/frequency branches, learnable pooling/importance weights, ordinary refinement, and learnable scale weighting.
- Empirically retained gap: MSHNet's max-pool transports a strongest activation without asking whether any spatial support remains after deleting that one site; later full-resolution skips can reintroduce evidence that failed this support test. Screened pooling methods optimize detail or importance, not low-component-FPPI survival under this explicit deletion counterfactual.
- Current candidate: Support-Persistence Transport (SPT), beginning with a parameter-free replacement of boundary-0 max-pool and permitted to extend only through the same survival law.
- Novelty caution: pooling alone is not an AAAI contribution. The defensible delta requires deletion-defined support persistence, measured component-FROC behavior, and one conserved transport law that also governs later reinjection; otherwise the route is rejected as another adaptive pooling variant.

## Paper Table

| # | Paper | Year | Venue/source | Type | Insight | Completeness | Numeric evidence | Overall | Structural lesson / risk |
|---:|---|---:|---|---|---:|---:|---:|---|---|
| 1 | [Infrared Small Target Detection with Scale and Location Sensitivity](https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Infrared_Small_Target_Detection_with_Scale_and_Location_Sensitivity_CVPR_2024_paper.html) | 2024 | CVPR | pure method | 4 | 4 | 4 | A | MSHNet baseline; simple multi-scale head plus scale/location-sensitive training. It does not make scale contributions identifiable. |
| 2 | [Pinwheel-shaped Convolution and Scale-based Dynamic Loss](https://ojs.aaai.org/index.php/AAAI/article/view/32996) | 2025 | AAAI | method + benchmark | 4 | 4 | 4 | A | Replaces low-level convolution according to Gaussian-like target geometry. Copying lower-layer spatial priors would collide directly. |
| 3 | [TCI-Former](https://ojs.aaai.org/index.php/AAAI/article/view/27882) | 2024 | AAAI | pure method | 5 | 4 | 4 | A | Derives the architecture from one feature-evolution law instead of collecting modules; strong design exemplar, not a component source. |
| 4 | [IRMamba](https://ojs.aaai.org/index.php/AAAI/article/view/33085) | 2025 | AAAI | pure method | 4 | 4 | 4 | A | Pixel-difference state evolution plus layer restoration; shows that “Mamba + module” is insufficient without a task-specific state equation. |
| 5 | [DEFANet](https://ojs.aaai.org/index.php/AAAI/article/view/37368) | 2026 | AAAI | pure method | 4 | 4 | 4 | A | Edge/target dual-path frequency interaction. Frequency and boundary branches are now a high-risk crowded route. |
| 6 | [Target-Aware Invertible Encoder with Reconstruction Guidance](https://openaccess.thecvf.com/content/CVPR2026/html/Yan_Target-Aware_Invertible_Encoder_with_Reconstruction_Guidance_for_Infrared_Small_Target_CVPR_2026_paper.html) | 2026 | CVPR | pure method | 5 | 5 | 4 | A | Makes information loss under downsampling explicitly optimizable. An invertible/wavelet encoder redesign would be too close. |
| 7 | [Seeing Through the Noise: NS-FPN](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.html) | 2026 | CVPR | pure method | 5 | 5 | 5 | A | Reframes enhancement as noise suppression and rewrites lateral fusion/sampling. Generic frequency purification is occupied. |
| 8 | [ISNet: Shape Matters](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_ISNet_Shape_Matters_for_Infrared_Small_Target_Detection_CVPR_2022_paper.pdf) | 2022 | CVPR | method + benchmark | 4 | 5 | 4 | A | Couples an ODE-inspired edge block with cross-level aggregation. Edge refinement alone is not a new structural claim. |
| 9 | [SCTransNet](https://arxiv.org/abs/2401.15583) | 2024 | arXiv | pure method | 4 | 4 | 4 | B | Cross-transformer interactions across all encoder/decoder levels; establishes that cross-level attention is crowded and expensive. |
| 10 | [DNANet](https://arxiv.org/abs/2106.00487) | 2021/2023 | arXiv / TIP work | method + benchmark | 4 | 5 | 5 | A | Dense nested interaction preserves targets through depth; simply adding cross-layer paths is no longer sufficient novelty. |
| 11 | [UIU-Net](https://arxiv.org/abs/2212.00968) | 2022 | arXiv / TIP work | pure method | 4 | 4 | 4 | A | Nested U-structures and resolution-maintenance deep supervision; warns against another nested decoder. |
| 12 | [One-Stage Cascade Refinement Networks (OSCAR)](https://arxiv.org/abs/2212.08472) | 2022 | arXiv | method + benchmark | 4 | 4 | 4 | Risk | High-level heads guide low-level refinement. A generic coarse-to-fine evidence chain would not be novel. |
| 13 | [Laplacian Pyramid Reconstruction and Refinement](https://arxiv.org/abs/1605.02264) | 2016 | arXiv / CVPR-era work | pure method | 4 | 4 | 4 | Risk | Successive multi-resolution boundary refinement is established prior art. OSO must emphasize source identifiability, not refinement. |
| 14 | [Residual Pyramid Learning](https://arxiv.org/abs/1903.09746) | 2019 | arXiv | pure method | 3 | 3 | 3 | Risk | Explicitly learns main segmentation and residual labels across levels. This rules out claiming novelty for residual-mask pyramids alone. |
| 15 | [LoMix](https://openreview.net/forum?id=87c2JwNJa0) | 2025 | NeurIPS / OpenReview | pure method | 4 | 4 | 4 | Risk | Learns combinations of multi-scale logits during training. OSO must differ as fixed complementary ownership, not better learnable mixing. |
| 16 | [On Single Source Robustness in Deep Fusion Models](https://papers.neurips.cc/paper/8728-on-single-source-robustness-in-deep-fusion-models.pdf) | 2019 | NeurIPS | pure method | 4 | 4 | 4 | Risk | Establishes that linear fusion is not automatically robust to one corrupted source. DSF must not claim invention of source robustness. |

Scores are qualitative screening judgments (1–5), not paper acceptance scores or reported benchmark values.

## Closest-Work Clusters

### 1. Target-preserving encoders

- Representatives: PConv, InvDet, DNANet, UIU-Net.
- Already covered: preserving weak targets through low layers, downsampling, dense skips, or invertible latents.
- Remaining gap: native output sources remain overlapping and non-identifiable.
- Decision: freeze MSHNet's encoder for the current route; changing it would move into a crowded problem and dilute the scale-ownership claim.

### 2. Domain-prior and frequency structures

- Representatives: TCI-Former, IRMamba, DEFANet, ISNet, NS-FPN.
- Already covered: thermal diffusion, pixel differences, edge transitions, wavelet/frequency purification, and structured sampling.
- Remaining gap: these improve representations but do not impose a mathematical partition on scale evidence at the prediction layer.
- Decision: do not add a wavelet, edge, Mamba, attention, or dynamic-sampling side module.

### 3. Coarse-to-fine and residual pyramids

- Representatives: OSCAR, Laplacian Pyramid Refinement, Residual Pyramid Learning.
- Already covered: high-level proposals guiding low-level heads, progressive boundary reconstruction, and residual label prediction.
- Remaining gap: no screened method ties each pre-existing native scale source to a mutually orthogonal output subspace and audits exact deletion responsibility.
- Decision: avoid “coarse-to-fine residual refinement” as the headline. Use exact complementary-subspace ownership as the mechanism.

### 4. Multi-scale logit mixing

- Representatives: MSHNet and LoMix.
- Already covered: side-output supervision, affine fusion, and learnable combinations of multi-scale logits.
- Remaining gap: overlapping sources are not identifiable; one output component can be redundantly expressed at several scales.
- Decision: replace unconstrained overlap with fixed structural ownership, then test whether the constraint improves false-alarm/IoU trade-offs.

## Opportunity Map

| Cluster | Status | Open gap | Candidate direction | Evidence needed | Main risk |
|---|---|---|---|---|---|
| Encoder preservation | crowded but open | output ownership untouched | freeze encoder; isolate fusion problem | identity audit | reviewers may ask why not improve representation |
| Frequency/edge priors | covered central mechanisms | no source identifiability | explicitly exclude from final model | closest-work table | OSO may be mistaken for a frequency module |
| Coarse-to-fine refinement | covered central claim | sources still overlap | complementary projections rather than residual cascade | projector identities + OSCAR/RPNet comparison | Laplacian-pyramid relabeling |
| Logit mixing | mechanism gap | arbitrary overlap and credit ambiguity | Orthogonal Scale Ownership | pairwise orthogonality, exact reconstruction, paired performance | constraint may reduce useful cross-scale cooperation |

## Positioning Cautions

- Do not claim invention of Laplacian pyramids, residual refinement, deletion attribution, or frequency decomposition.
- Do not describe OSO as attention, gating, dynamic fusion, or a learnable scale selector.
- The architectural novelty claim requires both exact complementary subspaces and source alignment. Removing either reduces the method to standard pyramid filtering.
- Positive NUAA/NUDT/IRSTD-1K results are still required; algebraic elegance alone is not performance evidence.
- For DSF, cite single-source robustness as broad prior art and limit novelty to exact internal scale deletion plus dense normalized worst-coalition fusion.

## 2026-07-13 Update: Front-to-Back Support-Persistence Route

The OSO/DSF/DCDF family was rejected empirically and is no longer the final
direction.  The official seed-20260713 baseline trace instead localizes two
native failures: matched-vs-false feature-energy AUC rises to 0.737 at
encoder2, collapses to 0.562 in the middle, recovers to 0.842 at decoder1,
and falls to 0.710 after decoder0.  A frozen intervention at the first pool
boundary also improves IoU from 0.72803 to 0.72851 and FA from 22.10 to 21.53
without changing PD when only 5% of the strongest-site-exclusive evidence is
removed.  This motivates replacing native resampling, starting at boundary 0,
before considering any output-side mechanism.

Additional close work screened:

| Paper | Boundary established | Consequence for this project |
|---|---|---|
| [Detail-Preserving Pooling in Deep Networks](https://openaccess.thecvf.com/content_cvpr_2018/html/Saeedan_Detail-Preserving_Pooling_in_CVPR_2018_paper.html), CVPR 2018 | Adaptive/learnable pooling that preserves local detail is established. | Do not claim novelty for replacing max pooling or learning a max/average trade-off. |
| [Ordinal Pooling](https://arxiv.org/abs/2109.01561), 2021 | Learned weights over ordered activations, spanning average to max behavior, are established. | A learned top-1/top-2 mixture alone is insufficient for AAAI. |
| [Topology-Preserving Deep Image Segmentation](https://arxiv.org/abs/1906.05404), 2019 | Persistent-homology losses can constrain connected-component topology. | Do not claim invention of topology-aware supervision; our current operator is architectural and does not compute PH. |
| [Scale-Free Image Keypoints Using Differentiable Persistent Homology](https://proceedings.mlr.press/v235/barbarani24a.html), ICML 2024 | Persistence can define scale-free feature salience. | “Persistence” terminology requires an exact operational definition and must not imply PH unless PH is actually computed. |
| [Improving Topology Accuracy by Penalizing Neighbor Pixels](https://openaccess.thecvf.com/content/CVPR2026/html/Valverde_Towards_High-Quality_Image_Segmentation_Improving_Topology_Accuracy_by_Penalizing_Neighbor_CVPR_2026_paper.html), CVPR 2026 | Neighbor-supported prediction can reduce isolated spurious components through a loss. | The novelty boundary is a task-derived resampling law and scale transport, not generic neighbor penalization. |
| [LIP: Local Importance-Based Pooling](https://openaccess.thecvf.com/content_ICCV_2019/html/Gao_LIP_Local_Importance-Based_Pooling_ICCV_2019_paper.html), ICCV 2019 | Learned local importance weighting unifies and improves standard downsampling, including small-object detection. | “Content-adaptive/local-importance pooling” is not a novelty claim; SPT must retain its deletion-counterfactual and component-budget distinction. |
| [AdaPool](https://arxiv.org/abs/2111.00772), TIP 2023 | Regional fusion of exponential-max and Dice-Sørensen kernels, with a paired unpool operator, is established. | A bidirectional adaptive pooling/unpooling story is occupied; later decoder work cannot simply add an AdaUnPool analogue. |
| [Content-Adaptive Downsampling](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Hesse_Content-Adaptive_Downsampling_in_Convolutional_Neural_Networks_CVPRW_2023_paper.html), CVPRW 2023 | Spatially varying resolution allocation is established. | SPT changes evidence survival inside a fixed-resolution transition; do not claim invention of adaptive downsampling. |

Current structural hypothesis: **Support-Persistence Transport (SPT)**.  In a
2x2 feature cell, let `m1` be the factual maximum and `m2` the maximum after
deleting the strongest site.  Their difference is strongest-site-exclusive
evidence.  A channel-consensus survival statistic controls how much of that
exclusive evidence crosses a resolution boundary.  The first experiment
replaces only MSHNet's first max-pool; no auxiliary branch, attention block,
frequency path, edge head, or new loss is present.

Strict idea-review status before training: **conditional reject as a complete
AAAI method**.  Pooling alone is crowded (DPP, rank/ordinal pooling), and the
current evidence covers one seed/dataset/checkpoint.  It is allowed as the
first causal structural step only.  Promotion requires (i) trained gains in
IoU and component-FROC, (ii) evidence that channel-consensus conditioning—not
just top-2 pooling—causes the gain, and (iii) a single transport law that also
explains the later decoder0 reinjection failure without becoming a stack.

The expanded pooling search strengthens this rejection boundary: LIP,
AdaPool, DPP, ordinal pooling, and content-adaptive downsampling already cover
learned importance, local detail, ordered values, bidirectional pooling, and
spatially varying resolution.  No screened primary source was found that uses
the maximum after deleting the strongest spatial site together with
cross-channel site agreement to target component-FROC, but phrase-level
novelty is insufficient.  A publishable claim still requires the same
support-persistence conservation law to govern later evidence reinjection.
