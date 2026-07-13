# MSHNet Structural Counterfactual Research Workbench

This repository is a reproducible workbench for diagnosing and redesigning
MSHNet for infrared small target detection. It contains the canonical baseline,
strict paired-training infrastructure, component-level FROC evaluation, and a
sequence of counterfactual scale/fusion/resampling interventions. Historical
candidates are retained as auditable negative evidence instead of being
presented as validated improvements.

> **Authoritative status (2026-07-13):** SDRR, OSO, DSF, DCDF, and CCFD are
> mechanism studies or rejected comparators. CCFD reduced false-alarm area but
> did not beat the independently selected canonical best-IoU checkpoint. SPT0,
> a parameter-free support-persistence replacement at the first 2×2 max-pooling
> boundary, has also completed its 400-epoch paired gate: it lowered FA but
> lost IoU and PD and is therefore `DESIGN FAIL`. No structural method currently
> has `DESIGN PASS` status.

The project is based on the CVPR 2024 MSHNet implementation:

> Infrared Small Target Detection with Scale and Location Sensitivity

## Overview

![Overview](assert/overview.png)

## Current Gate Result

The fixed NUAA-SIRST official train/test protocol compares each run at its own
best-IoU checkpoint:

| Variant | Best epoch | IoU | PD | FA/Mpixel |
| --- | ---: | ---: | ---: | ---: |
| Deterministic MSHNet | 379 | **0.728026** | **0.946768** | 22.1039 |
| CCFD | 379 | 0.725052 | 0.950570 | 18.1822 |
| SPT0 | 399 | 0.716904 | 0.931559 | **11.9076** |

CCFD and SPT0 improve false-alarm area but fail the primary IoU gate; SPT0 also
loses PD. Their raw-logit component FROC behavior is mixed rather than
consistently better across budgets, so both are retained as negative structural
comparators, not proposed models.

## Current Research Tracks

- **Latest completed candidate — SPT0:** replaces only the first native max-pooling
  boundary with a parameter-free support-persistence law derived from
  strongest-site deletion and cross-channel spatial agreement; the full gate
  failed despite a large FA reduction.
- **Baseline and protocol integrity:** physically isolated official-forward and
  deterministic-backward MSHNet variants; dataset-pair, manifest-hash,
  state-dict, forward, backward, and shared-prefix audits.
- **Independent checkpoint selection:** each run selects its own best-IoU point
  from the same fixed evaluation schedule; same-epoch snapshots are diagnostic
  only and cannot decide a method gate.
- **Component-aware evaluation:** raw-logit component FROC, FP components per
  image, target-instance detection probability, and pixel IoU/PD/FA.
- **Historical structural comparators:** SDRR/RDR, OSO, DSF, DCDF, and CCFD are
  retained with their controls and failure evidence; none is the current
  accepted model.
- **Earlier supervision controls:** DEA-lite, RODS, TCDS/TFDS, task-gradient,
  scale-coalition, subset, delayed-scale, homotopy, and null-target variants.
- **Cross-backbone infrastructure:** additive-fusion audit and a paired UIU-Net
  runner for future transfer checks; smoke completion is not performance proof.

The design history, negative-result ledger, and data protocol are documented in
[`MSHNet_AAAI27_SDRR_model_design.md`](MSHNet_AAAI27_SDRR_model_design.md).
The completed SPT0 result above supersedes that document's earlier
run-in-progress sentence in section 33.4.
The detailed SDRR code/submission review is retained in
[`MSHNet_SDRR_第三轮代码与投稿复核.md`](MSHNet_SDRR_第三轮代码与投稿复核.md).
The preceding TCDS/TFDS review is retained in
[`MSHNet_TCDS_TFDS_第二轮阶段复核.md`](MSHNet_TCDS_TFDS_第二轮阶段复核.md).
The earlier scale-ownership plan is retained in
[`Server_B_Backup_Scale_Ownership.md`](Server_B_Backup_Scale_Ownership.md).
A reproducible related-work snapshot is available under
[`literature-search-20260712-mshnet-structural-fusion/`](literature-search-20260712-mshnet-structural-fusion/).

## Repository Layout

```text
.
├── main.py
├── model/
│   ├── MSHNet.py
│   ├── baselines/
│   │   ├── mshnet_official.py
│   │   └── mshnet_deterministic.py
│   ├── sdr_mshnet.py
│   ├── support_persistence_transport.py
│   ├── additive_fusion.py
│   ├── orthogonal_scale_ownership.py
│   ├── deletion_stable_fusion.py
│   ├── decision_conditional_deletion_fusion.py
│   ├── counterfactual_conflict_diffusion.py
│   ├── loss.py
│   ├── partial_sls_loss.py
│   ├── resolution_owned_supervision.py
│   ├── task_consistent_supervision.py
│   ├── task_gradient_supervision.py
│   ├── scale_coalition_supervision.py
│   ├── counterfactual_responsibility.py
│   └── measure_conditioned_sls.py
├── tools/
│   ├── audit_rods_assignment.py
│   ├── audit_task_consistent_projection.py
│   ├── audit_counterfactual_responsibility.py
│   ├── audit_sdrr_deletion_stability.py
│   ├── audit_sdrr_optimizer_influence.py
│   ├── audit_dataset_pair_integrity.py
│   ├── audit_mshnet_stage_component_trace.py
│   ├── audit_pool_counterfactual.py
│   ├── audit_spt_mechanism.py
│   ├── branch_sdrr_shared_prefix.py
│   ├── compare_independent_best.py
│   ├── evaluate_component_froc.py
│   ├── summarize_sdrr_formal.py
│   └── optimizer_counterfactual.py
├── tests/
│   └── test_*.py
├── utils/
│   ├── data.py
│   ├── metric.py
│   ├── component_froc.py
│   └── order_statistic_pool.py
├── scripts/official/  # frozen experiment launchers
├── literature-search-20260712-mshnet-structural-fusion/
├── assert/
│   ├── overview.png
│   └── visual_result.png
├── datasets/      # local only, ignored by Git
├── weight/        # local only, ignored by Git
└── repro_runs/    # local only, ignored by Git
```

## Deep-Supervision Modes

Select the training topology with `--deep-supervision`. Important families
include:

| Family | Modes | Purpose |
| --- | --- | --- |
| Canonical controls | `legacy_exact`, `legacy_rescaled`, `final_only`, `side_no_location` | Establish the baseline and loss-budget effects |
| Scale ownership | `rods_interval`, `rods_hard`, `rods_random`, `rods_area_only` | Compare ownership rules and matched controls |
| TCDS projection | `tfds_projection`, `tfds_projection_active_renorm` | Train from scene-level projection recovery |
| Gradient constraint | `tgds_halfspace` | Project conflicting auxiliary gradients |
| Native coalitions | `cscs_leave_one_out`, `sfds_filtration`, `asfs_anchor_filtration`, `rdfs_continuation` | Test exact scale-deletion and filtration hypotheses |
| SDRR/RDR | `crs_flip_suppression`, `crs_responsibility_density`, `crs_responsibility_routing` | Test pivotal-event suppression, density risk, and responsibility-conserving routing |
| Attribution controls | `crs_matched_random`, `crs_same_pixel_random_scale`, `crs_magnitude_nonpivotal`, `crs_all_safe_fp`, `crs_same_pixel_fused` | Separate pivotality from event budget, scale identity, magnitude, generic hard negatives, and fused-logit suppression |
| Null-target handling | `mcsls_null_safe`, `zmsls_null_abstain` | Separate non-null SLS from zero-mass behavior |

Additional scale-subset, delayed-scale, and homotopy controls are listed by
`python main.py --help`.

## Physical MSHNet Variants

Select a physical architecture with `--mshnet-variant`:

| Variant | Role and current status |
| --- | --- |
| `workbench` | Historical checkpoint-compatible implementation with dormant DEA-era state; not the clean paper baseline |
| `official` | Recovered canonical forward for source-faithful checks |
| `deterministic` | Parameter-identical paper baseline with deterministic max-reduction backward |
| `sdr` | Canonical deployment graph with training-only SDRR state; historical mechanism candidate |
| `oso`, `dsf`, `dcdf`, `ccfd` | Auditable structural experiments retained in the negative/comparator ledger |
| `spt0` | Completed parameter-free stage-0 support-persistence experiment; `DESIGN FAIL`, retained as a comparator |

Presence in the CLI means a variant is reproducible, not that it passed the
scientific or submission gate.

## Training-Only Fusion Regularizers

Physical architecture and training-only regularization are separate axes. Use
`--fusion-regularizer` for the paper-facing names:

| Value | Meaning |
| --- | --- |
| `none` | Canonical objective for the chosen physical variant |
| `sdrr`, `rdr`, `rcr` | Deletion-pivotal suppression, density risk, or responsibility-conserving routing |
| `m1_all_safe_fp`, `m2_pivotal_pixel` | Generic safe-background and fused-logit pivotal controls |
| `m3_magnitude_nonpivotal`, `m4_random_scale`, `scale_budget_random` | Attribution controls for magnitude, scale identity, and event budget |

The historical `crs_*` deep-supervision names remain compatibility aliases.

## Dataset

Datasets live under `datasets/` locally and are intentionally ignored by Git.
The current paper-facing protocol uses only the published train/test manifests:

| Dataset | Train manifest | Train images | Test manifest | Test images |
| --- | --- | ---: | --- | ---: |
| NUAA-SIRST | `img_idx/train_NUAA-SIRST.txt` | 213 | `img_idx/test_NUAA-SIRST.txt` | 214 |
| NUDT-SIRST | `img_idx/train_NUDT-SIRST.txt` | 663 | `img_idx/test_NUDT-SIRST.txt` | 664 |
| IRSTD-1K | `img_idx/train_IRSTD-1K.txt` | 800 | `img_idx/test_IRSTD-1K.txt` | 201 |

Use `--evaluation-protocol official_train_test` with explicit train and test
manifests. Internal-holdout runs remain useful for diagnostics but cannot enter
the final comparison table. Dataset-pair integrity and manifest hashes can be
checked with `tools/audit_dataset_pair_integrity.py`.

## Training

Single GPU:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --batch-size 4 \
  --epochs 400 \
  --lr 0.05 \
  --mode train
```

Multi-GPU:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --batch-size 16 \
  --epochs 400 \
  --lr 0.05 \
  --mode train \
  --multi-gpus true \
  --gpu-ids 0,1,2,3
```

Reproducible NUAA-SIRST canonical baseline:

```bash
python main.py \
  --dataset-dir datasets/NUAA-SIRST \
  --train-split-file img_idx/train_NUAA-SIRST.txt \
  --test-split-file img_idx/test_NUAA-SIRST.txt \
  --evaluation-protocol official_train_test \
  --evaluation-interval 10 \
  --mode train \
  --model-type mshnet \
  --mshnet-variant deterministic \
  --deep-supervision legacy_exact \
  --epochs 400 \
  --seed 20260713 \
  --deterministic true \
  --run-label official-nuaa-baseline-seed-20260713
```

To reproduce a deep-supervision control, change only `--deep-supervision` and its
mode-specific arguments while keeping the dataset split, seed, optimizer, and
training budget matched. For example, the historical SDRR topology uses:

```bash
--deep-supervision crs_flip_suppression \
--crs-lambda 0.05 \
--crs-start-epoch 250 \
--crs-ramp-epochs 50 \
--crs-safe-kernel 15 \
--sdrr-normalization event
```

Use `--mshnet-variant official` for the recovered canonical forward, or
`--mshnet-variant deterministic` for the parameter-identical variant with a
deterministic max-reduction backward. The historical `workbench` default is
kept only for checkpoint compatibility and contains dormant DEA-era
parameters; it must not be presented as a physically isolated official
baseline.

The SDRR normalization controls are `event`, `safe_density`, and
`unique_pixel`. The `crs_magnitude_nonpivotal` and
`crs_same_pixel_random_scale` modes test attribution semantics. Despite its
legacy name, `crs_matched_random` matches only the image-by-scale event budget
and is not a complete matched-random attribution control.

DEA-lite loss weights can be adjusted with:

```bash
--dea-lambda-single 0.10 \
--dea-lambda-dec 0.05 \
--dea-lambda-empty 0.01 \
--dea-tau 0.3 \
--dea-ramp-epochs 20
```

## Resume Training

Resume from the latest checkpoint under `weight/MSHNet-*/checkpoint.pkl`:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --mode train \
  --if-checkpoint true
```

Resume from a specific checkpoint folder and reset optimizer state:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --mode train \
  --if-checkpoint true \
  --checkpoint-dir weight/MSHNet-YYYY-MM-DD-HH-MM-SS \
  --reset-optimizer true
```

## Testing

Run the unit and invariant tests with:

```bash
python -m pytest -q
```

The test suite covers clean-baseline identity, deterministic-backward behavior,
direct zero-channel deletion, decision-margin stability, responsibility
normalization, matched-control gradient support, shared-prefix branching,
exact canonical gradients when no responsibility event exists, formal summary
fail-closed checks, projection identities, coalition reconstruction, one-step
optimizer counterfactuals, dataset splits, and metrics.

Key SDRR audit entry points are:

```bash
python tools/audit_sdrr_deletion_stability.py --help
python tools/audit_sdrr_optimizer_influence.py --help
python tools/branch_sdrr_shared_prefix.py --help
python tools/summarize_sdrr_formal.py --help
```

Current structural and evaluation audits include:

```bash
python tools/audit_dataset_pair_integrity.py --help
python tools/audit_mshnet_stage_component_trace.py --help
python tools/audit_pool_counterfactual.py --help
python tools/audit_spt_mechanism.py --help
python tools/evaluate_component_froc.py --help
python tools/compare_independent_best.py --help
```

Evaluate a trained checkpoint with:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --batch-size 4 \
  --mode test \
  --weight-path weight/IRSTD-1k_weight.tar
```

## Outputs

Training writes checkpoints and logs to the selected `--run-dir` (formal runs
use `repro_runs/<run-name>/`) or the legacy `weight/MSHNet-<timestamp>/` path:

- `checkpoint.pkl`
- `checkpoint_best_iou.pkl`
- `checkpoint_pd_fa_best.pkl`
- `weight.pkl`
- `metric.log`
- `epoch_metric.log`
- `run_config.json`
- split snapshots and hashes
- optional `best_vs_baseline.json`, component-FROC, and mechanism-audit JSON
- optional `dea_debug/*.pt`

These outputs are ignored by Git. Keep datasets, trained weights, and
reproduction logs outside commits unless they are intentionally packaged with
`tools/package_sdr_mshnet_run.py` or published through external storage.

## Visual Results

![Visual Results](assert/visual_result.png)

## Citation

If this code is useful for your research, please cite the original MSHNet paper:

```bibtex
@inproceedings{liu2024infrared,
  title={Infrared Small Target Detection with Scale and Location Sensitivity},
  author={Liu, Qiankun and Liu, Rui and Zheng, Bolun and Wang, Hongkui and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition},
  year={2024}
}
```
