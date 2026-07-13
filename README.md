# MSHNet Structural Counterfactual Research Workbench

This repository is a reproducible workbench for diagnosing and redesigning
MSHNet for infrared small target detection. It contains the canonical baseline,
strict paired-training infrastructure, component-level FROC evaluation, and a
sequence of counterfactual scale/fusion/resampling interventions. Historical
candidates are retained as auditable negative evidence instead of being
presented as validated improvements.

> **Authoritative status (2026-07-13):** TRACE Stage 0 is formally `NO-GO`:
> the real multi-component/augmented labels do not belong to the proposed exact
> component state family, while a semantically complete explicit frontier state
> is not runnable at 256×256 under the frozen constraints. JIP is retained only
> as a historical, code-level candidate. The newer CSL/JCSL/JCPT/RTFE/SNLT/BCSF
> files are isolated research prototypes and audits, not validated models or
> registered paper variants. No method currently has `DESIGN PASS` status.

The project is based on the CVPR 2024 MSHNet implementation:

> Infrared Small Target Detection with Scale and Location Sensitivity

## Overview

![Overview](assert/overview.png)

## Current Stage-0 Decision

TRACE cannot proceed to solver implementation without changing the task-level
contract. The audited labels contain multiple connected components, empty crops,
multi-run components, holes, and augmentation-induced splits. The predefined
single-component/single-row-run family therefore cannot define an exact NLL for
the actual supervision stream. Fixed non-overlapping partitions are also ruled
out by the audited augmentation support. Approximate inference, dense-mask CRFs,
or adaptive/overlapping instance protocols would be new task assumptions and
must not be introduced silently.

The clean deterministic baseline contract has completed three fresh 400-epoch
NUAA internal-holdout runs:

| Seed | Internal-val best epoch | IoU | PD | FA/Mpixel |
| ---: | ---: | ---: | ---: | ---: |
| 20260711 | 359 | 0.680293 | 0.981481 | 61.745 |
| 20260712 | 129 | 0.674387 | 0.944444 | 39.744 |
| 20260713 | 389 | 0.707264 | 0.981481 | 50.744 |
| Mean ± sample SD | 292.33 ± 142.24 | 0.687315 ± 0.017527 | 0.969136 ± 0.021383 | 50.744 ± 11.001 |

These are development-validation results, not official-test results and not a
paper main table.

## Historical Structural Comparator Results

The earlier NUAA-SIRST official train/test workbench compared each run at its
own best-IoU checkpoint:

| Variant | Best epoch | IoU | PD | FA/Mpixel |
| --- | ---: | ---: | ---: | ---: |
| Deterministic MSHNet | 379 | **0.728026** | **0.946768** | 22.1039 |
| CCFD | 379 | 0.725052 | 0.950570 | 18.1822 |
| SPT0 | 399 | 0.716904 | 0.931559 | **11.9076** |

CCFD and SPT0 improve false-alarm area but fail the primary IoU gate; SPT0 also
loses PD. Their raw-logit component FROC behavior is mixed rather than
consistently better across budgets, so both are retained as negative structural
comparators, not proposed models. These test-selected historical results are
mechanism evidence only under the current leakage-safe development protocol.

## Current Research Tracks

- **TRACE Stage-0 contract audit:** exact component-state semantics, real-label
  support, augmentation closure, fixed-partition feasibility, and explicit
  frontier-state complexity are audited before any solver implementation.
- **Clean baseline provenance:** the official-forward and parameter-identical
  deterministic-backward variants are checked against the historical source,
  runtime environment, data bytes, split hashes, checkpoints, and metric logs.
- **Historical JIP candidate:** Jackknife Influence Pooling remains a documented
  code-level hypothesis only; it has no empirical design pass and is not the
  current frozen model.
- **Isolated front-end prototypes:** CSL, JCSL, JCPT, RTFE, SNLT, and BCSF test
  sufficient-statistic, geometric, or filtration ideas without being silently
  promoted into the main CLI or paper claim.
- **Fail-closed evaluation:** independent-best comparison, component FROC,
  runtime attestation, baseline finalization, and mechanism summaries reject
  incomplete schedules, provenance drift, leakage, or incompatible metrics.
- **Historical structural comparators:** SDRR/RDR, OSO, DSF, DCDF, CCFD, and
  SPT0 are retained with their controls and negative evidence.
- **Cross-backbone infrastructure:** additive-fusion and paired UIU-Net tools
  remain transfer infrastructure; smoke completion is not performance proof.

The current TRACE task-contract decision and reproduction evidence are in
[`TRACE_MSHNet_STAGE0_audit.md`](TRACE_MSHNet_STAGE0_audit.md).
The historical JIP candidate and its frozen claim boundary are in
[`MSHNet_AAAI27_JIP_model_design.md`](MSHNet_AAAI27_JIP_model_design.md).
The broader structural design history and negative-result ledger are in
[`MSHNet_AAAI27_SDRR_model_design.md`](MSHNet_AAAI27_SDRR_model_design.md).
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
│   ├── baseline_embedded_resnet.py
│   ├── counterfactual_sufficient_lift.py
│   ├── jet_coherent_sufficient_lift.py
│   ├── jet_coherent_potential_transport.py
│   ├── relative_trace_free_energy_lift.py
│   ├── scale_normalized_tangent_lift.py
│   ├── birth_constrained_scale_filtration.py
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
│   ├── audit_augmentation_partition_closure.py
│   ├── audit_csl_stage0.py
│   ├── audit_front_evidence_screen.py
│   ├── audit_trace_component_space.py
│   ├── audit_trace_fixed_partition.py
│   ├── audit_trace_slicing_families.py
│   ├── capture_trace_stage0_runtime_attestation.py
│   ├── finalize_trace_stage0_baseline.py
│   ├── branch_sdrr_shared_prefix.py
│   ├── compare_independent_best.py
│   ├── evaluate_component_froc.py
│   ├── summarize_sdrr_formal.py
│   ├── optimizer_counterfactual.py
│   └── run_jcpt_gain_gate.py
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
JIP, CSL, JCSL, JCPT, RTFE, SNLT, and BCSF are not accepted
`--mshnet-variant` values in `main.py`; their standalone files and audit
tools must not be mistaken for integrated or empirically accepted models.

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
Published train/test manifests are:

| Dataset | Train manifest | Train images | Test manifest | Test images |
| --- | --- | ---: | --- | ---: |
| NUAA-SIRST | `img_idx/train_NUAA-SIRST.txt` | 213 | `img_idx/test_NUAA-SIRST.txt` | 214 |
| NUDT-SIRST | `img_idx/train_NUDT-SIRST.txt` | 663 | `img_idx/test_NUDT-SIRST.txt` | 664 |
| IRSTD-1K | `img_idx/train_IRSTD-1K.txt` | 800 | `img_idx/test_IRSTD-1K.txt` | 201 |

The current Stage-0 development contract splits the official training manifest
80/20 with `split_seed=20260711`, uses only internal validation for checkpoint
selection, and sets `--evaluation-protocol internal_holdout`. The baseline
runner reads official-test IDs only for hash/overlap checks and does not iterate
test images or masks. A separate task-definition audit has read test masks for
descriptive statistics, so the repository must not claim that official test is
globally sealed. The historical `official_train_test` mode remains available
only for reproducing earlier comparator runs. Dataset-pair integrity and hashes
can be checked with `tools/audit_dataset_pair_integrity.py`.

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

Reproducible NUAA-SIRST internal-holdout canonical baseline:

```bash
python main.py \
  --dataset-dir datasets/NUAA-SIRST \
  --train-split-file img_idx/train_NUAA-SIRST.txt \
  --test-split-file img_idx/test_NUAA-SIRST.txt \
  --evaluation-protocol internal_holdout \
  --val-fraction 0.2 \
  --split-seed 20260711 \
  --evaluation-interval 10 \
  --num-workers 0 \
  --mode train \
  --model-type mshnet \
  --mshnet-variant deterministic \
  --deep-supervision legacy_exact \
  --fusion-regularizer none \
  --epochs 400 \
  --seed 20260713 \
  --deterministic true \
  --run-label trace-stage0-baseline-seed-20260713
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

The test suite covers historical-source provenance, clean-baseline identity,
deterministic backward, runtime/data attestation, TRACE component-space and
augmentation-closure proofs, fixed-partition and slicing audits, fail-closed
baseline finalization, independent checkpoint comparison, component FROC,
prototype operator/model invariants, historical SDRR controls, dataset splits,
and metrics.

Current TRACE contract and baseline audits:

```bash
python tools/audit_dataset_pair_integrity.py --help
python tools/audit_augmentation_partition_closure.py --help
python tools/audit_trace_component_space.py --help
python tools/audit_trace_fixed_partition.py --help
python tools/audit_trace_slicing_families.py --help
python tools/capture_trace_stage0_runtime_attestation.py --help
python tools/finalize_trace_stage0_baseline.py --help
```

Current standalone prototype and evaluation audits:

```bash
python tools/audit_csl_stage0.py --help
python tools/audit_front_evidence_screen.py --help
python tools/run_jcpt_gain_gate.py --help
python tools/evaluate_component_froc.py --help
python tools/compare_independent_best.py --help
```

Historical SDRR audit entry points:

```bash
python tools/audit_sdrr_deletion_stability.py --help
python tools/audit_sdrr_optimizer_influence.py --help
python tools/branch_sdrr_shared_prefix.py --help
python tools/summarize_sdrr_formal.py --help
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
