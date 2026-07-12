# MSHNet–SDRR Scale-Responsibility Research

This repository studies Scale-Deletion Responsibility Refinement (SDRR) and
controlled multi-scale supervision for infrared small target detection. SDRR
uses MSHNet's native linear fusion to identify safe-background decisions that
flip when one scale contribution is deleted, then regularizes only the pivotal
positive contribution. It adds no trainable inference module and does not
change the deployment graph.

> **Research status (2026-07-12):** the engineering paths and unit-level
> invariants are implemented. Baseline-identity and deletion-stability audits
> have been added, while normalization and attribution controls still need
> formal training results. SDRR remains **NO-GO pending** and must currently be
> described as sparse contribution regularization selected by scale-deletion
> pivotality, not as validated causal or responsibility learning.

The project is based on the CVPR 2024 MSHNet implementation:

> Infrared Small Target Detection with Scale and Location Sensitivity

## Overview

![Overview](assert/overview.png)

## Current Research Tracks

- **SDRR:** exact fixed-logit deletion at MSHNet's final linear fusion,
  safe-background decision-flip selection, late-ramped regularization, and
  event/anatomy logging.
- **Attribution controls:** event, safe-density, and unique-pixel normalization;
  magnitude-matched non-pivotal and same-pixel random-scale controls; and an
  explicitly unmatched scale-budget random control.
- **Baseline isolation:** physically separate official-forward and
  parameter-identical deterministic-backward MSHNet variants, plus strict
  state-dict, parameter-count, output, and gradient identity tests.
- **Deletion stability:** compare algebraic `z - c_i` against direct zero-channel
  fusion across decision margins and report event-set agreement.
- **Shared-prefix experiments:** branch SDRR and controls from the same completed
  canonical prefix with auditable checkpoint and metadata hashes.
- **Earlier supervision studies:** canonical, final-only, RODS, TCDS/TFDS,
  task-gradient, coalition, scale-subset, delayed-scale, and homotopy controls.
- **TCDS/TFDS projection:** scene-level projection recovery, partial-label
  supervision, active-sample accounting, and matched graph audits.
- **Task-gradient supervision:** half-space projection that removes only the
  auxiliary-gradient component opposing the final task.
- **Native fusion coalitions:** exact leave-one-scale-out counterfactual logits
  derived from MSHNet's linear fusion layer, without learned helper parameters.
- **Counterfactual responsibility:** safe-background suppression restricted to
  scale contributions that cause a final decision flip.
- **Measure-conditioned SLS:** canonical SLS for non-null masks and explicit
  null-image risk or abstention for zero-mass targets.
- **Reproducibility:** deterministic execution, explicit split files and hashes,
  checkpoint metadata, run labels, and audit utilities.

The current method, claim boundary, gate status, and local audit response are
documented in
[`MSHNet_SDRR_第三轮代码与投稿复核.md`](MSHNet_SDRR_第三轮代码与投稿复核.md).
The preceding TCDS/TFDS review is retained in
[`MSHNet_TCDS_TFDS_第二轮阶段复核.md`](MSHNet_TCDS_TFDS_第二轮阶段复核.md).
The earlier scale-ownership plan is retained in
[`Server_B_Backup_Scale_Ownership.md`](Server_B_Backup_Scale_Ownership.md).

## Repository Layout

```text
.
├── main.py
├── model/
│   ├── MSHNet.py
│   ├── baselines/
│   │   ├── mshnet_official.py
│   │   └── mshnet_deterministic.py
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
│   ├── branch_sdrr_shared_prefix.py
│   ├── summarize_sdrr_formal.py
│   └── optimizer_counterfactual.py
├── tests/
│   └── test_*.py
├── utils/
│   ├── data.py
│   └── metric.py
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
| SDRR | `crs_flip_suppression` | Penalize safe-background decision-flip contributions |
| SDRR attribution controls | `crs_matched_random`, `crs_same_pixel_random_scale`, `crs_magnitude_nonpivotal` | Separate pivotality from event budget, scale identity, and contribution magnitude |
| Null-target handling | `mcsls_null_safe`, `zmsls_null_abstain` | Separate non-null SLS from zero-mass behavior |

Additional scale-subset, delayed-scale, and homotopy controls are listed by
`python main.py --help`.

## Dataset

Put datasets under `datasets/` by default:

```text
datasets/IRSTD-1K/
├── images/
├── masks/
├── trainval.txt
└── test.txt
```

The loader also supports split files under `img_idx/`, for example `img_idx/train_IRSTD-1K.txt` and `img_idx/test_IRSTD-1K.txt`.

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

Reproducible canonical baseline:

```bash
python main.py \
  --dataset-dir datasets/IRSTD-1K \
  --mode train \
  --model-type mshnet \
  --mshnet-variant deterministic \
  --deep-supervision legacy_exact \
  --seed 20260706 \
  --deterministic true \
  --run-label canonical-seed-20260706
```

To run a candidate topology, change only `--deep-supervision` and its
mode-specific arguments while keeping the dataset split, seed, optimizer, and
training budget matched. For example, the counterfactual-responsibility
SDRR candidate uses:

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
formal summary fail-closed checks, projection identities, coalition
reconstruction, optimizer counterfactuals, dataset splits, and metrics.

Key SDRR audit entry points are:

```bash
python tools/audit_sdrr_deletion_stability.py --help
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

Training writes checkpoints and logs to `weight/MSHNet-<timestamp>/`:

- `checkpoint.pkl`
- `weight.pkl`
- `metric.log`
- `epoch_metric.log`
- optional `dea_debug/*.pt`

These outputs are ignored by Git. Keep datasets, trained weights, and reproduction logs outside commits unless they are intentionally published through a release or external storage.

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
