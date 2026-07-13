#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

wait_for_complete() {
  local run="$1"
  while [[ ! -f "$run/epoch_metric.log" ]] || \
        [[ "$(wc -l < "$run/epoch_metric.log")" -lt 400 ]]; do
    if [[ -f "$run/INVALID_RUN.json" ]]; then
      printf 'Upstream run is invalid: %s\n' "$run" >&2
      return 1
    fi
    sleep 15
  done
}

run_control() {
  local gpu="$1"
  local seed="$2"
  local regularizer="$3"
  local normalization="$4"
  local run="$5"

  test -f "$run/checkpoint.pkl"
  test "$(wc -l < "$run/epoch_metric.log")" -eq 80
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" main.py \
    --mode train \
    --model-type mshnet \
    --dataset-dir datasets/NUAA-SIRST \
    --train-split-file img_idx/train_NUAA-SIRST.txt \
    --test-split-file img_idx/test_NUAA-SIRST.txt \
    --val-fraction 0.2 \
    --split-seed 20260711 \
    --deterministic true \
    --epochs 400 \
    --batch-size 4 \
    --num-workers 4 \
    --lr 0.05 \
    --warm-epoch 5 \
    --if-checkpoint true \
    --seed "$seed" \
    --fusion-regularizer "$regularizer" \
    --sdrr-lambda 0.05 \
    --sdrr-start-ratio 0.625 \
    --sdrr-ramp-ratio 0.125 \
    --sdrr-safe-kernel 15 \
    --sdrr-normalization "$normalization" \
    --sdrr-match-shared-grad-norm true \
    --rods-log-interval 40 \
    --checkpoint-dir "$run" \
    --run-label "$(basename "$run")" \
    >"$run.console.log" 2>&1
  test "$(wc -l < "$run/epoch_metric.log")" -eq 400
}

(
  wait_for_complete \
    repro_runs/formal_sdrr_m1_all_safe_fp_nuaa_seed20260712_e400
  run_control 0 20260712 scale_budget_random event \
    repro_runs/formal_sdrr_m5_matched_random_nuaa_seed20260712_e400
) &
pid_m5=$!

(
  wait_for_complete \
    repro_runs/formal_sdrr_m3_nonpivotal_fixed_nuaa_seed20260712_e400
  GPU_ID=2 SEED=20260713 \
    bash scripts/official/run_sdrr_formal_nuaa_seed.sh
  run_control 2 20260711 sdrr safe_density \
    repro_runs/formal_sdrr_norm_density_nuaa_seed20260711_e400
) &
pid_official_then_density11=$!

(
  wait_for_complete \
    repro_runs/formal_sdrr_m4_random_scale_contribmatch_nuaa_seed20260712_e400
  run_control 1 20260713 sdrr safe_density \
    repro_runs/formal_sdrr_norm_density_nuaa_seed20260713_e400
) &
pid_density13=$!

printf 'M5 sequence pid=%s\nOfficial-then-density-11 sequence pid=%s\nDensity-13 sequence pid=%s\n' \
  "$pid_m5" "$pid_official_then_density11" "$pid_density13"

wait "$pid_m5"
wait "$pid_official_then_density11"
wait "$pid_density13"
