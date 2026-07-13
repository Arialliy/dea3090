#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

common=(
  main.py
  --mode train
  --model-type mshnet
  --dataset-dir datasets/NUAA-SIRST
  --train-split-file img_idx/train_NUAA-SIRST.txt
  --test-split-file img_idx/test_NUAA-SIRST.txt
  --val-fraction 0.2
  --split-seed 20260711
  --deterministic true
  --epochs 400
  --batch-size 4
  --num-workers 4
  --lr 0.05
  --warm-epoch 5
  --if-checkpoint true
  --seed 20260712
  --sdrr-lambda 0.05
  --sdrr-start-ratio 0.625
  --sdrr-ramp-ratio 0.125
  --sdrr-safe-kernel 15
  --rods-log-interval 40
)

m2="repro_runs/formal_sdrr_m2_pivotal_pixel_nuaa_seed20260712_e400"
m4="repro_runs/formal_sdrr_m4_random_scale_sharedgrad_nuaa_seed20260712_e400"
density="repro_runs/formal_sdrr_norm_density_nuaa_seed20260712_e400"

for run in "$m2" "$m4" "$density"; do
  test -f "$run/checkpoint.pkl"
  test "$(wc -l < "$run/epoch_metric.log")" -eq 80
done

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer m2_pivotal_pixel \
  --sdrr-normalization event \
  --sdrr-match-shared-grad-norm true \
  --checkpoint-dir "$m2" \
  --run-label "$(basename "$m2")" \
  >"$m2.console.log" 2>&1 &
pid_m2=$!

CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer m4_random_scale \
  --sdrr-normalization event \
  --sdrr-match-shared-grad-norm true \
  --checkpoint-dir "$m4" \
  --run-label "$(basename "$m4")" \
  >"$m4.console.log" 2>&1 &
pid_m4=$!

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer sdrr \
  --sdrr-normalization safe_density \
  --checkpoint-dir "$density" \
  --run-label "$(basename "$density")" \
  >"$density.console.log" 2>&1 &
pid_density=$!

printf 'M2 pid=%s\nM4 pid=%s\nDensity pid=%s\n' \
  "$pid_m2" "$pid_m4" "$pid_density"

wait "$pid_m2"
wait "$pid_m4"
wait "$pid_density"
