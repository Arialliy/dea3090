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
  --sdrr-normalization event
)

m1="repro_runs/formal_sdrr_m1_all_safe_fp_nuaa_seed20260712_e400"
m3="repro_runs/formal_sdrr_m3_nonpivotal_nuaa_seed20260712_e400"

for run in "$m1" "$m3"; do
  test -f "$run/checkpoint.pkl"
  test "$(wc -l < "$run/epoch_metric.log")" -eq 80
done

CUDA_VISIBLE_DEVICES="${M1_GPU:-0}" "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer m1_all_safe_fp \
  --checkpoint-dir "$m1" \
  --run-label "$(basename "$m1")" \
  >"$m1.console.log" 2>&1 &
pid_m1=$!

CUDA_VISIBLE_DEVICES="${M3_GPU:-2}" "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer m3_magnitude_nonpivotal \
  --checkpoint-dir "$m3" \
  --run-label "$(basename "$m3")" \
  >"$m3.console.log" 2>&1 &
pid_m3=$!

printf 'M1 pid=%s\nM3 pid=%s\n' "$pid_m1" "$pid_m3"

wait "$pid_m1"
wait "$pid_m3"

test "$(wc -l < "$m1/epoch_metric.log")" -eq 400
test "$(wc -l < "$m3/epoch_metric.log")" -eq 400
