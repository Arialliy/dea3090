#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

run="repro_runs/formal_sdrr_m3_nonpivotal_fixed_nuaa_seed20260712_e400"
test -f "$run/checkpoint.pkl"
test "$(wc -l < "$run/epoch_metric.log")" -eq 80

CUDA_VISIBLE_DEVICES="${GPU:-2}" "$PYTHON_BIN" main.py \
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
  --seed 20260712 \
  --fusion-regularizer m3_magnitude_nonpivotal \
  --sdrr-lambda 0.05 \
  --sdrr-start-ratio 0.625 \
  --sdrr-ramp-ratio 0.125 \
  --sdrr-safe-kernel 15 \
  --sdrr-normalization event \
  --sdrr-match-shared-grad-norm true \
  --rods-log-interval 40 \
  --checkpoint-dir "$run" \
  --run-label "$(basename "$run")" \
  >"$run.console.log" 2>&1

test "$(wc -l < "$run/epoch_metric.log")" -eq 400
test -z "$(rg -n 'canonical_loss=(nan|inf)|IoU nan|IoU inf' "$run.console.log" || true)"
