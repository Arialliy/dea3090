#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

parent="repro_runs/formal_official_parent_nuaa_seed20260713_through250"
run="repro_runs/formal_official_sdrr_density_nuaa_seed20260713_e400"
test -f "$parent/checkpoint.pkl"
test ! -s "$parent/epoch_metric.log"

if [[ ! -d "$run" ]]; then
  "$PYTHON_BIN" tools/branch_sdrr_shared_prefix.py \
    --source "$parent" --destination "$run" --variant sdrr \
    --epochs 400 --crs-lambda 0.05 --start-epoch 250 \
    --ramp-epochs 50 --safe-kernel 15 --log-interval 40 \
    --normalization safe_density
fi

CUDA_VISIBLE_DEVICES="${GPU_ID:-1}" "$PYTHON_BIN" main.py \
  --mode train --model-type mshnet --mshnet-variant deterministic \
  --dataset-dir datasets/NUAA-SIRST \
  --train-split-file img_idx/train_NUAA-SIRST.txt \
  --test-split-file img_idx/test_NUAA-SIRST.txt \
  --evaluation-protocol official_train_test --split-seed 20260711 \
  --batch-size 4 --num-workers 0 --lr 0.05 --warm-epoch 5 \
  --deterministic true --epochs 400 --evaluation-interval 400 \
  --seed 20260713 --if-checkpoint true --checkpoint-dir "$run" \
  --fusion-regularizer sdrr --sdrr-lambda 0.05 \
  --sdrr-start-ratio 0.625 --sdrr-ramp-ratio 0.125 \
  --sdrr-safe-kernel 15 --sdrr-normalization safe_density \
  --rods-log-interval 40 --run-label "$(basename "$run")" \
  >"$run.console.log" 2>&1

test "$(wc -l < "$run/epoch_metric.log")" -eq 1
rg -q -- '- 0399[[:space:]]+- IoU' "$run/epoch_metric.log"
test -z "$(rg -n 'canonical_loss=(nan|inf)|IoU nan|IoU inf' "$run.console.log" || true)"
