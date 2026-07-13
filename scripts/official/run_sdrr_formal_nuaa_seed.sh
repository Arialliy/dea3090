#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-20260713}"
case "$SEED" in
  20260711|20260712|20260713) ;;
  *) printf 'SEED must be one of 20260711, 20260712, 20260713\n' >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

common=(
  main.py
  --mode train
  --model-type mshnet
  --mshnet-variant deterministic
  --dataset-dir datasets/NUAA-SIRST
  --train-split-file img_idx/train_NUAA-SIRST.txt
  --test-split-file img_idx/test_NUAA-SIRST.txt
  --evaluation-protocol official_train_test
  --split-seed 20260711
  --batch-size 4
  --num-workers 0
  --lr 0.05
  --warm-epoch 5
  --deterministic true
  --crs-start-epoch 250
  --crs-ramp-epochs 50
  --sdrr-safe-kernel 15
)

parent="repro_runs/formal_official_parent_nuaa_seed${SEED}_through250"
baseline="repro_runs/formal_official_baseline_nuaa_seed${SEED}_e400"
sdrr="repro_runs/formal_official_sdrr_nuaa_seed${SEED}_e400"

if [[ ! -d "$parent" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "${common[@]}" \
    --epochs 251 --evaluation-interval 400 --skip-final-evaluation true \
    --seed "$SEED" \
    --fusion-regularizer none --deep-supervision legacy_exact \
    --crs-lambda 0 --rods-log-interval 0 \
    --run-dir "$parent" --run-label "$(basename "$parent")" \
    >"${parent}.console.log" 2>&1
fi

test -f "$parent/checkpoint.pkl"
test ! -s "$parent/epoch_metric.log"
test "$(wc -l < "$parent/split_train.txt")" -eq 213
test "$(wc -l < "$parent/split_test.txt")" -eq 214

if [[ ! -d "$baseline" ]]; then
  "$PYTHON_BIN" tools/branch_sdrr_shared_prefix.py \
    --source "$parent" --destination "$baseline" \
    --variant baseline --epochs 400
fi
if [[ ! -d "$sdrr" ]]; then
  "$PYTHON_BIN" tools/branch_sdrr_shared_prefix.py \
    --source "$parent" --destination "$sdrr" \
    --variant sdrr --epochs 400 --crs-lambda 0.05 \
    --start-epoch 250 --ramp-epochs 50 --safe-kernel 15 --log-interval 40
fi

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "${common[@]}" \
  --epochs 400 --evaluation-interval 400 --seed "$SEED" --if-checkpoint true \
  --checkpoint-dir "$baseline" \
  --fusion-regularizer none --deep-supervision legacy_exact \
  --crs-lambda 0 --rods-log-interval 0 \
  --run-label "$(basename "$baseline")" \
  >"${baseline}.console.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" "${common[@]}" \
  --epochs 400 --evaluation-interval 400 --seed "$SEED" --if-checkpoint true \
  --checkpoint-dir "$sdrr" \
  --fusion-regularizer sdrr --sdrr-lambda 0.05 \
  --sdrr-start-ratio 0.625 --sdrr-ramp-ratio 0.125 \
  --rods-log-interval 40 --run-label "$(basename "$sdrr")" \
  >"${sdrr}.console.log" 2>&1

test "$(wc -l < "$baseline/epoch_metric.log")" -eq 1
test "$(wc -l < "$sdrr/epoch_metric.log")" -eq 1
rg -q -- '- 0399[[:space:]]+- IoU' "$baseline/epoch_metric.log"
rg -q -- '- 0399[[:space:]]+- IoU' "$sdrr/epoch_metric.log"
