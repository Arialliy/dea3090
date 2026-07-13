#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

baseline="repro_runs/gate_rcr_baseline_nuaa_seed20260713_e120"
rcr="repro_runs/gate_rcr_nuaa_seed20260713_e120"
for run in "$baseline" "$rcr"; do
  test -f "$run/checkpoint.pkl"
  test "$(wc -l < "$run/epoch_metric.log")" -eq 80
done

common=(
  main.py
  --mode train
  --model-type mshnet
  --mshnet-variant deterministic
  --dataset-dir datasets/NUAA-SIRST
  --train-split-file img_idx/train_NUAA-SIRST.txt
  --test-split-file img_idx/test_NUAA-SIRST.txt
  --evaluation-protocol internal_holdout
  --val-fraction 0.2
  --split-seed 20260711
  --deterministic true
  --epochs 120
  --evaluation-interval 40
  --batch-size 4
  --num-workers 0
  --lr 0.05
  --warm-epoch 5
  --if-checkpoint true
  --seed 20260713
  --sdrr-start-ratio 0.6666666666666666
  --sdrr-ramp-ratio 0.16666666666666666
  --sdrr-safe-kernel 15
)

CUDA_VISIBLE_DEVICES="${BASELINE_GPU:-1}" "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer none \
  --deep-supervision legacy_exact \
  --sdrr-lambda 0 \
  --rods-log-interval 0 \
  --checkpoint-dir "$baseline" \
  --run-label "$(basename "$baseline")" \
  >"$baseline.console.log" 2>&1 &
pid_baseline=$!

CUDA_VISIBLE_DEVICES="${RCR_GPU:-2}" "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer rcr \
  --sdrr-lambda 0.05 \
  --rods-log-interval 10 \
  --checkpoint-dir "$rcr" \
  --run-label "$(basename "$rcr")" \
  >"$rcr.console.log" 2>&1 &
pid_rcr=$!

printf 'baseline pid=%s\nrcr pid=%s\n' "$pid_baseline" "$pid_rcr"
wait "$pid_baseline"
wait "$pid_rcr"

test "$(tail -n 1 "$baseline/epoch_metric.log" | sed -E 's/.*- ([0-9]{4})[[:space:]]+-.*/\1/')" = "0119"
test "$(tail -n 1 "$rcr/epoch_metric.log" | sed -E 's/.*- ([0-9]{4})[[:space:]]+-.*/\1/')" = "0119"
test -z "$(rg -n 'canonical_loss=(nan|inf)|IoU nan|IoU inf' "$rcr.console.log" || true)"
