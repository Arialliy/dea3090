#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

event="repro_runs/gate_sdrr_event_nuaa_seed20260713_e120"
density="repro_runs/gate_sdrr_density_nuaa_seed20260713_e120"
for run in "$event" "$density"; do
  test "$(wc -l < "$run/epoch_metric.log")" -eq 80
done

common=(
  main.py --mode train --model-type mshnet --mshnet-variant deterministic
  --dataset-dir datasets/NUAA-SIRST
  --train-split-file img_idx/train_NUAA-SIRST.txt
  --test-split-file img_idx/test_NUAA-SIRST.txt
  --evaluation-protocol internal_holdout --val-fraction 0.2
  --split-seed 20260711 --deterministic true
  --epochs 120 --evaluation-interval 40 --batch-size 4 --num-workers 0
  --lr 0.05 --warm-epoch 5 --if-checkpoint true --seed 20260713
  --fusion-regularizer sdrr --sdrr-lambda 0.05
  --sdrr-start-ratio 0.6666666666666666
  --sdrr-ramp-ratio 0.16666666666666666
  --sdrr-safe-kernel 15 --rods-log-interval 10
)

CUDA_VISIBLE_DEVICES="${EVENT_GPU:-1}" "$PYTHON_BIN" "${common[@]}" \
  --sdrr-normalization event --checkpoint-dir "$event" \
  --run-label "$(basename "$event")" >"$event.console.log" 2>&1 &
pid_event=$!

CUDA_VISIBLE_DEVICES="${DENSITY_GPU:-2}" "$PYTHON_BIN" "${common[@]}" \
  --sdrr-normalization safe_density --checkpoint-dir "$density" \
  --run-label "$(basename "$density")" >"$density.console.log" 2>&1 &
pid_density=$!

printf 'event pid=%s\ndensity pid=%s\n' "$pid_event" "$pid_density"
wait "$pid_event"
wait "$pid_density"

for run in "$event" "$density"; do
  test "$(tail -n 1 "$run/epoch_metric.log" | sed -E 's/.*- ([0-9]{4})[[:space:]]+-.*/\1/')" = "0119"
  test -z "$(rg -n 'canonical_loss=(nan|inf)|IoU nan|IoU inf' "$run.console.log" || true)"
done
