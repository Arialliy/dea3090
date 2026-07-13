#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/md0/ly/BasicIRSTD/infrarenet/bin/python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

baseline="repro_runs/gate_density_baseline_nudt_seed20260713_e120"
density="repro_runs/gate_density_nudt_seed20260713_e120"
common=(
  main.py --mode train --model-type mshnet --mshnet-variant deterministic
  --dataset-dir datasets/NUDT-SIRST
  --train-split-file img_idx/train_NUDT-SIRST.txt
  --test-split-file img_idx/test_NUDT-SIRST.txt
  --evaluation-protocol internal_holdout --val-fraction 0.2
  --split-seed 20260711 --deterministic true
  --epochs 120 --evaluation-interval 40 --batch-size 4 --num-workers 0
  --lr 0.05 --warm-epoch 5 --if-checkpoint true --seed 20260713
  --sdrr-start-ratio 0.6666666666666666
  --sdrr-ramp-ratio 0.16666666666666666 --sdrr-safe-kernel 15
)

CUDA_VISIBLE_DEVICES="${GPU_ID:-2}" "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer none --deep-supervision legacy_exact \
  --sdrr-lambda 0 --rods-log-interval 0 \
  --checkpoint-dir "$baseline" --run-label "$(basename "$baseline")" \
  >"$baseline.console.log" 2>&1 &
pid_baseline=$!

CUDA_VISIBLE_DEVICES="${GPU_ID:-2}" "$PYTHON_BIN" "${common[@]}" \
  --fusion-regularizer sdrr --sdrr-lambda 0.05 \
  --sdrr-normalization safe_density --rods-log-interval 20 \
  --checkpoint-dir "$density" --run-label "$(basename "$density")" \
  >"$density.console.log" 2>&1 &
pid_density=$!

printf 'baseline pid=%s\ndensity pid=%s\n' "$pid_baseline" "$pid_density"
wait "$pid_baseline"
wait "$pid_density"

for run in "$baseline" "$density"; do
  test "$(tail -n 1 "$run/epoch_metric.log" | sed -E 's/.*- ([0-9]{4})[[:space:]]+-.*/\1/')" = "0119"
done
