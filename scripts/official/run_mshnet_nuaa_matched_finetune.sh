#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-/home/ly/BasicIRSTD/infrarenet/bin/python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASET_DIR=${DATASET_DIR:-${ROOT}/datasets/NUAA-SIRST}
OUT_DIR=${OUT_DIR:-${ROOT}/repro_runs/mshnet_nuaa_matched_finetune}
INIT_FROM_BASELINE=${INIT_FROM_BASELINE:-${ROOT}/weight/MSHNet-2026-07-08-02-11-49/checkpoint_nuaa_mshnet_baseline_best_iou_e381.pkl}

EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-4}
PIN_MEMORY=${PIN_MEMORY:-false}
LR=${LR:-0.005}
SEED=${SEED:-20260706}
DETERMINISTIC=${DETERMINISTIC:-true}

BASELINE_IOU=${BASELINE_IOU:-0.7461767422765062}
BASELINE_PD=${BASELINE_PD:-0.9619771863117871}
BASELINE_FA=${BASELINE_FA:-25.312477183119157}
PD_MIN=${PD_MIN:-0.9569771863117871}

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

for path in "${DATASET_DIR}" "${INIT_FROM_BASELINE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: missing path: ${path}" >&2
    exit 2
  fi
done

CMD=(
  "${PYTHON}" -u main.py
  --model-type mshnet
  --dataset-dir "${DATASET_DIR}"
  --mode train
  --epochs "${EPOCHS}"
  --warm-epoch -1
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --pin-memory "${PIN_MEMORY}"
  --lr "${LR}"
  --seed "${SEED}"
  --deterministic "${DETERMINISTIC}"
  --init-from-baseline "${INIT_FROM_BASELINE}"
  --pd-fa-min-pd "${PD_MIN}"
  --pd-fa-min-iou "${BASELINE_IOU}"
  --paired-baseline-iou "${BASELINE_IOU}"
  --pd-fa-iou-margin 0.0
)

printf '%q ' "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}" "${CMD[@]}" > "${OUT_DIR}/command.txt"
printf '\n' >> "${OUT_DIR}/command.txt"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/train.log"

RUN_DIR="$(
  find "${ROOT}/weight" -maxdepth 1 -type d -name 'MSHNet-*' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
)"

if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: could not locate latest MSHNet run dir" >&2
  exit 3
fi

ln -sfn "${RUN_DIR}" "${OUT_DIR}/run_dir"
echo "${RUN_DIR}" > "${OUT_DIR}/run_dir.txt"

"${PYTHON}" tools/official/write_dea_checkpoint_summary.py \
  --checkpoint "${RUN_DIR}/checkpoint_best_iou.pkl" \
  --weight "${RUN_DIR}/weight.pkl" \
  --dataset NUAA-SIRST \
  --method MSHNet-matched-finetune \
  --checkpoint_role best_iou \
  --source_run_dir "${RUN_DIR}" \
  --output "${OUT_DIR}/mshnet_nuaa_matched_finetune_best_iou_summary.json"

"${PYTHON}" tools/official/analyze_dea_epoch_metrics.py \
  --epoch_metric_log "${RUN_DIR}/epoch_metric.log" \
  --baseline_iou "${BASELINE_IOU}" \
  --baseline_pd "${BASELINE_PD}" \
  --baseline_fa "${BASELINE_FA}" \
  --min_delta_iou 0.0 \
  --min_delta_pd -0.005 \
  --max_delta_fa 0.0 \
  --output "${OUT_DIR}/mshnet_nuaa_matched_finetune_epoch_gate_audit.json"

cat > "${OUT_DIR}/MANIFEST.json" <<JSON
{
  "stage": "P2_MATCHED_MSHNET_FINETUNE_CONTROL",
  "method": "MSHNet-matched-finetune",
  "dataset": "NUAA-SIRST",
  "run_dir": "${RUN_DIR}",
  "train_log": "${OUT_DIR}/train.log",
  "epochs": ${EPOCHS},
  "lr": ${LR},
  "seed": ${SEED},
  "deterministic": ${DETERMINISTIC},
  "init_from_baseline": "${INIT_FROM_BASELINE}"
}
JSON

echo "DONE: MSHNet matched finetune outputs written to ${OUT_DIR}"
