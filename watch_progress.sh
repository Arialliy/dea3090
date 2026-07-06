#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/ly/DEA"
INTERVAL="${INTERVAL:-5}"
LOG="${1:-}"

if [[ -z "$LOG" ]]; then
  LOG="$(find "$ROOT/repro_runs" -path '*/train.log' -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)"
fi

if [[ -z "$LOG" || ! -f "$LOG" ]]; then
  echo "No train.log found. Usage:"
  echo "  $0 /home/ly/DEA/repro_runs/<run_name>/train.log"
  echo "Optional: INTERVAL=2 $0 <train.log>"
  exit 1
fi

find_train_pid() {
  pgrep -af "main.py .*--mode train|main.py --dataset-dir .* --mode train" \
    | grep -v "watch_progress.sh" \
    | awk 'NR==1 {print $1}'
}

print_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits
  else
    echo "nvidia-smi not found"
  fi
}

print_best_iou() {
  grep -a " - IoU" "$LOG" | awk '
    {
      epoch=$3
      for (i=1; i<=NF; i++) {
        if ($i=="IoU") iou=$(i+1)
        if ($i=="PD") pd=$(i+1)
        if ($i=="FA") fa=$(i+1)
      }
      if (iou+0 > best+0) {
        best=iou
        best_epoch=epoch
        best_pd=pd
        best_fa=fa
      }
    }
    END {
      if (best != "") {
        printf "best IoU: epoch %s | IoU %s | PD %s | FA %s\n", best_epoch, best, best_pd, best_fa
      } else {
        print "best IoU: waiting for first eval"
      }
    }'
}

while true; do
  printf '\033[H\033[J'
  echo "Time: $(date '+%F %T')"
  echo "Log:  $LOG"
  echo

  PID="${PID:-$(find_train_pid || true)}"
  if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
    echo "Process:"
    ps -p "$PID" -o pid,stat,etime,cmd
  else
    echo "Process: not found or already finished"
  fi
  echo

  echo "Current train progress:"
  tr '\r' '\n' < "$LOG" | grep -aE "Epoch [0-9]+, loss" | tail -n 1 || echo "waiting for train progress"
  echo

  echo "Latest eval:"
  grep -a " - IoU" "$LOG" | tail -n 8 || echo "waiting for eval"
  echo

  print_best_iou
  echo

  echo "GPU:"
  print_gpu
  echo
  echo "Refresh interval: ${INTERVAL}s. Press Ctrl+C to stop watching."

  sleep "$INTERVAL"
done
