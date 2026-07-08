#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ly/DEA}
PYTHON=${PYTHON:-python3}

cd "${ROOT}"

METHOD_DIR="docs/internal/dea_lite"
EVIDENCE_DIR="docs/internal/dea_lite_0p005"
DECISION_JSON="${METHOD_DIR}/DEA_LITE_FREEZE_DECISION.json"
CLAIM_GUARD_JSON="${METHOD_DIR}/DEA_LITE_FULL_DEA_CLAIM_GUARD.json"

mkdir -p "${METHOD_DIR}"

if [[ ! -f "${METHOD_DIR}/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md" ]]; then
  echo "ERROR: missing method freeze document: ${METHOD_DIR}/DEA_LITE_METHOD_FREEZE_AFTER_NUAA.md" >&2
  exit 2
fi

if [[ ! -d "${EVIDENCE_DIR}" ]]; then
  echo "ERROR: missing evidence directory: ${EVIDENCE_DIR}" >&2
  exit 3
fi

if [[ ! -f "tools/official/check_no_full_dea_claim_from_dea_lite.py" ]]; then
  echo "ERROR: missing claim guard tool." >&2
  exit 4
fi

# Protect method code from accidental freeze-commit contamination.
PROTECTED_CHANGED="$(git diff --name-only -- model/MSHNet.py model/loss.py main.py utils 2>/dev/null || true)"
if [[ -n "${PROTECTED_CHANGED}" ]]; then
  echo "ERROR: protected implementation files have uncommitted changes:" >&2
  echo "${PROTECTED_CHANGED}" >&2
  echo "Freeze commit must not modify model/loss/main/utils implementation." >&2
  exit 5
fi

"${PYTHON}" tools/official/check_no_full_dea_claim_from_dea_lite.py \
  --root "${ROOT}" \
  --output "${CLAIM_GUARD_JSON}"

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git branch --show-current 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git status --short 2>/dev/null | wc -l | tr -d ' ')"

cat > "${DECISION_JSON}" <<JSON
{
  "decision": "DEA_LITE_FROZEN_AS_PILOT_ABLATION_LIMITATION_EVIDENCE",
  "root": "${ROOT}",
  "branch": "${GIT_BRANCH}",
  "commit": "${GIT_COMMIT}",
  "git_dirty_entry_count_after_freeze_files": ${GIT_DIRTY},
  "method_status": "DEA-lite is not full DEA",
  "evidence_status": {
    "NUDT-SIRST": "positive_anchor",
    "IRSTD-1K": "fa_control_positive_signal",
    "NUAA-SIRST": "stable_negative_dataset_dependent_failure"
  },
  "nuaa_negative": {
    "gate_pass": false,
    "decision": "DEA_LITE_NEGATIVE_DATASET_DEPENDENT",
    "delta_iou": -0.0336,
    "delta_pd": -0.0266,
    "delta_fa": 2.2104,
    "num_gate_pass_epochs": 0
  },
  "allowed_next_step": "open full-dea-predeclare-design branch and add protocol document only",
  "forbidden_next_steps": [
    "claim DEA-lite validates full DEA",
    "claim universal DEA-lite improvement",
    "run lambda 0.01 as DEA-main rescue",
    "mix Full DEA implementation code into the DEA-lite freeze branch"
  ],
  "claim_guard_json": "${CLAIM_GUARD_JSON}"
}
JSON

echo "DONE: wrote ${DECISION_JSON}"
