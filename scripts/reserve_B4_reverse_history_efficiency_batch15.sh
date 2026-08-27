#!/usr/bin/env bash
# Reserve batch15 behind the complete batch14 official evaluation queue.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG_DIR="${ROOT}/runs/launch_logs"
PREV_QUEUE_PID_FILE="${LOG_DIR}/pid_eval_B4_reverse_history_causal_batch14_common10_seed43_run1"
PREV_OUT="${ROOT}/runs/eval_official_imagenet256_reverse_history_causal_common10_seed43_run1"
EXPECTED_PREV_JSONS=12
RESERVATION_PID_FILE="${LOG_DIR}/pid_reserve_B4_reverse_history_efficiency_batch15"
NEXT_EVAL_LOG="${LOG_DIR}/eval_B4_reverse_history_efficiency_batch15_common10_seed43_run1.log"

mkdir -p "${LOG_DIR}"
echo "$$" > "${RESERVATION_PID_FILE}"
trap 'rm -f "${RESERVATION_PID_FILE}"' EXIT

echo "[reserve-history-efficiency] waiting for batch14 official evaluations"
while true; do
  json_count=0
  if [[ -d "${PREV_OUT}" ]]; then
    json_count="$(find "${PREV_OUT}" -maxdepth 1 -name '*.json' | wc -l)"
  fi
  prev_alive=0
  if [[ -f "${PREV_QUEUE_PID_FILE}" ]]; then
    prev_pid="$(tr -d '[:space:]' < "${PREV_QUEUE_PID_FILE}")"
    if [[ -n "${prev_pid}" ]] && kill -0 "${prev_pid}" 2>/dev/null; then
      prev_alive=1
    fi
  fi
  if (( json_count >= EXPECTED_PREV_JSONS && prev_alive == 0 )); then break; fi
  if (( prev_alive == 0 && json_count < EXPECTED_PREV_JSONS )); then
    echo "[reserve-history-efficiency] previous queue not active yet; results=${json_count}/${EXPECTED_PREV_JSONS}"
  else
    echo "[reserve-history-efficiency] previous queue active; results=${json_count}/${EXPECTED_PREV_JSONS}"
  fi
  sleep "${POLL_SECONDS}"
done

echo "[reserve-history-efficiency] batch14 complete; launching batch15"
bash "${ROOT}/scripts/run_B4_reverse_history_efficiency_ablation_batch15.sh"

echo "[reserve-history-efficiency] starting batch15 evaluation waiter"
setsid -f bash "${ROOT}/scripts/queue_eval_B4_reverse_history_efficiency_ablation_batch15.sh" \
  >> "${NEXT_EVAL_LOG}" 2>&1
echo "[reserve-history-efficiency] reservation handed off successfully"
