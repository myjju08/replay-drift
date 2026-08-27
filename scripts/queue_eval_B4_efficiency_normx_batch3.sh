#!/usr/bin/env bash
# Wait for batch3 training to finish, then evaluate 4 checkpoints x 4 runs at 50K.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION_TAG="${ABLATION_TAG:-run1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LABELS=(no_stage1_normx2 no_stage2_normx2 no_stage12 no_stage12_normx2)
STEPS=(0010010 0020019 0030028 0040037)
LOG_DIR="${ROOT}/runs/launch_logs"
QUEUE_NAME="eval_B4_efficiency_normx_batch3_${ABLATION_TAG}"
PID_FILE="${LOG_DIR}/pid_${QUEUE_NAME}"

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

echo "[eval-queue] waiting for all four training runs to finish at step 40037"
while true; do
  any_alive=0
  all_final=1
  missing=()
  for label in "${LABELS[@]}"; do
    run_name="gen_B4_revdrift_mae256_effabl_p32_n16_${label}_${ABLATION_TAG}"
    train_pid_file="${LOG_DIR}/pid_${run_name}"
    final_ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_0040037.pt"
    if [[ ! -f "${final_ckpt}" ]]; then
      all_final=0
      missing+=("${label}")
    fi
    if [[ -f "${train_pid_file}" ]]; then
      train_pid="$(tr -d '[:space:]' < "${train_pid_file}")"
      if [[ -n "${train_pid}" ]] && kill -0 "${train_pid}" 2>/dev/null; then
        any_alive=1
      fi
    fi
  done

  if (( all_final == 1 && any_alive == 0 )); then
    break
  fi
  if (( any_alive == 0 && all_final == 0 )); then
    echo "[eval-queue:error] training stopped before final checkpoints: ${missing[*]}"
    exit 1
  fi
  echo "[eval-queue] training active; final checkpoints still missing: ${missing[*]:-none}"
  sleep "${POLL_SECONDS}"
done

ckpts=""
for label in "${LABELS[@]}"; do
  run_name="gen_B4_revdrift_mae256_effabl_p32_n16_${label}_${ABLATION_TAG}"
  for step in "${STEPS[@]}"; do
    ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_${step}.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[eval-queue:error] missing checkpoint: ${ckpt}"
      exit 1
    fi
    ckpts+="${ckpt} "
  done
done

out_dir="${ROOT}/runs/eval_official_imagenet256_effabl_normx_${ABLATION_TAG}"
echo "[eval-queue] training complete; starting 16 sequential 50K evaluations"
echo "[eval-queue] output=${out_dir}"
PYTHON_BIN="/opt/conda/envs/dualdrift/bin/python" \
CONFIG="${ROOT}/configs/gen/B4_rev-drift_mae256.yaml" \
CKPTS="${ckpts}" \
CFG_SCALE=1.4 \
N_SAMPLES=50000 \
BATCH_SIZE=256 \
LABEL_SOURCE=official_val \
GPU_ID_GROUPS="0,1,2,3,4,5,6,7" \
MAX_PARALLEL=1 \
OUT_DIR="${out_dir}" \
bash "${ROOT}/scripts/eval.sh"

echo "[eval-queue] all 16 evaluations completed"
