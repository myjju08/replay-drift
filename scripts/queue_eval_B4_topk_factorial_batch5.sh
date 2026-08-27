#!/usr/bin/env bash
# Wait for batch5, then run 4 variants x 3 checkpoints at 50K samples.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION_TAG="${ABLATION_TAG:-seed43_run1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LABELS=(dense pos16 neg40 both16_40)
STEPS=(0010010 0020019 0030028)
LOG_DIR="${ROOT}/runs/launch_logs"
QUEUE_NAME="eval_B4_topk_factorial_${ABLATION_TAG}"
PID_FILE="${LOG_DIR}/pid_${QUEUE_NAME}"

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

echo "[eval-factorial] waiting for all four final step-30028 checkpoints"
while true; do
  any_alive=0
  all_final=1
  missing=()
  for label in "${LABELS[@]}"; do
    run_name="gen_B4_revdrift_mae256_topk_factorial_s234_p32_n16_${label}_${ABLATION_TAG}"
    train_pid_file="${LOG_DIR}/pid_${run_name}"
    final_ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_0030028.pt"
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
    echo "[eval-factorial:error] training stopped before final checkpoints: ${missing[*]}"
    exit 1
  fi
  echo "[eval-factorial] training active; final checkpoints missing: ${missing[*]:-none}"
  sleep "${POLL_SECONDS}"
done

# Interleave variants by epoch so each evaluation wave gives a matched comparison.
ckpts=""
for step in "${STEPS[@]}"; do
  for label in "${LABELS[@]}"; do
    run_name="gen_B4_revdrift_mae256_topk_factorial_s234_p32_n16_${label}_${ABLATION_TAG}"
    ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_${step}.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[eval-factorial:error] missing checkpoint: ${ckpt}"
      exit 1
    fi
    ckpts+="${ckpt} "
  done
done

out_dir="${ROOT}/runs/eval_official_imagenet256_topk_factorial_${ABLATION_TAG}"
echo "[eval-factorial] starting 12 official 50K evaluations"
echo "[eval-factorial] output=${out_dir}"
PYTHON_BIN="/opt/conda/envs/dualdrift/bin/python" \
CONFIG="${ROOT}/configs/gen/B4_rev-drift_mae256.yaml" \
CKPTS="${ckpts}" \
CFG_SCALE=1.4 \
N_SAMPLES=50000 \
BATCH_SIZE=64 \
SEED=0 \
LABEL_SOURCE=official_val \
PR_REF_COUNT=10000 \
GPU_ID_GROUPS="0,1;2,3;4,5;6,7" \
MAX_PARALLEL=4 \
SAVE_SAMPLE_NPZ=0 \
KEEP_SAMPLE_NPZ=0 \
OUT_DIR="${out_dir}" \
bash "${ROOT}/scripts/eval.sh"

echo "[eval-factorial] all 12 evaluations completed"
