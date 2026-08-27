#!/usr/bin/env bash
# Wait for Phase-1 training, then run 16 official 50K evaluations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION_TAG="${ABLATION_TAG:-seed43_run1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
VARIANTS=(frozen s4_supcon s34_supcon s34_supcon_ce)
STEPS=(0010010 0020019 0030028 0040037)
LOG_DIR="${ROOT}/runs/launch_logs"
QUEUE_NAME="eval_B4_reverse_feature_adapter_batch11_${ABLATION_TAG}"
PID_FILE="${LOG_DIR}/pid_${QUEUE_NAME}"

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

echo "[feature-adapter-eval] waiting for all four final checkpoints"
while true; do
  any_alive=0
  all_final=1
  missing=()
  for variant in "${VARIANTS[@]}"; do
    run_name="gen_B4_revdrift_mae256_full86_p128_n32_gauss075_adapter_${variant}_${ABLATION_TAG}"
    train_pid_file="${LOG_DIR}/pid_${run_name}"
    final_ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_0040037.pt"
    if [[ ! -f "${final_ckpt}" ]]; then
      all_final=0
      missing+=("${variant}")
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
    echo "[feature-adapter-eval:error] training stopped before final checkpoints: ${missing[*]}"
    exit 1
  fi
  echo "[feature-adapter-eval] training active; missing finals: ${missing[*]:-none}"
  sleep "${POLL_SECONDS}"
done

ckpts=""
for step in "${STEPS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    run_name="gen_B4_revdrift_mae256_full86_p128_n32_gauss075_adapter_${variant}_${ABLATION_TAG}"
    ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_${step}.pt"
    [[ -f "${ckpt}" ]] || { echo "[feature-adapter-eval:error] missing checkpoint: ${ckpt}"; exit 1; }
    ckpts+="${ckpt} "
  done
done

out_dir="${ROOT}/runs/eval_official_imagenet256_reverse_feature_adapter_${ABLATION_TAG}"
echo "[feature-adapter-eval] training complete; starting 16 official 50K evaluations"
echo "[feature-adapter-eval] output=${out_dir}"
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

echo "[feature-adapter-eval] all 16 evaluations completed"
