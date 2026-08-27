#!/usr/bin/env bash
# Evaluate 4 top-k variants x 4 generated-sample epochs at 50K samples.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABLATION_TAG="${ABLATION_TAG:-run1}"
LABELS=(dense k16_40 k8_20 k4_10)
STEPS=(0010010 0020019 0030028 0040037)
OUT_DIR="${ROOT}/runs/eval_official_imagenet256_topk_s234_${ABLATION_TAG}"
LOG_DIR="${ROOT}/runs/launch_logs"
QUEUE_NAME="eval_B4_topk_s234_${ABLATION_TAG}"
PID_FILE="${LOG_DIR}/pid_${QUEUE_NAME}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"
echo "$$" > "${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

# Interleave variants by epoch: the first wave evaluates all four 10-epoch
# checkpoints, followed by all four 20-, 30-, and 40-epoch checkpoints.
ckpts=""
for step in "${STEPS[@]}"; do
  for label in "${LABELS[@]}"; do
    run_name="gen_B4_revdrift_mae256_topk_s234_p32_n16_${label}_${ABLATION_TAG}"
    ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_${step}.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[eval-topk:error] missing checkpoint: ${ckpt}"
      exit 1
    fi
    ckpts+="${ckpt} "
  done
done

echo "[eval-topk] starting 16 official 50K evaluations"
echo "[eval-topk] output=${OUT_DIR}"
echo "[eval-topk] gpu_groups=0,1;2,3;4,5;6,7 batch_size=64 max_parallel=4"

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
OUT_DIR="${OUT_DIR}" \
bash "${ROOT}/scripts/eval.sh"

echo "[eval-topk] all 16 evaluations completed"
