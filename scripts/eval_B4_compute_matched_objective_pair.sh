#!/usr/bin/env bash
# Official 50K evaluation for DualDrift 3+3 versus force-matched reverse-6.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_TAG="${RUN_TAG:-run1}"
VARIANTS=(dual3x3 reverse6)
STEPS=(0010010 0020019 0030028 0040037)
OUT_DIR="${ROOT}/runs/eval_official_imagenet256_compute_matched_objectives_${RUN_TAG}"

mkdir -p "${OUT_DIR}"
ckpts=""
for step in "${STEPS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    run_name="gen_B4_mae256_full86_p128_n32_compute_matched_${variant}_${RUN_TAG}"
    ckpt="${ROOT}/runs/${run_name}/checkpoints/ckpt_step_${step}.pt"
    if [[ ! -f "${ckpt}" ]]; then
      echo "[eval-compute-match:error] missing checkpoint: ${ckpt}"
      exit 1
    fi
    ckpts+="${ckpt} "
  done
done

echo "[eval-compute-match] starting 8 official 50K evaluations"
PYTHON_BIN="/opt/conda/envs/dualdrift/bin/python" \
CONFIG="${ROOT}/configs/gen/B4_dual-drift_mae256.yaml" \
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

echo "[eval-compute-match] all 8 evaluations completed"
