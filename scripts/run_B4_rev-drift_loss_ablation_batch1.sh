#!/usr/bin/env bash
# First feature-loss necessity screen: four concurrent 2-GPU B/4 reverse runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_LAUNCHER="${ROOT}/scripts/run_B4_rev-drift_mae256.sh"
ABLATION_TAG="${ABLATION_TAG:-run1}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"

PROFILES=(all no_global no_norm_x no_stage1)
LABELS=(baseline no_global no_norm_x no_stage1)
GPU_GROUPS=(0,1 2,3 4,5 6,7)
PORTS=(29710 29711 29712 29713)

for index in "${!PROFILES[@]}"; do
  profile="${PROFILES[$index]}"
  label="${LABELS[$index]}"
  workdir="${ROOT}/runs/gen_B4_revdrift_mae256_lossabl_p32_n16_${label}_${ABLATION_TAG}"
  if [[ "${ALLOW_RESUME}" != "1" && -f "${workdir}/checkpoints/ckpt_latest.pt" ]]; then
    echo "[error] existing checkpoint would be resumed: ${workdir}/checkpoints/ckpt_latest.pt"
    echo "[error] choose a new ABLATION_TAG or set ALLOW_RESUME=1"
    exit 1
  fi

  echo "[batch1] launching ${label} on GPUs ${GPU_GROUPS[$index]}"
  FEATURE_LOSS_PROFILE="${profile}" \
  LAYER_TEMPERATURE_PROFILE=uniform \
  TRAIN_BATCH_SIZE=10 \
  POS_PER_SAMPLE=32 \
  NEG_PER_SAMPLE=16 \
  TOTAL_GENERATED_EPOCHS=100 \
  SAVE_PER_GENERATED_EPOCHS=10 \
  EVAL_PER_STEP=1000000000 \
  GPU_IDS="${GPU_GROUPS[$index]}" \
  NPROC_PER_NODE=2 \
  MASTER_PORT="${PORTS[$index]}" \
  WORKDIR="${workdir}" \
  RUN_NAME="$(basename "${workdir}")" \
  LAUNCH_MODE=background \
  bash "${BASE_LAUNCHER}"
done

echo "[batch1] launched all four ablations"
for label in "${LABELS[@]}"; do
  echo "[batch1] tail -f ${ROOT}/runs/launch_logs/gen_B4_revdrift_mae256_lossabl_p32_n16_${label}_${ABLATION_TAG}.log"
done
