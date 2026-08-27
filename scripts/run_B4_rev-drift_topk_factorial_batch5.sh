#!/usr/bin/env bash
# New-seed 2x2 factorial: positive and negative top-k truncation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_LAUNCHER="${ROOT}/scripts/run_B4_rev-drift_mae256.sh"
ABLATION_TAG="${ABLATION_TAG:-seed43_run1}"
TRAIN_SEED="${TRAIN_SEED:-43}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"

LABELS=(dense pos16 neg40 both16_40)
TOP_K_POS=(0 16 0 16)
TOP_K_NEG=(0 0 40 40)
GPU_GROUPS=(0,1 2,3 4,5 6,7)
PORTS=(29750 29751 29752 29753)

for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  workdir="${ROOT}/runs/gen_B4_revdrift_mae256_topk_factorial_s234_p32_n16_${label}_${ABLATION_TAG}"
  if [[ "${ALLOW_RESUME}" != "1" && -f "${workdir}/checkpoints/ckpt_latest.pt" ]]; then
    echo "[error] existing checkpoint would be resumed: ${workdir}/checkpoints/ckpt_latest.pt"
    echo "[error] choose a new ABLATION_TAG or set ALLOW_RESUME=1"
    exit 1
  fi

  echo "[topk-factorial] launching ${label} on GPUs ${GPU_GROUPS[$index]}"
  FEATURE_LOSS_PROFILE=no_stage1 \
  LAYER_TEMPERATURE_PROFILE=uniform \
  DRIFT_TOP_K_POS="${TOP_K_POS[$index]}" \
  DRIFT_TOP_K_NEG="${TOP_K_NEG[$index]}" \
  DRIFT_TOP_K_GROUPS=stage2,stage3,stage4 \
  REV_DRIFT_TOP_P=1.0 \
  TRAIN_SEED="${TRAIN_SEED}" \
  TRAIN_BATCH_SIZE=10 \
  POS_PER_SAMPLE=32 \
  NEG_PER_SAMPLE=16 \
  TOTAL_GENERATED_EPOCHS=30 \
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

echo "[topk-factorial] launched all four runs through 30 generated-sample epochs (step 30028)"
for label in "${LABELS[@]}"; do
  echo "[topk-factorial] tail -f ${ROOT}/runs/launch_logs/gen_B4_revdrift_mae256_topk_factorial_s234_p32_n16_${label}_${ABLATION_TAG}.log"
done
