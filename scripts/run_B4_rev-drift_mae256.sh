#!/usr/bin/env bash
# B/4 generator + official full-resolution latent MAE-256 + reverse drift.
# Matched to run_B2_rev-drift.sh except for generator patch size and MAE width.
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/dualdrift/bin/torchrun}"

CONFIG="${CONFIG:-${ROOT}/configs/gen/B4_rev-drift_mae256.yaml}"
LAYER_TEMPERATURE_PROFILE="${LAYER_TEMPERATURE_PROFILE:-uniform}"
case "${LAYER_TEMPERATURE_PROFILE}" in
  uniform|shallow_hot|deep_sharp|depth_profile) ;;
  *)
    echo "[error] unknown LAYER_TEMPERATURE_PROFILE=${LAYER_TEMPERATURE_PROFILE}"
    echo "[error] choose: uniform, shallow_hot, deep_sharp, depth_profile"
    exit 1
    ;;
esac
FEATURE_LOSS_PROFILE="${FEATURE_LOSS_PROFILE:-all}"
case "${FEATURE_LOSS_PROFILE}" in
  all|no_global|no_norm_x|no_stage1|no_stage2|no_stage3|no_stage4|global_x8|no_stage1_norm_x2|no_stage2_norm_x2|no_stage12|no_stage12_norm_x2) ;;
  *)
    echo "[error] unknown FEATURE_LOSS_PROFILE=${FEATURE_LOSS_PROFILE}"
    echo "[error] choose: all, no_global, no_norm_x, no_stage1, no_stage2, no_stage3, no_stage4, global_x8, no_stage1_norm_x2, no_stage2_norm_x2, no_stage12, no_stage12_norm_x2"
    exit 1
    ;;
esac
DEFAULT_WORKDIR="${ROOT}/runs/gen_B4_revdrift_mae256_layerT_${LAYER_TEMPERATURE_PROFILE}"
if [[ "${FEATURE_LOSS_PROFILE}" != "all" ]]; then
  DEFAULT_WORKDIR="${DEFAULT_WORKDIR}_lossW_${FEATURE_LOSS_PROFILE}"
fi
WORKDIR="${WORKDIR:-${DEFAULT_WORKDIR}}"
RUN_NAME="${RUN_NAME:-$(basename "${WORKDIR}")}"
LOG_DIR="${ROOT}/runs/launch_logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pid_${RUN_NAME}}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29626}"
LAUNCH_MODE="${LAUNCH_MODE:-foreground}"  # foreground | background | follow
REV_DRIFT_TOP_P="${REV_DRIFT_TOP_P:-1.0}"
DRIFT_TOP_P_MIN_KEEP="${DRIFT_TOP_P_MIN_KEEP:-1}"
DRIFT_TOP_K_POS="${DRIFT_TOP_K_POS:-0}"
DRIFT_TOP_K_NEG="${DRIFT_TOP_K_NEG:-0}"
DRIFT_TOP_K_GROUPS="${DRIFT_TOP_K_GROUPS:-all}"
TRAIN_SEED="${TRAIN_SEED:--1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-0}"
POS_PER_SAMPLE="${POS_PER_SAMPLE:-0}"
NEG_PER_SAMPLE="${NEG_PER_SAMPLE:-0}"
TOTAL_GENERATED_EPOCHS="${TOTAL_GENERATED_EPOCHS:-0}"
SAVE_PER_GENERATED_EPOCHS="${SAVE_PER_GENERATED_EPOCHS:-0}"
EVAL_PER_STEP="${EVAL_PER_STEP:-0}"

IMAGENET_PATH="${IMAGENET_PATH:-/home1/irteam/data-vol1/osilab/hojung/data/imagenet_kaggle/ILSVRC2012}"
IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-/home1/irteam/data-vol1/osilab/hojung/data/image_test/image_latents}"
MAE_CKPT="${MAE_CKPT:-${ROOT}/weights/pt/mae_latent_256/ckpt_latest.pt}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] config not found: ${CONFIG}"; exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "[error] torchrun executable not found: ${TORCHRUN_BIN}"; exit 1
fi
if [[ ! -f "${MAE_CKPT}" ]]; then
  echo "[error] MAE checkpoint not found: ${MAE_CKPT}"; exit 1
fi
if [[ ! -d "${IMAGENET_CACHE_PATH}/train" || ! -d "${IMAGENET_CACHE_PATH}/val" ]]; then
  echo "[error] latent cache path invalid: ${IMAGENET_CACHE_PATH}"; exit 1
fi

mkdir -p "${WORKDIR}" "${LOG_DIR}"
: > "${LOG_FILE}"

echo "[launch] B/4 + official full-resolution latent MAE-256 + reverse drift"
echo "[launch] config=${CONFIG}"
echo "[launch] workdir=${WORKDIR}"
echo "[launch] GPUs=${GPU_IDS} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "[launch] mode=${LAUNCH_MODE}"
echo "[launch] torchrun=${TORCHRUN_BIN}"
echo "[launch] layer_temperature_profile=${LAYER_TEMPERATURE_PROFILE}"
echo "[launch] feature_loss_profile=${FEATURE_LOSS_PROFILE}"
echo "[launch] train_seed=${TRAIN_SEED} (negative keeps config)"
echo "[launch] batch=${TRAIN_BATCH_SIZE:-config} pos=${POS_PER_SAMPLE:-config} neg=${NEG_PER_SAMPLE:-config}"
echo "[launch] generated_epochs total=${TOTAL_GENERATED_EPOCHS:-config} save_every=${SAVE_PER_GENERATED_EPOCHS:-config}"
echo "[launch] eval_per_step=${EVAL_PER_STEP} (0 keeps config)"
echo "[launch] top-p rev=${REV_DRIFT_TOP_P} min_keep=${DRIFT_TOP_P_MIN_KEEP}"
echo "[launch] top-k row-wise pos=${DRIFT_TOP_K_POS} neg=${DRIFT_TOP_K_NEG} groups=${DRIFT_TOP_K_GROUPS}"
echo "[launch] cache=${IMAGENET_CACHE_PATH}"
echo "[launch] mae_checkpoint=${MAE_CKPT}"
echo "[launch] log=${LOG_FILE}"

RUN_CMD=(
  env
  PYTHONUNBUFFERED=1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IMAGENET_PATH="${IMAGENET_PATH}"
  IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH}"
  "${TORCHRUN_BIN}"
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT}/train_imagenet_gen.py"
  --config "${CONFIG}"
  --workdir "${WORKDIR}"
  --mae_checkpoint "${MAE_CKPT}"
  --seed "${TRAIN_SEED}"
  --rev_drift_top_p "${REV_DRIFT_TOP_P}"
  --drift_top_p_min_keep "${DRIFT_TOP_P_MIN_KEEP}"
  --drift_top_k_pos "${DRIFT_TOP_K_POS}"
  --drift_top_k_neg "${DRIFT_TOP_K_NEG}"
  --drift_top_k_groups "${DRIFT_TOP_K_GROUPS}"
  --layer_temperature_profile "${LAYER_TEMPERATURE_PROFILE}"
  --feature_loss_profile "${FEATURE_LOSS_PROFILE}"
  --batch_size "${TRAIN_BATCH_SIZE}"
  --pos_per_sample "${POS_PER_SAMPLE}"
  --neg_per_sample "${NEG_PER_SAMPLE}"
  --total_generated_epochs "${TOTAL_GENERATED_EPOCHS}"
  --save_per_generated_epochs "${SAVE_PER_GENERATED_EPOCHS}"
  --eval_per_step "${EVAL_PER_STEP}"
)

if [[ "${LAUNCH_MODE}" == "foreground" ]]; then
  "${RUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  exit "${PIPESTATUS[0]}"
fi

setsid -f "${RUN_CMD[@]}" >> "${LOG_FILE}" 2>&1
sleep 2

PID="$(pgrep -f "torchrun.*--master_port=${MASTER_PORT}.*${ROOT}/train_imagenet_gen.py.*--config ${CONFIG}.*--workdir ${WORKDIR}" | head -n 1 || true)"
if [[ -z "${PID}" ]]; then
  PID="$(pgrep -f "${ROOT}/train_imagenet_gen.py.*--config ${CONFIG}.*--workdir ${WORKDIR}" | head -n 1 || true)"
fi
if [[ -n "${PID}" ]]; then
  echo "${PID}" > "${PID_FILE}"
  echo "[launch] pid=${PID}"
  echo "[launch] pid file=${PID_FILE}"
else
  echo "[warn] detached launch completed, but PID auto-detect failed."
  echo "[warn] monitor with: pgrep -fa \"${ROOT}/train_imagenet_gen.py\""
fi

if [[ "${LAUNCH_MODE}" == "follow" ]]; then
  echo "[launch] following ${LOG_FILE} (Ctrl-C stops log following only)"
  exec tail -n 50 -f "${LOG_FILE}"
fi

echo "[launch] tail -f ${LOG_FILE}"
