#!/usr/bin/env bash
# B/2 + MAE-640 + version_b + cfg_sample_ratio + TEMP×2 (8x RTX 5090, GPU 0..7)
#   Identical to the cfgratio config except R_list = [0.4, 0.10, 0.04] (×2).
#   loader pos=16, neg=32; weights all 1; CFG influence via per-sample count.
#   total_steps=125114, ckpt every 12511 steps. wandb project: "ImageNet - B2".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${ROOT}/configs/gen/B2_fwd-drift.yaml"
WORKDIR="${WORKDIR:-${ROOT}/runs/gen_B2_version_b_cfgratio_temp2x_mae640_ncls8_gpu01234567_neg32}"
LOG_DIR="${ROOT}/runs/launch_logs"
LOG_FILE="${LOG_DIR}/B2_version_b_cfgratio_temp2x_mae640_ncls8_gpu01234567_neg32.log"
PID_FILE="${LOG_DIR}/pid_B2_version_b_cfgratio_temp2x_mae640_ncls8_gpu01234567_neg32"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29622}"
LAUNCH_MODE="${LAUNCH_MODE:-foreground}"  # foreground | background | follow

IMAGENET_PATH="${IMAGENET_PATH:-${ROOT}/data/imagenet/ILSVRC2012}"
IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-${ROOT}/data/imagenet/latent_cache_sd_vae_mse}"
MAE_CKPT="${MAE_CKPT:-${ROOT}/weights/pt/mae_latent_640/ckpt_latest.pt}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] config not found: ${CONFIG}"; exit 1
fi
if [[ ! -f "${MAE_CKPT}" ]]; then
  echo "[error] MAE checkpoint not found: ${MAE_CKPT}"; exit 1
fi
if [[ ! -d "${IMAGENET_CACHE_PATH}/train" || ! -d "${IMAGENET_CACHE_PATH}/val" ]]; then
  echo "[error] latent cache path invalid: ${IMAGENET_CACHE_PATH}"; exit 1
fi

mkdir -p "${WORKDIR}" "${LOG_DIR}"
: > "${LOG_FILE}"

echo "[launch] B/2 version_b + cfg_sample_ratio + TEMP×2 (8 GPU, mae640)"
echo "[launch] config=${CONFIG}"
echo "[launch] workdir=${WORKDIR}"
echo "[launch] GPUs=${GPU_IDS} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "[launch] mode=${LAUNCH_MODE}"
echo "[launch] cache=${IMAGENET_CACHE_PATH}"
echo "[launch] mae_checkpoint=${MAE_CKPT}"
echo "[launch] R_list=[0.4, 0.10, 0.04]  (temperature ×2)"
echo "[launch] wandb project=ImageNet - B2"
echo "[launch] log=${LOG_FILE}"

RUN_CMD=(
  env
  PYTHONUNBUFFERED=1
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  IMAGENET_PATH="${IMAGENET_PATH}"
  IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH}"
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${ROOT}/train_imagenet_gen.py"
  --config "${CONFIG}"
  --workdir "${WORKDIR}"
  --mae_checkpoint "${MAE_CKPT}"
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
