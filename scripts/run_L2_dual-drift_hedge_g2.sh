#!/usr/bin/env bash
# L/2 + MAE-640 + mixed_baseline_versionb + ADAPTIVE hedge (preset 1 = balanced, γ=2)
#   gamma=2.0, eta=1e-4, decay=1e-4, gamma_warmup_steps=10000
#   discounted hedge → finite memory ≈ 10000 steps, ℓ auto-bounded
#   8x RTX 5090, GPU 0..7. wandb project: "ImageNet - L2".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${ROOT}/configs/gen/L2_dual-drift_hedge_g2.yaml"
WORKDIR="${WORKDIR:-${ROOT}/runs/gen_L2_mix_adaptive_hedge_g2_mae640_ncls8_gpu01234567_neg32}"
LOG_DIR="${ROOT}/runs/launch_logs"
LOG_FILE="${LOG_DIR}/L2_mix_adaptive_hedge_g2_mae640_ncls8_gpu01234567_neg32.log"
PID_FILE="${LOG_DIR}/pid_L2_mix_adaptive_hedge_g2_mae640_ncls8_gpu01234567_neg32"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29641}"
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

echo "[launch] L/2 mixed (adaptive hedge: γ=2.0, η=1e-4, decay=1e-4, γ_warmup=10000)"
echo "[launch] config=${CONFIG}"
echo "[launch] workdir=${WORKDIR}"
echo "[launch] GPUs=${GPU_IDS} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "[launch] mode=${LAUNCH_MODE}"
echo "[launch] cache=${IMAGENET_CACHE_PATH}"
echo "[launch] mae_checkpoint=${MAE_CKPT}"
echo "[launch] wandb project=ImageNet - L2"
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
