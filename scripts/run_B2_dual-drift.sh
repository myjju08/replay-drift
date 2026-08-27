#!/usr/bin/env bash
# B/2 + MAE-640 + mix(baseline ⊕ version_b) + cfg_sample_ratio (8x RTX 5090, GPU 0..7)
#   drift_matching: mixed_baseline_versionb
#   alpha schedule: 0 → 1 over [0, total_steps] (pure version_b → pure baseline)
#   self_mask_on_raw: false
#   dense softmax coupling (top-p disabled)
#   cfg_sample_ratio: pos_base=16, pos_min=4, total_budget=32 (loader pos=16, neg=28)
#   bf16, total_steps=125114, ckpt every 12511 steps
#   wandb project: "ImageNet - B2"
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="${ROOT}/configs/gen/B2_dual-drift.yaml"
WORKDIR="${WORKDIR:-${ROOT}/runs/gen_B2_dualdrift_dense}"
LOG_DIR="${ROOT}/runs/launch_logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/$(basename "${WORKDIR}").log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/pid_$(basename "${WORKDIR}")}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29622}"
LAUNCH_MODE="${LAUNCH_MODE:-foreground}"  # foreground | background | follow
REV_DRIFT_TOP_P="${REV_DRIFT_TOP_P:-1.0}"
FWD_DRIFT_TOP_P="${FWD_DRIFT_TOP_P:-1.0}"
DRIFT_TOP_P_MIN_KEEP="${DRIFT_TOP_P_MIN_KEEP:-1}"
DRIFT_TOP_K_POS="${DRIFT_TOP_K_POS:-0}"
DRIFT_TOP_K_NEG="${DRIFT_TOP_K_NEG:-0}"

IMAGENET_PATH="${IMAGENET_PATH:-/home1/irteam/data-vol1/osilab/hojung/data/imagenet_kaggle/ILSVRC2012}"
IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-/home1/irteam/data-vol1/osilab/hojung/data/image_test/image_latents}"
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

echo "[launch] B/2 mix(baseline ⊕ version_b) + cfg_sample_ratio (8 GPU, pos16 neg28 loader, mae640)"
echo "[launch] config=${CONFIG}"
echo "[launch] workdir=${WORKDIR}"
echo "[launch] GPUs=${GPU_IDS} nproc=${NPROC_PER_NODE} port=${MASTER_PORT}"
echo "[launch] mode=${LAUNCH_MODE}"
echo "[launch] top-p rev=${REV_DRIFT_TOP_P} fwd=${FWD_DRIFT_TOP_P} min_keep=${DRIFT_TOP_P_MIN_KEEP}"
echo "[launch] top-k asymmetric pos=${DRIFT_TOP_K_POS} neg=${DRIFT_TOP_K_NEG}"
echo "[launch] cache=${IMAGENET_CACHE_PATH}"
echo "[launch] mae_checkpoint=${MAE_CKPT}"
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
  --rev_drift_top_p "${REV_DRIFT_TOP_P}"
  --fwd_drift_top_p "${FWD_DRIFT_TOP_P}"
  --drift_top_p_min_keep "${DRIFT_TOP_P_MIN_KEEP}"
  --drift_top_k_pos "${DRIFT_TOP_K_POS}"
  --drift_top_k_neg "${DRIFT_TOP_K_NEG}"
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
