#!/usr/bin/env bash
# Four concurrent two-GPU causal controls forked from one epoch-10 checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/dualdrift/bin/torchrun}"
CONFIG="${CONFIG:-${ROOT}/configs/gen/B4_rev-drift_mae256.yaml}"
ABLATION_TAG="${ABLATION_TAG:-common10_seed43_run1}"
TRAIN_SEED="${TRAIN_SEED:-43}"
HISTORY_COUNT="${HISTORY_COUNT:-16}"
HISTORY_START_EPOCH="${HISTORY_START_EPOCH:-10}"
GENERATED_EPOCHS="${GENERATED_EPOCHS:-40}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
SOURCE_RUN="${SOURCE_RUN:-${ROOT}/runs/gen_B4_revdrift_mae256_full86_p128_n32_gauss075_history_rho010_h16_e10_seed43_run1}"
SOURCE_CKPT="${SOURCE_CKPT:-${SOURCE_RUN}/checkpoints/ckpt_step_0010010.pt}"
LOG_DIR="${ROOT}/runs/launch_logs"

VARIANTS=(a_baseline b_weakcurrent c_freshanchors d_history)
GPU_GROUPS=(0,1 2,3 4,5 6,7)
PORTS=(29950 29951 29952 29953)

[[ -f "${SOURCE_CKPT}" ]] || { echo "[error] missing shared checkpoint: ${SOURCE_CKPT}"; exit 1; }
for rank in 00 01; do
  source_bank="${SOURCE_RUN}/historical_gen_replay_rank${rank}.npz"
  [[ -f "${source_bank}" ]] || { echo "[error] missing shared replay bank: ${source_bank}"; exit 1; }
done

mkdir -p "${LOG_DIR}"

# Refuse to mix this publication run with any previous continuation.
for variant in "${VARIANTS[@]}"; do
  workdir="${ROOT}/runs/gen_B4_revdrift_mae256_history_causal_${variant}_${ABLATION_TAG}"
  if [[ -e "${workdir}/checkpoints/ckpt_latest.pt" ]]; then
    echo "[error] target already contains a checkpoint: ${workdir}"
    exit 1
  fi
done

source_hash="$(sha256sum "${SOURCE_CKPT}" | awk '{print $1}')"
echo "[history-causal] shared_checkpoint=${SOURCE_CKPT}"
echo "[history-causal] shared_checkpoint_sha256=${source_hash}"

for variant in "${VARIANTS[@]}"; do
  workdir="${ROOT}/runs/gen_B4_revdrift_mae256_history_causal_${variant}_${ABLATION_TAG}"
  mkdir -p "${workdir}/checkpoints"
  cp --reflink=auto "${SOURCE_CKPT}" "${workdir}/checkpoints/ckpt_latest.pt"
  copied_hash="$(sha256sum "${workdir}/checkpoints/ckpt_latest.pt" | awk '{print $1}')"
  [[ "${copied_hash}" == "${source_hash}" ]] || { echo "[error] checkpoint hash mismatch: ${variant}"; exit 1; }
  printf '%s  %s\n' "${source_hash}" "${SOURCE_CKPT}" > "${workdir}/shared_epoch10_source.sha256"
done

# D alone consumes the frozen epoch-10 anchors. A-C still share its exact model,
# EMA, and optimizer checkpoint.
d_workdir="${ROOT}/runs/gen_B4_revdrift_mae256_history_causal_d_history_${ABLATION_TAG}"
for rank in 00 01; do
  cp --reflink=auto \
    "${SOURCE_RUN}/historical_gen_replay_rank${rank}.npz" \
    "${d_workdir}/historical_gen_replay_rank${rank}.npz"
done

for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  workdir="${ROOT}/runs/gen_B4_revdrift_mae256_history_causal_${variant}_${ABLATION_TAG}"
  run_name="$(basename "${workdir}")"
  log_file="${LOG_DIR}/${run_name}.log"
  pid_file="${LOG_DIR}/pid_${run_name}"
  : > "${log_file}"

  echo "[history-causal] launching ${variant} GPUs=${GPU_GROUPS[$index]} log=${log_file}"
  setsid -f env \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_VISIBLE_DEVICES="${GPU_GROUPS[$index]}" \
    "${TORCHRUN_BIN}" \
      --nproc_per_node=2 \
      --master_port="${PORTS[$index]}" \
      "${ROOT}/scripts/run_B4_reverse_history_causal_ablation.py" \
      --config "${CONFIG}" \
      --workdir "${workdir}" \
      --variant "${variant}" \
      --seed "${TRAIN_SEED}" \
      --history-count "${HISTORY_COUNT}" \
      --history-start-epoch "${HISTORY_START_EPOCH}" \
      --generated-epochs "${GENERATED_EPOCHS}" \
      --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
      --disable-wandb \
      >> "${log_file}" 2>&1

  sleep 2
  pid="$(pgrep -f "run_B4_reverse_history_causal_ablation.py.*--workdir ${workdir}.*--variant ${variant}" | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${pid_file}"
    echo "[history-causal] ${variant} pid=${pid}"
  else
    echo "[warn] ${variant} launched but PID auto-detection failed"
  fi
done

echo "[history-causal] all four continuations launched through ${GENERATED_EPOCHS} generated epochs"
for variant in "${VARIANTS[@]}"; do
  echo "tail -f ${LOG_DIR}/gen_B4_revdrift_mae256_history_causal_${variant}_${ABLATION_TAG}.log"
done
