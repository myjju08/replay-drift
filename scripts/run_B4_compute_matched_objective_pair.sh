#!/usr/bin/env bash
# Two concurrent dense, full-feature, two-GPU objective-comparison runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/dualdrift/bin/torchrun}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/dualdrift/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/gen/B4_dual-drift_mae256.yaml}"
RUN_TAG="${RUN_TAG:-run1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
GENERATED_EPOCHS="${GENERATED_EPOCHS:-40}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
LOG_DIR="${ROOT}/runs/launch_logs"

VARIANTS=(dual3x3 reverse6)
GPU_GROUPS=(0,1 2,3)
PORTS=(29850 29851)

mkdir -p "${LOG_DIR}"
for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  workdir="${ROOT}/runs/gen_B4_mae256_full86_p128_n32_compute_matched_${variant}_${RUN_TAG}"
  run_name="$(basename "${workdir}")"
  log_file="${LOG_DIR}/${run_name}.log"
  pid_file="${LOG_DIR}/pid_${run_name}"

  if [[ "${ALLOW_RESUME}" != "1" && -f "${workdir}/checkpoints/ckpt_latest.pt" ]]; then
    echo "[error] existing checkpoint would be resumed: ${workdir}/checkpoints/ckpt_latest.pt"
    exit 1
  fi
  mkdir -p "${workdir}"
  : > "${log_file}"

  echo "[compute-match] launching ${variant} GPUs=${GPU_GROUPS[$index]} log=${log_file}"
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
      "${ROOT}/scripts/run_B4_compute_matched_objective.py" \
      --config "${CONFIG}" \
      --workdir "${workdir}" \
      --variant "${variant}" \
      --seed "${TRAIN_SEED}" \
      --generated-epochs "${GENERATED_EPOCHS}" \
      --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
      >> "${log_file}" 2>&1

  sleep 2
  pid="$(pgrep -f "run_B4_compute_matched_objective.py.*--workdir ${workdir}.*--variant ${variant}" | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${pid_file}"
    echo "[compute-match] ${variant} pid=${pid}"
  else
    echo "[warn] ${variant} launched but PID auto-detection failed"
  fi
done

echo "[compute-match] both runs launched in background"
for variant in "${VARIANTS[@]}"; do
  echo "tail -f ${LOG_DIR}/gen_B4_mae256_full86_p128_n32_compute_matched_${variant}_${RUN_TAG}.log"
done
