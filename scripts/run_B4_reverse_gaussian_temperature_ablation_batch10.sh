#!/usr/bin/env bash
# Four concurrent, two-GPU Gaussian temperature/mixing ablations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/dualdrift/bin/torchrun}"
CONFIG="${CONFIG:-${ROOT}/configs/gen/B4_rev-drift_mae256.yaml}"
ABLATION_TAG="${ABLATION_TAG:-run1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
GENERATED_EPOCHS="${GENERATED_EPOCHS:-40}"
SAVE_EVERY_EPOCHS="${SAVE_EVERY_EPOCHS:-10}"
ALLOW_RESUME="${ALLOW_RESUME:-0}"
LOG_DIR="${ROOT}/runs/launch_logs"

VARIANTS=(gaussian_r045 gaussian_r075 gaussian_kernelmix_r045_r075 gaussian_fieldmix_r045_r075)
GPU_GROUPS=(0,1 2,3 4,5 6,7)
PORTS=(29910 29911 29912 29913)

mkdir -p "${LOG_DIR}"

for variant in "${VARIANTS[@]}"; do
  workdir="${ROOT}/runs/gen_B4_revdrift_mae256_full86_p128_n32_gausstemp_${variant}_${ABLATION_TAG}"
  if [[ "${ALLOW_RESUME}" != "1" && -f "${workdir}/checkpoints/ckpt_latest.pt" ]]; then
    echo "[error] existing checkpoint would be resumed: ${workdir}/checkpoints/ckpt_latest.pt"
    echo "[error] choose a new ABLATION_TAG or set ALLOW_RESUME=1"
    exit 1
  fi
done

for index in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$index]}"
  workdir="${ROOT}/runs/gen_B4_revdrift_mae256_full86_p128_n32_gausstemp_${variant}_${ABLATION_TAG}"
  run_name="$(basename "${workdir}")"
  log_file="${LOG_DIR}/${run_name}.log"
  pid_file="${LOG_DIR}/pid_${run_name}"
  mkdir -p "${workdir}"
  : > "${log_file}"

  echo "[gaussian-temperature] launching ${variant} GPUs=${GPU_GROUPS[$index]} log=${log_file}"
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
      "${ROOT}/scripts/run_B4_reverse_gaussian_temperature_ablation.py" \
      --config "${CONFIG}" \
      --workdir "${workdir}" \
      --variant "${variant}" \
      --seed "${TRAIN_SEED}" \
      --generated-epochs "${GENERATED_EPOCHS}" \
      --save-every-epochs "${SAVE_EVERY_EPOCHS}" \
      >> "${log_file}" 2>&1

  sleep 2
  pid="$(pgrep -f "run_B4_reverse_gaussian_temperature_ablation.py.*--workdir ${workdir}.*--variant ${variant}" | head -n 1 || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${pid_file}"
    echo "[gaussian-temperature] ${variant} pid=${pid}"
  else
    echo "[warn] ${variant} launched but PID auto-detection failed"
  fi
done

echo "[gaussian-temperature] all four runs launched through ${GENERATED_EPOCHS} generated epochs"
for variant in "${VARIANTS[@]}"; do
  echo "tail -f ${LOG_DIR}/gen_B4_revdrift_mae256_full86_p128_n32_gausstemp_${variant}_${ABLATION_TAG}.log"
done
