#!/usr/bin/env bash
#SBATCH --account=osilab
#SBATCH --qos=normal
#SBATCH --partition=srv09
#SBATCH --nodelist=srv09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=3-00:00:00
#SBATCH --output=/shared/juhyeong/I-Drift-iclr-b4/logs/%x-%j.out
#SBATCH --error=/shared/juhyeong/I-Drift-iclr-b4/logs/%x-%j.err

set -euo pipefail

: "${VARIANT:?Submit with --export=ALL,VARIANT=rev or replay}"
if [[ "${VARIANT}" != "rev" && "${VARIANT}" != "replay" ]]; then
  echo "VARIANT must be rev or replay, got: ${VARIANT}" >&2
  exit 2
fi
MAX_STEPS="${MAX_STEPS:-0}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"

CODE_ROOT="/home/juhyeong/I-Drift"
WORK_ROOT="/shared/juhyeong/I-Drift-iclr-b4"
ASSET_ROOT="/home/juhyeong/I-Drift-assets"
PYTHON="/home/juhyeong/.venvs/idrift/bin/python"
TORCHRUN="/home/juhyeong/.venvs/idrift/bin/torchrun"
CACHE_PATH="${ASSET_ROOT}/data/hf_imagenet_latents"
MAE_CHECKPOINT="${ASSET_ROOT}/weights/mae_latent_256/ckpt_latest.pt"

for required in "${CODE_ROOT}" "${WORK_ROOT}" "${PYTHON}" "${TORCHRUN}" "${CACHE_PATH}" "${MAE_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

RUN_NAME="B4-${VARIANT}-stage34-normx2-p128-n32-g64-s43-e40-job${SLURM_JOB_ID}"
if [[ "${VARIANT}" == "replay" ]]; then
  RUN_NAME="B4-replay-rho050-h16-e10-stage34-normx2-p128-n32-g64-s43-e40-job${SLURM_JOB_ID}"
fi
if [[ "${MAX_STEPS}" -gt 0 ]]; then
  RUN_NAME="smoke-${RUN_NAME}-max${MAX_STEPS}"
fi
WORKDIR="${WORK_ROOT}/runs/${RUN_NAME}"
mkdir -p "${WORKDIR}" "${WORK_ROOT}/runs/wandb"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export WANDB_DIR="${WORK_ROOT}/runs/wandb"

cd "${CODE_ROOT}"

# Materialize/validate the Arrow cache once before four DDP workers open it.
"${PYTHON}" - "${CACHE_PATH}" <<'PY'
import sys
from train.train_data import _HFParquetLatentDataset

dataset = _HFParquetLatentDataset(sys.argv[1], random_flip=False)
print(f"[data-preflight] samples={len(dataset)} first_label={dataset[0][1]}", flush=True)
PY

echo "[launch] variant=${VARIANT} job=${SLURM_JOB_ID} node=$(hostname) workdir=${WORKDIR}"
EXTRA_ARGS=()
if [[ "${MAX_STEPS}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max-steps "${MAX_STEPS}")
fi
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-wandb)
fi
exec "${TORCHRUN}" --standalone --nproc_per_node=4 \
  "${CODE_ROOT}/scripts/run_B4_iclr_stage34.py" \
  --variant "${VARIANT}" \
  --workdir "${WORKDIR}" \
  --cache-path "${CACHE_PATH}" \
  --mae-checkpoint "${MAE_CHECKPOINT}" \
  "${EXTRA_ARGS[@]}"
