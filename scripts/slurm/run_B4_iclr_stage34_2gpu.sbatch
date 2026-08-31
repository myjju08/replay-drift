#!/usr/bin/env bash
#SBATCH --account=osilab
#SBATCH --qos=normal
#SBATCH --partition=srv09
#SBATCH --nodelist=srv09
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=2-20:00:00
#SBATCH --output=/shared/juhyeong/I-Drift-iclr-b4/logs/%x-%j.out
#SBATCH --error=/shared/juhyeong/I-Drift-iclr-b4/logs/%x-%j.err

set -euo pipefail

: "${VARIANT:?Submit with VARIANT=rev, replay_k16, replay_k64, or replay_proxy}"
case "${VARIANT}" in
  rev|replay_k16|replay_k64|replay_proxy) ;;
  *)
    echo "Invalid VARIANT=${VARIANT}" >&2
    exit 2
    ;;
esac
MAX_STEPS="${MAX_STEPS:-0}"
DISABLE_WANDB="${DISABLE_WANDB:-0}"
BATCH_SIZE="${BATCH_SIZE:-10}"
GENERATOR_REMAT="${GENERATOR_REMAT:-config}"
MAE_REMAT="${MAE_REMAT:-config}"
THROUGHPUT_OPT_LEVEL="${THROUGHPUT_OPT_LEVEL:-0}"
PROFILE_TRAIN_STEP="${PROFILE_TRAIN_STEP:-0}"
REPLAY_RATIO="${REPLAY_RATIO:-0.5}"

case "${GENERATOR_REMAT}" in config|on|off) ;; *) echo "Invalid GENERATOR_REMAT=${GENERATOR_REMAT}" >&2; exit 2 ;; esac
case "${MAE_REMAT}" in config|on|off) ;; *) echo "Invalid MAE_REMAT=${MAE_REMAT}" >&2; exit 2 ;; esac
case "${THROUGHPUT_OPT_LEVEL}" in 0|1|2|3) ;; *) echo "Invalid THROUGHPUT_OPT_LEVEL=${THROUGHPUT_OPT_LEVEL}" >&2; exit 2 ;; esac

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

case "${VARIANT}" in
  rev) run_label="rev" ;;
  replay_k16) run_label="replay-frozen-k16-h16" ;;
  replay_k64) run_label="replay-frozen-k64-h16" ;;
  replay_proxy) run_label="replay-fresh-proxy-h16" ;;
esac
RATIO_LABEL="${REPLAY_RATIO/./p}"
RUN_NAME="B4-${run_label}-rho${RATIO_LABEL}-2gpu-b${BATCH_SIZE}-gremat-${GENERATOR_REMAT}-mremat-${MAE_REMAT}-opt${THROUGHPUT_OPT_LEVEL}-stage34-normx2-p128-n32-g64-s43-e40-job${SLURM_JOB_ID}"
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
if [[ "$(hostname -s)" == "srv04" ]]; then
  # srv04 concurrently hosts independent NCCL jobs, and its free A6000 pair
  # stalls on direct P2P.  Use isolated diagnostics and the verified SHM path.
  export NCCL_RAS_ENABLE=0
  export NCCL_P2P_DISABLE=1
fi
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export WANDB_DIR="${WORK_ROOT}/runs/wandb"

cd "${CODE_ROOT}"

"${PYTHON}" - "${CACHE_PATH}" <<'PY'
import sys
from train.train_data import _HFParquetLatentDataset

dataset = _HFParquetLatentDataset(sys.argv[1], random_flip=False)
expected_samples = 1_281_167
if len(dataset) != expected_samples:
    raise RuntimeError(
        f"ImageNet latent cache is incomplete: {len(dataset)} != {expected_samples}"
    )
print(f"[data-preflight] samples={len(dataset)} first_label={dataset[0][1]}", flush=True)
PY

EXTRA_ARGS=()
if [[ "${MAX_STEPS}" -gt 0 ]]; then
  EXTRA_ARGS+=(--max-steps "${MAX_STEPS}")
fi
if [[ "${DISABLE_WANDB}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-wandb)
fi
if [[ "${PROFILE_TRAIN_STEP}" == "1" ]]; then
  EXTRA_ARGS+=(--profile-train-step)
fi

echo "[launch-2gpu] variant=${VARIANT} job=${SLURM_JOB_ID} node=$(hostname) workdir=${WORKDIR}"
echo "[slurm-gpu] job_gpus=${SLURM_JOB_GPUS:-unknown} visible=${CUDA_VISIBLE_DEVICES:-unset}"
exec "${TORCHRUN}" --standalone --nproc_per_node=2 \
  "${CODE_ROOT}/scripts/run_B4_iclr_stage34.py" \
  --variant "${VARIANT}" \
  --workdir "${WORKDIR}" \
  --cache-path "${CACHE_PATH}" \
  --mae-checkpoint "${MAE_CHECKPOINT}" \
  --batch-size "${BATCH_SIZE}" \
  --generator-remat "${GENERATOR_REMAT}" \
  --mae-remat "${MAE_REMAT}" \
  --throughput-opt-level "${THROUGHPUT_OPT_LEVEL}" \
  --replay-ratio "${REPLAY_RATIO}" \
  "${EXTRA_ARGS[@]}"
