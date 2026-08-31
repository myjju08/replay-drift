#!/usr/bin/env bash
#SBATCH --account=osilab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/shared/juhyeong/I-Drift/logs/%x-%j.out
#SBATCH --error=/shared/juhyeong/I-Drift/logs/%x-%j.err

set -euo pipefail

: "${BENCH_NAME:?Set BENCH_NAME}"
: "${THROUGHPUT_OPT_LEVEL:?Set THROUGHPUT_OPT_LEVEL}"

CODE_ROOT="/shared/juhyeong/I-Drift"
PYTHON="${IDRIFT_PYTHON:-/shared/juhyeong/.venvs/idrift/bin/python}"
CONFIG="${CODE_ROOT}/configs/gen/S4_rev-drift_mae256.yaml"
WORKDIR="${CODE_ROOT}/runs/bench-${BENCH_NAME}-${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export WANDB_MODE=disabled
export TORCH_HOME="${IDRIFT_TORCH_HOME:-/shared/juhyeong/.cache/torch}"
export HF_HOME="${IDRIFT_HF_HOME:-/shared/juhyeong/.cache/huggingface}"

ARGS=(
  --config "${CONFIG}"
  --workdir "${WORKDIR}"
  --steps "${BENCH_STEPS:-80}"
  --batch-size "${BENCH_BATCH_SIZE:-4}"
  --throughput-opt-level "${THROUGHPUT_OPT_LEVEL}"
  --diagnostics-every-k "${DIAGNOSTICS_EVERY_K:-10}"
  --cudnn-benchmark-limit "${CUDNN_BENCHMARK_LIMIT:-10}"
)
if [[ "${USE_SDPA:-0}" == "1" ]]; then
  ARGS+=(--use-sdpa)
fi
if [[ "${CUDNN_BENCHMARK:-0}" == "1" ]]; then
  ARGS+=(--cudnn-benchmark)
fi
if [[ "${ALLOW_TF32:-0}" == "1" ]]; then
  ARGS+=(--allow-tf32)
fi

echo "[benchmark-launch] name=${BENCH_NAME} node=$(hostname -s) gpus=${CUDA_VISIBLE_DEVICES:-unset} args=${ARGS[*]}"
cd "${CODE_ROOT}"
exec "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "${CODE_ROOT}/scripts/benchmark_s4_throughput.py" "${ARGS[@]}"
