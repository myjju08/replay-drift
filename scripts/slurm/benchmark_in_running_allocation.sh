#!/usr/bin/env bash
# Temporarily pause one production process tree, benchmark on its allocated
# GPUs, then resume it. Run this only through `srun --jobid ... --overlap`.

set -euo pipefail

: "${TARGET_RUN_TOKEN:?Set TARGET_RUN_TOKEN to a unique production workdir token}"

CODE_ROOT="/shared/juhyeong/I-Drift"
PYTHON="${IDRIFT_PYTHON:-/shared/juhyeong/.venvs/idrift/bin/python}"
CONFIG="${CODE_ROOT}/configs/gen/S4_rev-drift_mae256.yaml"
BENCH_STEPS="${BENCH_STEPS:-70}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

launcher_pid="$({
  ps -u "$(id -u)" -o pid=,args= \
    | awk -v token="${TARGET_RUN_TOKEN}" \
        'index($0, "torch.distributed.run") && index($0, token) {print $1}'
} | head -n 1)"
if [[ -z "${launcher_pid}" ]]; then
  echo "Could not find the production torch launcher for ${TARGET_RUN_TOKEN}" >&2
  exit 2
fi

declare -a paused_pids=("${launcher_pid}")
declare -a frontier=("${launcher_pid}")
while ((${#frontier[@]} > 0)); do
  declare -a next_frontier=()
  for parent_pid in "${frontier[@]}"; do
    while IFS= read -r child_pid; do
      [[ -n "${child_pid}" ]] || continue
      paused_pids+=("${child_pid}")
      next_frontier+=("${child_pid}")
    done < <(pgrep -P "${parent_pid}" || true)
  done
  frontier=("${next_frontier[@]}")
done

resumed=0
resume_training() {
  if ((resumed == 0)); then
    kill -CONT "${paused_pids[@]}" 2>/dev/null || true
    resumed=1
    echo "[production-resume] token=${TARGET_RUN_TOKEN} pids=${#paused_pids[@]}" >&2
  fi
}
trap resume_training EXIT HUP INT TERM

echo "[production-pause] token=${TARGET_RUN_TOKEN} launcher=${launcher_pid} pids=${#paused_pids[@]}"
kill -STOP "${paused_pids[@]}"
sleep 2
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

run_case() {
  local name="$1"
  local opt_level="$2"
  local batch_size="$3"
  local use_sdpa="$4"
  local allow_tf32="$5"
  local cudnn_benchmark="$6"
  local workdir="${CODE_ROOT}/runs/bench-${name}-${STAMP}"
  local -a args=(
    --config "${CONFIG}"
    --workdir "${workdir}"
    --steps "${BENCH_STEPS}"
    --batch-size "${batch_size}"
    --throughput-opt-level "${opt_level}"
    --diagnostics-every-k 10
  )
  [[ "${use_sdpa}" == "1" ]] && args+=(--use-sdpa)
  [[ "${allow_tf32}" == "1" ]] && args+=(--allow-tf32)
  [[ "${cudnn_benchmark}" == "1" ]] && args+=(--cudnn-benchmark)

  echo "[case-start] name=${name} opt=${opt_level} batch=${batch_size} sdpa=${use_sdpa} tf32=${allow_tf32} cudnn=${cudnn_benchmark}"
  if "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
      "${CODE_ROOT}/scripts/benchmark_s4_throughput.py" "${args[@]}"; then
    echo "[case-complete] name=${name} workdir=${workdir}"
  else
    status=$?
    echo "[case-failed] name=${name} status=${status}" >&2
  fi
}

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

cd "${CODE_ROOT}"
run_case opt3-base 3 4 0 0 0
run_case opt3-tf32-cudnn 3 4 0 1 1
run_case opt4-manual 4 4 0 1 1
run_case opt4-sdpa 4 4 1 1 1
run_case opt4-sdpa-b8 4 8 1 1 1
run_case opt4-sdpa-b12 4 12 1 1 1

resume_training
trap - EXIT HUP INT TERM
echo "[benchmark-all-complete] stamp=${STAMP}"
