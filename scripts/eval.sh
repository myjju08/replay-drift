#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-${ROOT}/configs/gen/B4_rev-drift_mae256.yaml}"


# 'dualdrift' : /home/irteam/data-vol1/osilab/hojung/dualdrift/runs/gen_B2_mix_dualdrift_mae640_officialsetup_try3/checkpoints
# 'baseline' : /home/irteam/data-vol1/osilab/hojung/dualdrift/runs/gen_B2_baseline_mae640_0726_run1/checkpoints
# /home/irteam/data-vol1/osilab/hojung/dualdrift/runs/gen_B2_dualdrift_fwdinit_full_h100k_0804_run1/checkpoints
CKPT_DIR="${CKPT_DIR:-/home/irteam/data-vol1/osilab/hojung/dualdrift/runs/gen_B4_revdrift_mae256_lossabl_p32_n16_global_x8_run1/checkpoints}"
CKPT_STEPS="${CKPT_STEPS:-0010010 0020019 0030028 0040037}"
CKPTS="${CKPTS:-}"
if [[ -z "${CKPTS}" ]]; then
  for step in ${CKPT_STEPS}; do
    CKPTS+="${CKPT_DIR}/ckpt_step_${step}.pt "
  done
fi
CFG_SCALE="${CFG_SCALE:-1.4}"
CFG_SCALES="${CFG_SCALES:-${CFG_SCALE}}"
GPU_ID_GROUPS="${GPU_ID_GROUPS:-${GPU_IDS}}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
N_SAMPLES="${N_SAMPLES:-50000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED="${SEED:-0}"
LABEL_SOURCE="${LABEL_SOURCE:-official_val}"
PR_REF_COUNT="${PR_REF_COUNT:-10000}"
REF_NPZ="${REF_NPZ:-}"
FID_REF_NPZ="${FID_REF_NPZ:-${REF_NPZ:-${ROOT}/data/eval/imagenet_256_fid_stats.npz}}"
PR_REF_NPZ="${PR_REF_NPZ:-${REF_NPZ:-${ROOT}/data/eval/imagenet_val_prc_arr0.npz}}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/eval_official_imagenet256}"
SAVE_SAMPLE_NPZ="${SAVE_SAMPLE_NPZ:-0}"
KEEP_SAMPLE_NPZ="${KEEP_SAMPLE_NPZ:-0}"
RUN_OPENAI_EVAL="${RUN_OPENAI_EVAL:-0}"
OPENAI_EVAL_PY="${OPENAI_EVAL_PY:-}"
OPENAI_EVAL_PYTHON_BIN="${OPENAI_EVAL_PYTHON_BIN:-${PYTHON_BIN}}"
OPENAI_REF_NPZ="${OPENAI_REF_NPZ:-${PR_REF_NPZ}}"

trim_spaces() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

split_space_or_comma_list() {
  local raw="$1"
  local -n out="$2"
  local item had_noglob=0
  raw="${raw//,/ }"
  out=()
  case $- in
    *f*) had_noglob=1 ;;
    *) set -f ;;
  esac
  for item in ${raw}; do
    out+=("${item}")
  done
  if [[ "${had_noglob}" == "0" ]]; then
    set +f
  fi
}

split_semicolon_list() {
  local raw="$1"
  local -n out="$2"
  local item
  local -a raw_items=()
  out=()
  IFS=';' read -r -a raw_items <<< "${raw}"
  for item in "${raw_items[@]}"; do
    item="$(trim_spaces "${item}")"
    if [[ -n "${item}" ]]; then
      out+=("${item}")
    fi
  done
}

checkpoint_tag() {
  local job_index="$1"
  local ckpt="$2"
  local stem parent_dir parent tag
  stem="$(basename "${ckpt}")"
  stem="${stem%.*}"
  parent_dir="$(dirname "${ckpt}")"
  parent="$(basename "${parent_dir}")"
  if [[ "${parent}" == "checkpoints" ]]; then
    parent="$(basename "$(dirname "${parent_dir}")")"
  fi
  if (( JOB_COUNT > 1 )); then
    printf -v tag 'job%03d_%s_%s' "${job_index}" "${parent}" "${stem}"
  else
    tag="${stem}"
  fi
  tag="${tag// /_}"
  printf '%s' "${tag}"
}

run_one_eval() {
  local job_index="$1"
  local ckpt="$2"
  local cfg_scale="$3"
  local gpu_ids="$4"
  local ckpt_tag out_json sample_npz openai_eval_log

  ckpt_tag="$(checkpoint_tag "${job_index}" "${ckpt}")"
  if [[ "${JOB_COUNT}" == "1" && -n "${OUT_JSON:-}" ]]; then
    out_json="${OUT_JSON}"
  else
    out_json="${OUT_DIR}/${ckpt_tag}_cfg${cfg_scale}_n${N_SAMPLES}.json"
  fi
  if [[ "${JOB_COUNT}" == "1" && -n "${SAMPLE_NPZ:-}" ]]; then
    sample_npz="${SAMPLE_NPZ}"
  else
    sample_npz="${out_json%.json}.samples.npz"
  fi
  if [[ "${JOB_COUNT}" == "1" && -n "${OPENAI_EVAL_LOG:-}" ]]; then
    openai_eval_log="${OPENAI_EVAL_LOG}"
  else
    openai_eval_log="${out_json%.json}.openai_eval.txt"
  fi

  echo "[eval:${job_index}] config=${CONFIG}"
  echo "[eval:${job_index}] ckpt=${ckpt}"
  echo "[eval:${job_index}] python_bin=${PYTHON_BIN}"
  "${PYTHON_BIN}" -c "import sys; print('[eval:${job_index}] python_executable=' + sys.executable)"
  echo "[eval:${job_index}] gpu_ids=${gpu_ids}"
  echo "[eval:${job_index}] cfg_scale=${cfg_scale}"
  echo "[eval:${job_index}] n_samples=${N_SAMPLES}"
  echo "[eval:${job_index}] batch_size=${BATCH_SIZE}"
  echo "[eval:${job_index}] seed=${SEED}"
  echo "[eval:${job_index}] label_source=${LABEL_SOURCE}"
  echo "[eval:${job_index}] pr_ref_count=${PR_REF_COUNT}"
  echo "[eval:${job_index}] fid_ref_npz=${FID_REF_NPZ}"
  echo "[eval:${job_index}] pr_ref_npz=${PR_REF_NPZ}"
  echo "[eval:${job_index}] out=${out_json}"
  echo "[eval:${job_index}] sample_npz=${sample_npz}"
  echo "[eval:${job_index}] save_sample_npz=${SAVE_SAMPLE_NPZ}"
  echo "[eval:${job_index}] keep_sample_npz=${KEEP_SAMPLE_NPZ}"
  echo "[eval:${job_index}] run_openai_eval=${RUN_OPENAI_EVAL}"
  echo "[eval:${job_index}] openai_eval_py=${OPENAI_EVAL_PY:-<disabled>}"
  echo "[eval:${job_index}] openai_eval_log=${openai_eval_log}"

  local cmd=(
    env
    CUDA_VISIBLE_DEVICES="${gpu_ids}"
    "${PYTHON_BIN}"
    "${ROOT}/scripts/eval_official_imagenet256.py"
    --config "${CONFIG}"
    --ckpt "${ckpt}"
    --cfg_scale "${cfg_scale}"
    --fid_ref_npz "${FID_REF_NPZ}"
    --pr_ref_npz "${PR_REF_NPZ}"
    --pr_ref_count "${PR_REF_COUNT}"
    --n_samples "${N_SAMPLES}"
    --batch_size "${BATCH_SIZE}"
    --seed "${SEED}"
    --label_source "${LABEL_SOURCE}"
    --out "${out_json}"
    --sample_npz "${sample_npz}"
  )

  if [[ "${KEEP_SAMPLE_NPZ}" == "1" || "${SAVE_SAMPLE_NPZ}" == "1" || "${RUN_OPENAI_EVAL}" == "1" ]]; then
    cmd+=(--keep_sample_npz)
  fi

  "${cmd[@]}"

  if [[ "${RUN_OPENAI_EVAL}" == "1" ]]; then
    local openai_eval_dir
    local -a openai_cmd
    openai_eval_dir="$(cd "$(dirname "${OPENAI_EVAL_PY}")" && pwd)"
    openai_cmd=(
      "${OPENAI_EVAL_PYTHON_BIN}"
      "$(basename "${OPENAI_EVAL_PY}")"
      "${OPENAI_REF_NPZ}"
      "${sample_npz}"
    )
    echo "[eval:${job_index}] running OpenAI evaluator..."
    echo "[eval:${job_index}] openai_eval_cmd=${openai_cmd[*]}"
    (
      cd "${openai_eval_dir}"
      "${openai_cmd[@]}"
    ) | tee "${openai_eval_log}"
    echo "[eval:${job_index}] OpenAI evaluator log: ${openai_eval_log}"
  fi

  if [[ "${KEEP_SAMPLE_NPZ}" != "1" && "${SAVE_SAMPLE_NPZ}" != "1" && -f "${sample_npz}" ]]; then
    rm -f "${sample_npz}"
    echo "[eval:${job_index}] removed sample archive: ${sample_npz}"
  fi
}

wait_for_next_job() {
  if ! wait -n; then
    FAILED_JOBS=1
  fi
  RUNNING_JOBS=$((RUNNING_JOBS - 1))
}

if [[ "${RUN_OPENAI_EVAL}" == "1" && -z "${OPENAI_EVAL_PY}" ]]; then
  OPENAI_EVAL_PY="${ROOT}/third_party/openai_guided_diffusion/evaluator.py"
fi

split_space_or_comma_list "${CKPTS}" CKPT_LIST
split_space_or_comma_list "${CFG_SCALES}" CFG_SCALE_LIST
split_semicolon_list "${GPU_ID_GROUPS}" GPU_GROUP_LIST

if [[ "${#CKPT_LIST[@]}" -eq 0 ]]; then
  echo "[error] set CKPT=/abs/path/to/checkpoint.pt or CKPTS='ckpt1.pt ckpt2.pt'"; exit 1
fi
if [[ "${#CFG_SCALE_LIST[@]}" -eq 0 ]]; then
  echo "[error] set CFG_SCALE=1.3 or CFG_SCALES='1.1 1.3 1.5'"; exit 1
fi
if [[ "${#GPU_GROUP_LIST[@]}" -eq 0 ]]; then
  echo "[error] set GPU_IDS=0,1 or GPU_ID_GROUPS='0;1;2;3'"; exit 1
fi

JOB_COUNT=$(( ${#CKPT_LIST[@]} * ${#CFG_SCALE_LIST[@]} ))
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_GROUP_LIST[@]}}"
if ! [[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] || (( MAX_PARALLEL < 1 )); then
  echo "[error] MAX_PARALLEL must be a positive integer, got: ${MAX_PARALLEL}"; exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] config not found: ${CONFIG}"; exit 1
fi
for ckpt in "${CKPT_LIST[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[error] checkpoint not found: ${ckpt}"; exit 1
  fi
done
if [[ ! -f "${FID_REF_NPZ}" ]]; then
  echo "[error] FID reference stats not found: ${FID_REF_NPZ}"; exit 1
fi
if [[ ! -f "${PR_REF_NPZ}" ]]; then
  echo "[error] precision/recall reference batch not found: ${PR_REF_NPZ}"; exit 1
fi
if [[ "${RUN_OPENAI_EVAL}" == "1" && ! -f "${OPENAI_EVAL_PY}" ]]; then
  echo "[error] OpenAI evaluator not found: ${OPENAI_EVAL_PY}"; exit 1
fi
if ! "${PYTHON_BIN}" -c "import diffusers" >/dev/null 2>&1; then
  echo "[error] diffusers is not importable with PYTHON_BIN=${PYTHON_BIN}"
  echo "[error] rerun with: PYTHON_BIN=/home/kaist_ghwj/miniconda/envs/dualdrift/bin/python bash scripts/eval.sh"
  exit 1
fi

mkdir -p "${OUT_DIR}"

if (( JOB_COUNT > 1 )); then
  if [[ -n "${OUT_JSON:-}" || -n "${SAMPLE_NPZ:-}" || -n "${OPENAI_EVAL_LOG:-}" ]]; then
    echo "[eval] warning: OUT_JSON/SAMPLE_NPZ/OPENAI_EVAL_LOG are ignored for multi-job runs"
  fi
fi

echo "[eval] ckpt_count=${#CKPT_LIST[@]}"
echo "[eval] cfg_scales=${CFG_SCALE_LIST[*]}"
echo "[eval] total_jobs=${JOB_COUNT}"
echo "[eval] gpu_id_groups=${GPU_GROUP_LIST[*]}"
echo "[eval] max_parallel=${MAX_PARALLEL}"

RUNNING_JOBS=0
FAILED_JOBS=0
JOB_INDEX=0

for ckpt in "${CKPT_LIST[@]}"; do
  for cfg_scale in "${CFG_SCALE_LIST[@]}"; do
    gpu_group="${GPU_GROUP_LIST[$(( JOB_INDEX % ${#GPU_GROUP_LIST[@]} ))]}"
    run_one_eval "${JOB_INDEX}" "${ckpt}" "${cfg_scale}" "${gpu_group}" &
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
    JOB_INDEX=$((JOB_INDEX + 1))
    if (( RUNNING_JOBS >= MAX_PARALLEL )); then
      wait_for_next_job
    fi
  done
done

while (( RUNNING_JOBS > 0 )); do
  wait_for_next_job
done

if [[ "${FAILED_JOBS}" != "0" ]]; then
  echo "[error] one or more eval jobs failed"
  exit 1
fi
