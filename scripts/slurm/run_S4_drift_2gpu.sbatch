#!/usr/bin/env bash
#SBATCH --account=osilab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --output=/shared/juhyeong/I-Drift/logs/%x-%j.out
#SBATCH --error=/shared/juhyeong/I-Drift/logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail

: "${CONFIG_BASENAME:?Set CONFIG_BASENAME to one of the S4 YAML filenames}"
: "${RUN_NAME:?Set RUN_NAME to a new, unique run directory name}"

CODE_ROOT="/shared/juhyeong/I-Drift"
DATA_ROOT="${IDRIFT_DATA_ROOT:-${CODE_ROOT}/data/imagenet}"
CONFIG_PATH="${CODE_ROOT}/configs/gen/${CONFIG_BASENAME}"
WORKDIR="${CODE_ROOT}/runs/${RUN_NAME}"
MAE_CHECKPOINT="${DATA_ROOT}/mae_latent_256/ckpt_latest.pt"
CACHE_PATH="${IDRIFT_CACHE_PATH:-${DATA_ROOT}/latent_cache_256}"

PYTHON="${IDRIFT_PYTHON:-/shared/juhyeong/.venvs/idrift/bin/python}"

for required in \
  "${CODE_ROOT}/train_imagenet_gen.py" \
  "${CONFIG_PATH}" \
  "${MAE_CHECKPOINT}" \
  "${CACHE_PATH}/train" \
  "${CACHE_PATH}/val" \
  "${PYTHON}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required path: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - <<'PY'
import torch_fidelity  # noqa: F401
import torchmetrics  # noqa: F401
PY

# Fail before allocating a run directory if a matched S/4 pre-opt job was
# accidentally pointed at an optimized model path or early-stage features.
IDRIFT_CONFIG_PATH="${CONFIG_PATH}" \
IDRIFT_EXPECTED_CACHE_PATH="${CACHE_PATH}" \
"${PYTHON}" - <<'PY'
import os
import sys

code_root = "/shared/juhyeong/I-Drift"
sys.path.insert(0, code_root)
from train_imagenet_gen import load_yaml_config

cfg = load_yaml_config(os.environ["IDRIFT_CONFIG_PATH"])
raw = cfg["_raw"]
expected = {
    "batch_size": 4,
    "pos_per_sample": 32,
    "neg_per_sample": 32,
    "gen_per_label": 32,
    "throughput_opt_level": 3,
}
for key, value in expected.items():
    if cfg.get(key) != value:
        raise SystemExit(f"pre-opt S/4 preflight failed: {key}={cfg.get(key)!r}, expected {value!r}")
model = raw["model"]
for key, value in {
    "hidden_size": 384,
    "depth": 12,
    "num_heads": 6,
    "patch_size": 4,
    "use_sdpa": False,
    "use_remat": False,
}.items():
    if model.get(key) != value:
        raise SystemExit(f"pre-opt S/4 preflight failed: model.{key}={model.get(key)!r}, expected {value!r}")
if cfg.get("allow_tf32", False) or cfg.get("cudnn_benchmark", False):
    raise SystemExit("pre-opt S/4 preflight failed: TF32/cuDNN benchmark must be disabled")
if list(cfg.get("activation_kwargs", {}).get("active_stages", [])) != ["stage3", "stage4"]:
    raise SystemExit("pre-opt S/4 preflight failed: active_stages must be [stage3, stage4]")
profile = cfg.get("feature_loss_profiles", {}).get(cfg.get("feature_loss_profile"), {})
if float(profile.get("stage1", 1.0)) != 0.0 or float(profile.get("stage2", 1.0)) != 0.0:
    raise SystemExit("pre-opt S/4 preflight failed: stage1/stage2 loss weights must both be zero")
if not bool(cfg.get("prune_skipped_feature_tensors", False)):
    raise SystemExit("pre-opt S/4 preflight failed: skipped feature tensors must be pruned")
if cfg.get("cache_path") != os.environ["IDRIFT_EXPECTED_CACHE_PATH"]:
    raise SystemExit(
        f"cache mismatch: config={cfg.get('cache_path')!r}, "
        f"launcher={os.environ['IDRIFT_EXPECTED_CACHE_PATH']!r}"
    )
if not bool(cfg.get("use_wandb", False)):
    raise SystemExit("pre-opt S/4 preflight failed: W&B logging is disabled")
if bool(cfg.get("feature_adapter", False)):
    objective = cfg.get("feature_adapter_objective")
    if objective == "gen_real_multipos_infonce":
        adapter_expected = {
            "feature_adapter_objective": "gen_real_multipos_infonce",
            "feature_adapter_samples_per_class": 32,
            "feature_adapter_generated_samples_per_class": 32,
            "feature_adapter_generated_anchor_weight": 1.0,
            "feature_adapter_real_anchor_weight": 0.0,
            "feature_adapter_global_negatives": False,
            "feature_adapter_require_distinct_real": True,
            "feature_adapter_reg_lambda": 0.0,
            "feature_adapter_weight_decay": 0.0,
            "feature_adapter_ema_decay": 0.0,
        }
    elif objective == "real_anchor_multipos_infonce":
        adapter_expected = {
            "feature_adapter_objective": "real_anchor_multipos_infonce",
            "feature_adapter_samples_per_class": 32,
            "feature_adapter_generated_samples_per_class": 32,
            "feature_adapter_generated_anchor_weight": 0.0,
            "feature_adapter_real_anchor_weight": 1.0,
            "feature_adapter_global_negatives": False,
            "feature_adapter_require_distinct_real": True,
            "feature_adapter_reg_lambda": 0.0,
            "feature_adapter_weight_decay": 0.0,
            "feature_adapter_ema_decay": 0.0,
        }
    elif objective == "raw_drift_snr":
        adapter_expected = {
            "feature_adapter_objective": "raw_drift_snr",
            "feature_adapter_samples_per_class": 32,
            "feature_adapter_generated_samples_per_class": 32,
            "feature_adapter_drift_query_count": 8,
            "feature_adapter_drift_bank_count": 12,
            "feature_adapter_global_negatives": False,
            "feature_adapter_require_distinct_real": True,
            "feature_adapter_reg_lambda": 0.0,
            "feature_adapter_weight_decay": 0.0,
            "feature_adapter_lr": 0.00002,
            "feature_adapter_update_freq": 4,
            "feature_adapter_ema_decay": 0.999,
            "feature_adapter_drift_consistency_weight": 0.25,
            "feature_adapter_drift_variance_epsilon": 0.000001,
            "feature_adapter_drift_global_scale_stats": False,
            "feature_adapter_freeze_after_generated_epochs": 30.0,
        }
        if int(cfg.get("drift_top_k_pos", 0)) != 0 or int(cfg.get("drift_top_k_neg", 0)) != 0:
            raise SystemExit("raw-drift adapter preflight failed: dense top-k=0 is required")
    else:
        raise SystemExit(f"adapter preflight failed: unsupported objective={objective!r}")
    for key, value in adapter_expected.items():
        if cfg.get(key) != value:
            raise SystemExit(
                f"adapter preflight failed: {key}={cfg.get(key)!r}, expected {value!r}"
            )
    if list(cfg.get("feature_adapter_keys", [])) != ["layer3", "layer4"]:
        raise SystemExit("adapter preflight failed: keys must be [layer3, layer4]")
print(
    "[preflight] matched S/4 pre-opt: B4 P32 N32 G32, manual attention, "
    "TF32=false, cuDNN benchmark=false, drift features=stage3+stage4, W&B=true",
    flush=True,
)
PY

# Require the committed DDP synchronization, active-stage extraction, and
# throughput implementation while rejecting uncommitted runtime-code edits.
# Configs and this launcher may be added/edited to define a new experiment;
# the imported Python modules must match the recorded source HEAD exactly.
RUNTIME_SOURCE_FILES=(
  "drifting_core/imagenet_loss.py"
  "memory_bank.py"
  "models/imagenet_generator.py"
  "models/feature_adapter.py"
  "models/mae_resnet.py"
  "train/train_data.py"
  "train_imagenet_gen.py"
)
if ! git -C "${CODE_ROOT}" diff --quiet -- "${RUNTIME_SOURCE_FILES[@]}" || \
   ! git -C "${CODE_ROOT}" diff --cached --quiet -- "${RUNTIME_SOURCE_FILES[@]}"; then
  echo "Uncommitted runtime source changes in I-Drift." >&2
  git -C "${CODE_ROOT}" status --short -- "${RUNTIME_SOURCE_FILES[@]}" >&2
  exit 2
fi
if ! grep -Fq 'out = generator(expanded_labels, cfg_scale=expanded_cfg, train=True)' \
  "${CODE_ROOT}/train_imagenet_gen.py"; then
  echo "Required DDP synchronization fix is missing." >&2
  exit 2
fi
if ! grep -Fq 'fuse_fnorm_across_R = throughput_opt_level >= 4' \
  "${CODE_ROOT}/train_imagenet_gen.py"; then
  echo "Required committed throughput implementation is missing." >&2
  exit 2
fi

mkdir -p "${WORKDIR}" "${CODE_ROOT}/logs" "${CODE_ROOT}/wandb"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export TORCH_HOME="${IDRIFT_TORCH_HOME:-/shared/juhyeong/.cache/torch}"
export HF_HOME="${IDRIFT_HF_HOME:-/shared/juhyeong/.cache/huggingface}"
if [[ "$(hostname -s)" == "srv04" ]]; then
  export NCCL_RAS_ENABLE=0
  export NCCL_P2P_DISABLE=1
fi
export WANDB_DIR="${CODE_ROOT}/wandb"

echo "[launch] job=${SLURM_JOB_ID} node=$(hostname) config=${CONFIG_PATH}"
echo "[launch] run=${RUN_NAME} workdir=${WORKDIR} source_head=$(git -C "${CODE_ROOT}" rev-parse HEAD)"
echo "[launch] data_root=${DATA_ROOT}"
echo "[launch] cache_path=${CACHE_PATH}"
echo "[launch] visible_gpus=${CUDA_VISIBLE_DEVICES:-unset} job_gpus=${SLURM_JOB_GPUS:-unknown}"
echo "[launch] config_sha256=$(sha256sum "${CONFIG_PATH}" | awk '{print $1}')"

cd "${CODE_ROOT}"
TRAIN_ARGS=()
if [[ -n "${THROUGHPUT_OPT_LEVEL_OVERRIDE:-}" ]]; then
  TRAIN_ARGS+=(--throughput_opt_level "${THROUGHPUT_OPT_LEVEL_OVERRIDE}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
  TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
fi
exec "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2 \
  "${CODE_ROOT}/train_imagenet_gen.py" \
  --config "${CONFIG_PATH}" \
  --workdir "${WORKDIR}" \
  "${TRAIN_ARGS[@]}"
