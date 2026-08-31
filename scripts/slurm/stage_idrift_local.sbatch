#!/usr/bin/env bash
#SBATCH --account=osilab
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/juhyeong/I-Drift/logs/%x-%j.out
#SBATCH --error=/shared/juhyeong/I-Drift/logs/%x-%j.err
#SBATCH --open-mode=append

set -euo pipefail

SHARED_ROOT="/shared/juhyeong"
CODE_ROOT="${SHARED_ROOT}/I-Drift"
LOCAL_ROOT="${IDRIFT_LOCAL_ROOT:-/home/juhyeong}"
LOCAL_VENV="${LOCAL_ROOT}/idrift-venv"
LOCAL_DATA="${LOCAL_ROOT}/idrift-data/imagenet"
READY_FILE="${LOCAL_DATA}/.idrift-stage-ready"

mkdir -p \
  "${LOCAL_VENV}" \
  "${LOCAL_DATA}/mae_latent_256" \
  "${LOCAL_DATA}/latent_cache_256/train" \
  "${LOCAL_DATA}/latent_cache_256/val" \
  "${CODE_ROOT}/logs"

echo "[stage] node=$(hostname -s) started=$(date --iso-8601=seconds)"
echo "[stage] local_venv=${LOCAL_VENV} local_data=${LOCAL_DATA}"
df -h "${LOCAL_ROOT}"
df -i "${LOCAL_ROOT}"

echo "[stage] syncing venv"
rsync -a --partial --stats \
  "${SHARED_ROOT}/.venvs/idrift/" \
  "${LOCAL_VENV}/"

if [[ ! -x "${LOCAL_VENV}/bin/python" ]]; then
  echo "Local Python is not executable: ${LOCAL_VENV}/bin/python" >&2
  exit 2
fi
"${LOCAL_VENV}/bin/python" - <<'PY'
import einops
import torch
import torch_fidelity
import torchmetrics
import torchvision
import wandb
import yaml

print(f"[stage] local Python imports passed; torch={torch.__version__}")
PY

echo "[stage] syncing MAE checkpoint"
rsync -a --partial --stats \
  "${CODE_ROOT}/data/imagenet/mae_latent_256/ckpt_latest.pt" \
  "${LOCAL_DATA}/mae_latent_256/ckpt_latest.pt"

source_ckpt_size=$(stat -c %s \
  "${CODE_ROOT}/data/imagenet/mae_latent_256/ckpt_latest.pt")
local_ckpt_size=$(stat -c %s \
  "${LOCAL_DATA}/mae_latent_256/ckpt_latest.pt")
if [[ "${source_ckpt_size}" != "${local_ckpt_size}" ]]; then
  echo "MAE checkpoint size mismatch: shared=${source_ckpt_size} local=${local_ckpt_size}" >&2
  exit 2
fi

echo "[stage] syncing train latent cache"
rsync -a --partial --stats \
  "${CODE_ROOT}/data/imagenet/latent_cache_256/train/" \
  "${LOCAL_DATA}/latent_cache_256/train/"

echo "[stage] syncing val latent cache"
rsync -a --partial --stats \
  "${CODE_ROOT}/data/imagenet/latent_cache_256/val/" \
  "${LOCAL_DATA}/latent_cache_256/val/"

train_count=$(find "${LOCAL_DATA}/latent_cache_256/train" -type f | wc -l)
val_count=$(find "${LOCAL_DATA}/latent_cache_256/val" -type f | wc -l)
if [[ "${train_count}" -ne 1281167 ]]; then
  echo "Unexpected local train cache count: ${train_count}" >&2
  exit 2
fi
if [[ "${val_count}" -ne 50000 ]]; then
  echo "Unexpected local val cache count: ${val_count}" >&2
  exit 2
fi

printf '%s\n' \
  "node=$(hostname -s)" \
  "completed=$(date --iso-8601=seconds)" \
  "python=${LOCAL_VENV}/bin/python" \
  "data_root=${LOCAL_DATA}" \
  "train_files=${train_count}" \
  "val_files=${val_count}" \
  "mae_checkpoint_bytes=${local_ckpt_size}" \
  > "${READY_FILE}"

echo "[stage] complete"
cat "${READY_FILE}"
