#!/usr/bin/env python3
"""Fetch and convert official Drifting MAE JAX checkpoints to PyTorch.

Examples:
  python scripts/convert_mae_hf_jax_to_torch.py \
    --model-id mae_latent_640 \
    --output /data/imagenet/mae_latent_640/ckpt_latest.pt

  python scripts/convert_mae_hf_jax_to_torch.py \
    --artifact-dir weights/hf/models/mae/jax/mae_latent_640 \
    --output weights/pt/mae_latent_640/ckpt_latest.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    import msgpack
except ImportError as exc:  # pragma: no cover - handled at runtime
    msgpack = None
    _MSGPACK_IMPORT_ERROR = exc
else:
    _MSGPACK_IMPORT_ERROR = None

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled at runtime
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import torch
except ImportError as exc:  # pragma: no cover - handled at runtime
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OFFICIAL_REPO_ID = "Goodeat/drifting"
OFFICIAL_REVISION = "main"
OFFICIAL_ARTIFACT_PREFIX = "models/mae/jax"
DEFAULT_MODEL_ID = "mae_latent_640"
ARTIFACT_FILENAMES = ("metadata.json", "ema_params.msgpack")
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_EVERY_BYTES = 256 * 1024 * 1024
MSGPACK_CHUNK_MARKER = "__msgpack_chunked_array__"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required to convert the checkpoint. "
            "Install the repo requirements first, then rerun this script."
        ) from _TORCH_IMPORT_ERROR


def _require_conversion_deps() -> None:
    if np is None:
        raise RuntimeError(
            "NumPy is required to convert the checkpoint. "
            "Install the repo requirements first, then rerun this script."
        ) from _NUMPY_IMPORT_ERROR
    if msgpack is None:
        raise RuntimeError(
            "msgpack is required to convert the checkpoint. "
            "Install the repo requirements first, then rerun this script."
        ) from _MSGPACK_IMPORT_ERROR


def _load_mae_builder():
    _require_torch()
    from models.mae_resnet import build_mae_from_config

    return build_mae_from_config


def _flatten_mapping(tree: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in tree.items():
        key_str = str(key)
        next_prefix = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, Mapping):
            out.update(_flatten_mapping(value, next_prefix))
        else:
            out[next_prefix] = value
    return out


def _msgpack_ext_unpack(code: int, data: bytes) -> Any:
    if code == 1:  # ndarray
        shape, dtype_name, buffer = msgpack.unpackb(data, raw=False)
        dtype_name = str(dtype_name)
        shape = tuple(int(v) for v in shape)
        if dtype_name == "bfloat16":
            bf16 = np.frombuffer(buffer, dtype=np.uint16).astype(np.uint32)
            arr = (bf16 << 16).view(np.float32)
            return arr.reshape(shape)
        return np.frombuffer(buffer, dtype=np.dtype(dtype_name)).reshape(shape)
    if code == 2:  # native complex
        complex_tuple = msgpack.unpackb(data, raw=False)
        return complex(*complex_tuple)
    if code == 3:  # numpy scalar
        ar = _msgpack_ext_unpack(1, data)
        return ar[()]
    return msgpack.ExtType(code, data)


def _unchunk_array_leaves(data: Any) -> Any:
    if isinstance(data, dict):
        if MSGPACK_CHUNK_MARKER in data:
            flat = np.concatenate(
                [np.asarray(chunk).reshape(-1) for chunk in data[MSGPACK_CHUNK_MARKER]],
                axis=0,
            )
            return flat.reshape(tuple(int(v) for v in data["shape"]))
        return {k: _unchunk_array_leaves(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_unchunk_array_leaves(v) for v in data]
    if isinstance(data, tuple):
        return tuple(_unchunk_array_leaves(v) for v in data)
    return data


def _msgpack_restore(path: Path) -> Any:
    restored = msgpack.unpackb(
        path.read_bytes(),
        ext_hook=_msgpack_ext_unpack,
        raw=False,
        strict_map_key=False,
    )
    return _unchunk_array_leaves(restored)


def _hf_resolve_url(repo_id: str, revision: str, relative_path: str) -> str:
    quoted = urllib.parse.quote(relative_path, safe="/")
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{quoted}"


def _download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".tmp")
    print(f"[download] {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "dualdrift-mae-converter/1.0"})
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        next_report = PROGRESS_EVERY_BYTES
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total and downloaded >= next_report:
                pct = 100.0 * downloaded / total
                print(f"[download] {destination.name}: {downloaded / 2**20:.1f} MiB ({pct:.1f}%)")
                next_report += PROGRESS_EVERY_BYTES
    tmp_path.replace(destination)
    size_mib = destination.stat().st_size / 2**20
    print(f"[download] saved {destination} ({size_mib:.1f} MiB)")


def download_official_artifact(
    model_id: str,
    download_root: Path,
    repo_id: str = OFFICIAL_REPO_ID,
    revision: str = OFFICIAL_REVISION,
    force_download: bool = False,
) -> Path:
    artifact_dir = download_root / model_id
    for filename in ARTIFACT_FILENAMES:
        relative_path = f"{OFFICIAL_ARTIFACT_PREFIX}/{model_id}/{filename}"
        url = _hf_resolve_url(repo_id, revision, relative_path)
        _download_file(url, artifact_dir / filename, force_download)
    return artifact_dir


def _to_torch_key(jax_key: str) -> str:
    key = jax_key
    key = re.sub(r"encoder\.stages_(\d+)\.layers_(\d+)", r"encoder.stages.\1.\2", key)
    key = key.replace(".concat_norm_fn.", ".concat_norm.")
    key = key.replace(".kernel", ".weight")
    key = key.replace(".scale", ".weight")
    key = key.replace(".proj_conv.weight", ".skip.0.weight")
    key = key.replace(".proj_gn.weight", ".skip.1.weight")
    key = key.replace(".proj_gn.bias", ".skip.1.bias")
    return key


def _to_torch_tensor(value: np.ndarray, torch_key: str) -> "torch.Tensor":
    arr = np.asarray(value)
    if torch_key.endswith(".weight"):
        if arr.ndim == 4:
            # Flax conv: HWIO -> PyTorch conv: OIHW
            arr = np.transpose(arr, (3, 2, 0, 1))
        elif arr.ndim == 2:
            # Flax dense: IO -> PyTorch linear: OI
            arr = np.transpose(arr, (1, 0))
    arr = np.array(arr, dtype=np.float32, copy=True, order="C")
    return torch.from_numpy(arr)


def _build_torch_state(msgpack_path: Path) -> Dict[str, "torch.Tensor"]:
    restored = _msgpack_restore(msgpack_path)
    params = restored["params"] if isinstance(restored, dict) and "params" in restored else restored
    flat = _flatten_mapping(params)

    out: Dict[str, "torch.Tensor"] = {}
    for jax_key, value in flat.items():
        torch_key = _to_torch_key(jax_key)
        out[torch_key] = _to_torch_tensor(value, torch_key)
    return out


def _validate_state(
    converted: Dict[str, "torch.Tensor"],
    expected: Dict[str, "torch.Tensor"],
) -> Tuple[int, int, int]:
    expected_keys = set(expected.keys())
    converted_keys = set(converted.keys())
    missing = expected_keys - converted_keys
    unexpected = converted_keys - expected_keys

    mismatched = []
    for key in sorted(expected_keys & converted_keys):
        if tuple(converted[key].shape) != tuple(expected[key].shape):
            mismatched.append((key, tuple(converted[key].shape), tuple(expected[key].shape)))

    if missing or unexpected or mismatched:
        lines = ["State-dict mapping validation failed."]
        if missing:
            lines.append(f"Missing keys: {len(missing)} (e.g. {sorted(missing)[:5]})")
        if unexpected:
            lines.append(f"Unexpected keys: {len(unexpected)} (e.g. {sorted(unexpected)[:5]})")
        if mismatched:
            preview = ", ".join([f"{k}: got {a}, want {b}" for k, a, b in mismatched[:5]])
            lines.append(f"Shape mismatches: {len(mismatched)} ({preview})")
        raise RuntimeError("\n".join(lines))

    return len(missing), len(unexpected), len(mismatched)


def convert_artifact_dir(
    artifact_dir: Path,
    output_path: Optional[Path] = None,
    state_dict_only: bool = False,
) -> Path:
    _require_torch()
    _require_conversion_deps()

    artifact_dir = artifact_dir.resolve()
    metadata_path = artifact_dir / "metadata.json"
    msgpack_path = artifact_dir / "ema_params.msgpack"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    if not msgpack_path.exists():
        raise FileNotFoundError(f"ema_params.msgpack not found: {msgpack_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_config = dict(metadata.get("model_config", {}) or {})
    if not model_config:
        raise ValueError("metadata.json is missing model_config.")

    build_mae_from_config = _load_mae_builder()
    model = build_mae_from_config(model_config).cpu().eval()
    expected_state = model.state_dict()
    converted_state = _build_torch_state(msgpack_path)
    _validate_state(converted_state, expected_state)
    model.load_state_dict(converted_state, strict=True)

    if output_path is None:
        model_id = metadata.get("model_id", artifact_dir.name)
        resolved_output = REPO_ROOT / "weights" / "pt" / model_id / "ckpt_latest.pt"
    else:
        resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    if state_dict_only:
        payload: Any = converted_state
    else:
        step_loaded = int(metadata.get("source", {}).get("step_loaded", 0) or 0)
        payload = {
            "step": step_loaded,
            "model": converted_state,
            "ema": converted_state,
            "model_config": model_config,
            "hf_metadata": metadata,
            "source_repo": OFFICIAL_REPO_ID,
        }
    torch.save(payload, resolved_output)

    print(f"[OK] Converted checkpoint saved to: {resolved_output}")
    print(f"[OK] Number of tensors: {len(converted_state)}")
    print(f"[OK] Model config: {model_config}")
    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Drifting MAE JAX checkpoint to PyTorch")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Local directory containing metadata.json and ema_params.msgpack.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Official Hugging Face artifact id to download (default: mae_latent_640).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=OFFICIAL_REPO_ID,
        help="Hugging Face repo hosting the JAX artifact.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=OFFICIAL_REVISION,
        help="Hugging Face revision to download from.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=REPO_ROOT / "weights" / "hf" / "models" / "mae" / "jax",
        help="Where to cache downloaded JAX artifacts.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload metadata.json and ema_params.msgpack even if already cached.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the official JAX artifact and exit without converting it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .pt path (default: weights/pt/<model_id>/ckpt_latest.pt).",
    )
    parser.add_argument(
        "--state-dict-only",
        action="store_true",
        help="Save raw state_dict only instead of nested ckpt with model/ema.",
    )
    args = parser.parse_args()

    if args.artifact_dir is None:
        artifact_dir = download_official_artifact(
            model_id=args.model_id,
            download_root=args.download_root.resolve(),
            repo_id=args.repo_id,
            revision=args.revision,
            force_download=args.force_download,
        )
    else:
        artifact_dir = args.artifact_dir.resolve()

    if args.download_only:
        print(f"[OK] Downloaded artifact to: {artifact_dir}")
        return

    convert_artifact_dir(
        artifact_dir=artifact_dir,
        output_path=args.output,
        state_dict_only=args.state_dict_only,
    )


if __name__ == "__main__":
    main()
