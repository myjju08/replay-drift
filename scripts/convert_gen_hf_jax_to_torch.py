#!/usr/bin/env python3
"""Fetch and convert official Drifting generator JAX checkpoints to PyTorch.

Example:
  python scripts/convert_gen_hf_jax_to_torch.py \
    --model-id latent_B_sota \
    --output weights/pt/gen/latent_B_sota/ckpt_latest.pt
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

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.imagenet_generator import build_ditgen_from_config  # noqa: E402
from scripts.convert_mae_hf_jax_to_torch import _flatten_mapping, _msgpack_restore  # noqa: E402

OFFICIAL_REPO_ID = "Goodeat/drifting"
OFFICIAL_REVISION = "main"
OFFICIAL_ARTIFACT_PREFIX = "models/gen/jax"
DEFAULT_MODEL_ID = "latent_B_sota"
ARTIFACT_FILENAMES = ("metadata.json", "ema_params.msgpack")
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_EVERY_BYTES = 256 * 1024 * 1024


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
    request = urllib.request.Request(url, headers={"User-Agent": "dualdrift-gen-converter/1.0"})
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
    print(f"[download] saved {destination} ({destination.stat().st_size / 2**20:.1f} MiB)")


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
        _download_file(
            _hf_resolve_url(repo_id, revision, relative_path),
            artifact_dir / filename,
            force_download,
        )
    return artifact_dir


def _map_block_key(jax_key: str) -> Optional[str]:
    match = re.fullmatch(r"LightningDiT_0\.blocks_(\d+)\.(.+)", jax_key)
    if not match:
        return None
    idx, rest = match.groups()
    prefix = f"model.blocks.{idx}"

    direct = {
        "RMSNorm_0.weight": "norm1.weight",
        "RMSNorm_1.weight": "norm2.weight",
        "Attention_0.q_norm.weight": "attn.q_norm.weight",
        "Attention_0.k_norm.weight": "attn.k_norm.weight",
    }
    if rest in direct:
        return f"{prefix}.{direct[rest]}"

    linear_prefixes = (
        ("Attention_0.TorchLinear_0.Dense_0.", "attn.qkv."),
        ("Attention_0.TorchLinear_1.Dense_0.", "attn.proj."),
        ("SwiGLUFFN_0.TorchLinear_0.Dense_0.", "mlp.w1."),
        ("SwiGLUFFN_0.TorchLinear_1.Dense_0.", "mlp.w3."),
        ("SwiGLUFFN_0.TorchLinear_2.Dense_0.", "mlp.w2."),
        ("TorchLinear_0.Dense_0.", "adaLN_mod.1."),
    )
    for source, target in linear_prefixes:
        if rest.startswith(source):
            suffix = rest[len(source) :]
            return f"{prefix}.{target}{_param_suffix(suffix)}"
    return None


def _param_suffix(suffix: str) -> str:
    if suffix == "kernel":
        return "weight"
    if suffix == "bias":
        return "bias"
    raise KeyError(f"Unsupported parameter suffix: {suffix}")


def _to_torch_key(jax_key: str) -> str:
    top_level = {
        "Embed_0.embedding": "class_embed.weight",
        "RMSNorm_0.weight": "cfg_norm.weight",
        "LightningDiT_0.pos_embed": "model.pos_embed",
        "LightningDiT_0.cls_embed": "model.cls_embed",
        "LightningDiT_0.FinalLayer_0.RMSNorm_0.weight": "model.final_layer.norm.weight",
    }
    if jax_key in top_level:
        return top_level[jax_key]

    match = re.fullmatch(r"noise_embeds_(\d+)\.embedding", jax_key)
    if match:
        return f"noise_embeds.{match.group(1)}.weight"

    linear_prefixes = (
        ("TimestepEmbedder_0.TorchLinear_0.Dense_0.", "cfg_embedder.mlp.0."),
        ("TimestepEmbedder_0.TorchLinear_1.Dense_0.", "cfg_embedder.mlp.2."),
        ("LightningDiT_0.TorchLinear_0.Dense_0.", "model.patch_embed."),
        ("LightningDiT_0.TorchLinear_1.Dense_0.", "model.cls_proj."),
        ("LightningDiT_0.FinalLayer_0.TorchLinear_0.Dense_0.", "model.final_layer.adaLN_mod.1."),
        ("LightningDiT_0.FinalLayer_0.TorchLinear_1.Dense_0.", "model.final_layer.linear."),
    )
    for source, target in linear_prefixes:
        if jax_key.startswith(source):
            suffix = jax_key[len(source) :]
            return f"{target}{_param_suffix(suffix)}"

    block_key = _map_block_key(jax_key)
    if block_key is not None:
        return block_key

    raise KeyError(f"Unsupported JAX generator key: {jax_key}")


def _to_torch_tensor(value: np.ndarray, jax_key: str, torch_key: str) -> torch.Tensor:
    arr = np.asarray(value)
    if jax_key.endswith(".kernel"):
        if arr.ndim != 2 or not torch_key.endswith(".weight"):
            raise ValueError(f"Unexpected dense kernel shape/key: {jax_key} -> {torch_key}, {arr.shape}")
        arr = arr.T
    arr = np.array(arr, dtype=np.float32, copy=True, order="C")
    return torch.from_numpy(arr)


def _build_torch_state(msgpack_path: Path) -> Dict[str, torch.Tensor]:
    restored = _msgpack_restore(msgpack_path)
    params = restored["params"] if isinstance(restored, Mapping) and "params" in restored else restored
    flat = _flatten_mapping(params)

    out: Dict[str, torch.Tensor] = {}
    for jax_key, value in flat.items():
        torch_key = _to_torch_key(jax_key)
        if torch_key in out:
            raise KeyError(f"Duplicate mapped key: {jax_key} -> {torch_key}")
        out[torch_key] = _to_torch_tensor(value, jax_key, torch_key)
    return out


def _validate_state(
    converted: Dict[str, torch.Tensor],
    expected: Dict[str, torch.Tensor],
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
        lines = ["Generator state-dict mapping validation failed."]
        if missing:
            lines.append(f"Missing keys: {len(missing)} (e.g. {sorted(missing)[:8]})")
        if unexpected:
            lines.append(f"Unexpected keys: {len(unexpected)} (e.g. {sorted(unexpected)[:8]})")
        if mismatched:
            preview = ", ".join([f"{k}: got {a}, want {b}" for k, a, b in mismatched[:8]])
            lines.append(f"Shape mismatches: {len(mismatched)} ({preview})")
        raise RuntimeError("\n".join(lines))

    return len(missing), len(unexpected), len(mismatched)


def _build_expected_model(model_config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = dict(model_config)
    num_classes = int(model_cfg.pop("num_classes", 1000))
    return build_ditgen_from_config(model_cfg, {"num_classes": num_classes}).cpu().eval()


def convert_artifact_dir(
    artifact_dir: Path,
    output_path: Optional[Path] = None,
    state_dict_only: bool = False,
) -> Path:
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

    model = _build_expected_model(model_config)
    expected_state = model.state_dict()
    converted_state = _build_torch_state(msgpack_path)
    _validate_state(converted_state, expected_state)
    model.load_state_dict(converted_state, strict=True)

    if output_path is None:
        model_id = str(metadata.get("model_id", artifact_dir.name))
        resolved_output = REPO_ROOT / "weights" / "pt" / "gen" / model_id / "ckpt_latest.pt"
    else:
        resolved_output = output_path.resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    step_loaded = int(metadata.get("source", {}).get("step_loaded", 0) or 0)
    if state_dict_only:
        payload: Any = converted_state
    else:
        payload = {
            "step": step_loaded,
            "model": converted_state,
            "ema": converted_state,
            "model_config": model_config,
            "hf_metadata": metadata,
            "source_repo": OFFICIAL_REPO_ID,
        }
    torch.save(payload, resolved_output)

    print(f"[OK] Converted generator checkpoint saved to: {resolved_output}")
    print(f"[OK] Number of tensors: {len(converted_state)}")
    print(f"[OK] Model config: {model_config}")
    return resolved_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Drifting generator JAX checkpoint to PyTorch")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--repo-id", type=str, default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision", type=str, default=OFFICIAL_REVISION)
    parser.add_argument(
        "--download-root",
        type=Path,
        default=REPO_ROOT / "weights" / "hf" / "models" / "gen" / "jax",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--state-dict-only", action="store_true")
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
