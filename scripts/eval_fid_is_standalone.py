"""Standalone FID/IS eval for a single checkpoint at one CFG scale.

Reuses eval_fid_is from train_imagenet_gen.py. Single-GPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train_imagenet_gen import (  # noqa: E402
    eval_fid_is,
    load_yaml_config,
)
from models.imagenet_generator import build_ditgen_from_config  # noqa: E402
from train.train_data import create_imagenet_split  # noqa: E402
from utils import EMA  # noqa: E402


def _load_ema_state_dict_compat(ema: EMA, state_dict: dict) -> None:
    """Load EMA weights, trimming the old extra null-class row when present."""
    target_state = ema.shadow.state_dict()
    key = "class_embed.weight"
    if key in state_dict and key in target_state:
        src = state_dict[key]
        dst = target_state[key]
        if (
            torch.is_tensor(src)
            and torch.is_tensor(dst)
            and src.ndim == 2
            and dst.ndim == 2
            and src.shape != dst.shape
            and src.shape[0] == dst.shape[0] + 1
            and src.shape[1:] == dst.shape[1:]
        ):
            state_dict = dict(state_dict)
            state_dict[key] = src[: dst.shape[0]]
            print(
                "[eval] trimmed legacy null class embedding "
                f"{tuple(src.shape)} -> {tuple(state_dict[key].shape)}"
            )
    ema.load_state_dict(state_dict)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg_scale", type=float, required=True)
    p.add_argument("--n_samples", type=int, default=1024)
    p.add_argument("--out", required=True, help="JSON output path")
    p.add_argument(
        "--eval_ref_npz",
        default=os.environ.get("IMAGENET256_REF_BATCH", ""),
        help="Optional official ImageNet-256 reference batch (.npz). If set, labels and real images come from this file.",
    )
    p.add_argument(
        "--save_sample_npz",
        action="store_true",
        help="Also save a generated sample batch in ADM/OpenAI .npz format next to the JSON output.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml_config(args.config)
    raw_cfg = cfg.get("_raw", {})

    print(f"[eval] device={device}  cfg_scale={args.cfg_scale}  n_samples={args.n_samples}")
    print(f"[eval] ckpt={args.ckpt}")

    # Build generator + EMA shell, then load checkpoint into EMA shadow.
    gen_raw = build_ditgen_from_config(raw_cfg.get("model", cfg), raw_cfg.get("dataset", cfg)).to(device)
    ema = EMA(gen_raw, decay=float(cfg.get("ema_decay", 0.999)))

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    _load_ema_state_dict_compat(ema, state["ema"])
    step_loaded = int(state.get("step", -1))
    print(f"[eval] loaded EMA from step {step_loaded}")

    ema.shadow.to(device).eval()

    # Free the non-EMA generator since we only eval EMA.
    del gen_raw, state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_loader = None
    eval_postprocess_fn = None
    if args.eval_ref_npz:
        # We still need the same decode/postprocess function used in training, but
        # not the validation loader itself when evaluating against a reference .npz.
        imagenet_path = cfg.get("imagenet_path") or os.environ.get("IMAGENET_PATH", "")
        cache_path = cfg.get("cache_path") or os.environ.get("IMAGENET_CACHE_PATH", "")
        _, _, eval_postprocess_fn = create_imagenet_split(
            imagenet_path=imagenet_path,
            resolution=int(cfg.get("resolution", 256)),
            batch_size=int(cfg.get("eval_batch_size", 16)),
            split="train",
            use_aug=False,
            use_latent=bool(cfg.get("use_latent", True)),
            use_cache=bool(cfg.get("use_cache", True)),
            cache_path=cache_path,
            num_workers=0,
            pin_memory=bool(cfg.get("pin_memory", True)),
            distributed=False,
            rank=0,
            world_size=1,
            latent_device=device,
        )
    else:
        # Build val loader (rank-0-style, no DDP).
        imagenet_path = cfg.get("imagenet_path") or os.environ.get("IMAGENET_PATH", "")
        cache_path = cfg.get("cache_path") or os.environ.get("IMAGENET_CACHE_PATH", "")
        eval_loader, _, eval_postprocess_fn = create_imagenet_split(
            imagenet_path=imagenet_path,
            resolution=int(cfg.get("resolution", 256)),
            batch_size=int(cfg.get("eval_batch_size", 16)),
            split="val",
            use_aug=False,
            use_latent=bool(cfg.get("use_latent", True)),
            use_cache=bool(cfg.get("use_cache", True)),
            cache_path=cache_path,
            num_workers=int(cfg.get("num_workers", 4)),
            pin_memory=bool(cfg.get("pin_memory", True)),
            distributed=False,
            rank=0,
            world_size=1,
            latent_device=device,
        )

    t0 = time.time()
    stats = eval_fid_is(
        ema.shadow,
        eval_postprocess_fn,
        eval_loader,
        device,
        cfg_scale=float(args.cfg_scale),
        n_samples=int(args.n_samples),
        workdir=str(Path(args.out).parent),
        step=step_loaded,
        label=f"CFG{args.cfg_scale}",
        eval_ref_npz=args.eval_ref_npz,
        save_sample_npz=bool(args.save_sample_npz),
    )
    elapsed = time.time() - t0

    out = {
        "ckpt": args.ckpt,
        "step": step_loaded,
        "cfg_scale": args.cfg_scale,
        "n_samples": args.n_samples,
        "elapsed_sec": elapsed,
        **stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] DONE cfg={args.cfg_scale}  fid={stats.get('fid')}  is_mean={stats.get('is_mean')}  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
