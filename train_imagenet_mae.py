"""MAE pretraining for ImageNet — PyTorch port of the official JAX train_mae.py.

Usage (single GPU):
    python train_imagenet_mae.py --config configs/mae/latent_640.yaml --workdir runs/mae_latent_640

Usage (multi-GPU, torchrun):
    torchrun --nproc_per_node=8 train_imagenet_mae.py \
        --config configs/mae/latent_640.yaml --workdir runs/mae_latent_640
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# Local imports
from models.mae_resnet import MAEResNet, build_mae_from_config
from train.train_data import create_imagenet_split, infinite_sampler
from utils import EMA


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

def load_yaml_config(path: str) -> dict:
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    # Flatten nested dicts: merge logging/env/dataset/model/optimizer/train/feature
    cfg: dict = {}
    for section in ("logging", "env", "dataset", "model", "optimizer", "train", "feature"):
        cfg.update(raw.get(section, {}))
    cfg["_raw"] = raw
    return cfg


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, torch.device]:
    """Initialize DDP if available; fall back to single-process."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def reduce_dict(metrics: Dict[str, torch.Tensor], world_size: int) -> Dict[str, float]:
    """Average scalar metrics across all DDP ranks."""
    if world_size == 1:
        return {k: v.mean().item() for k, v in metrics.items()}
    tensors = torch.stack([v.mean() for v in metrics.values()])
    dist.all_reduce(tensors, op=dist.ReduceOp.AVG)
    return {k: tensors[i].item() for i, k in enumerate(metrics.keys())}


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    lr = float(cfg.get("lr", 4e-3))
    wd = float(cfg.get("weight_decay", 0.01))
    b1 = float(cfg.get("adam_b1", 0.9))
    b2 = float(cfg.get("adam_b2", 0.95))
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(b1, b2))


def get_lr(
    step: int,
    warmup_steps: int,
    base_lr: float,
    schedule: str = "const",
    total_steps: Optional[int] = None,
) -> float:
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    if schedule == "const":
        return base_lr
    if schedule == "cosine":
        # Cosine decay from warmup end to `total_steps`.
        if total_steps is None or total_steps <= warmup_steps:
            return base_lr
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = max(0.0, min(1.0, float(progress)))
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    workdir: str,
    step: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    keep_last: int = 2,
    keep_every: int = 50000,
) -> None:
    ckpt_dir = Path(workdir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_step_{step:07d}.pt"
    torch.save(
        {
            "step": step,
            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        },
        path,
    )
    # Always save a "latest" symlink-style copy
    latest = ckpt_dir / "ckpt_latest.pt"
    torch.save(torch.load(path, map_location="cpu", weights_only=False), latest)

    # Rotation: keep_last latest + every keep_every
    checkpoints = sorted(ckpt_dir.glob("ckpt_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for old in checkpoints[:-keep_last]:
        old_step = int(old.stem.split("_")[-1])
        if old_step % keep_every != 0:
            old.unlink(missing_ok=True)


def load_checkpoint(
    workdir: str,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load latest checkpoint; returns the step we resume from (0 if fresh)."""
    ckpt_dir = Path(workdir) / "checkpoints"
    latest = ckpt_dir / "ckpt_latest.pt"
    if not latest.exists():
        return 0
    state = torch.load(latest, map_location=device, weights_only=False)
    raw = model.module if hasattr(model, "module") else model
    raw.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    step = int(state.get("step", 0))
    return step


# ---------------------------------------------------------------------------
# W&B / file logger
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, workdir: str, cfg: dict, rank: int):
        self.rank = rank
        self.log_file = Path(workdir) / "train_log.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.use_wandb = bool(cfg.get("use_wandb", False)) and rank == 0
        self._step = 0
        if self.use_wandb and rank == 0:
            try:
                import wandb
                wandb.init(
                    project=cfg.get("project", "ReplayDrift"),
                    entity=cfg.get("entity") or None,
                    name=cfg.get("name", Path(workdir).name),
                    config=cfg,
                )
                self._wandb = wandb
            except Exception as e:
                print(f"[W&B] init failed: {e}. Falling back to file logging.")
                self.use_wandb = False

    def log(self, metrics: dict, step: Optional[int] = None) -> None:
        if self.rank != 0:
            return
        s = step if step is not None else self._step
        record = {"step": s, **metrics}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.use_wandb:
            self._wandb.log(metrics, step=s)

    def set_step(self, step: int) -> None:
        self._step = step

    def finish(self) -> None:
        if self.use_wandb and self.rank == 0:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loop(
    model: nn.Module,
    eval_loader,
    preprocess_fn,
    device: torch.device,
    world_size: int,
    cfg: dict,
    eval_samples: int = 5000,
    mask_ratio: float = 0.5,
    lambda_cls: float = 0.0,
    label: str = "eval",
) -> Dict[str, float]:
    model.eval()
    total_metrics: Dict[str, list] = {}
    n = 0
    for batch in eval_loader:
        processed = preprocess_fn(batch)
        images = processed["images"].to(device)
        labels = processed["labels"].to(device)

        raw_model = model.module if hasattr(model, "module") else model
        _, metrics = raw_model(
            images,
            labels,
            lambda_cls=lambda_cls,
            mask_ratio_min=mask_ratio,
            mask_ratio_max=mask_ratio,
            train=False,
        )
        bs = images.shape[0]
        for k, v in metrics.items():
            total_metrics.setdefault(k, []).append(v.mean().item() * bs)
        n += bs
        if n >= eval_samples:
            break

    result = {}
    for k, vals in total_metrics.items():
        result[f"{label}/{k}"] = sum(vals) / max(n, 1)
    return result


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_mae(cfg: dict, workdir: str, rank: int, world_size: int, device: torch.device) -> None:
    torch.manual_seed(cfg.get("seed", 42) + rank)
    log_every = cfg.get("log_every_k", 40)

    # --- Build model ---
    model_raw = build_mae_from_config(cfg)
    model_raw = model_raw.to(device)
    if world_size > 1:
        model = DDP(model_raw, device_ids=[device.index], find_unused_parameters=False)
    else:
        model = model_raw

    ema = EMA(model_raw, decay=float(cfg.get("ema_decay", 0.9995)))

    # --- Optimizer ---
    optimizer = build_optimizer(model_raw, cfg)
    base_lr    = float(cfg.get("lr", 4e-3))
    warmup     = int(cfg.get("warmup_steps", 4000))
    lr_sched   = cfg.get("lr_schedule", "const")

    # --- Data ---
    imagenet_path = cfg.get("imagenet_path") or os.environ.get("IMAGENET_PATH", "")
    cache_path    = cfg.get("cache_path") or os.environ.get("IMAGENET_CACHE_PATH", "")
    use_latent    = bool(cfg.get("use_latent", False))
    use_cache     = bool(cfg.get("use_cache", False))
    use_aug       = bool(cfg.get("use_aug", True))
    resolution    = int(cfg.get("resolution", 256))
    batch_size    = int(cfg.get("batch_size", 512))
    eval_bsz      = int(cfg.get("eval_batch_size", 512))

    train_loader, preprocess_fn, _ = create_imagenet_split(
        imagenet_path=imagenet_path,
        resolution=resolution,
        batch_size=batch_size,
        split="train",
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        cache_path=cache_path,
        num_workers=int(cfg.get("num_workers", 8)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
        latent_device=device,
    )
    eval_loader, _, _ = create_imagenet_split(
        imagenet_path=imagenet_path,
        resolution=resolution,
        batch_size=eval_bsz,
        split="val",
        use_aug=False,
        use_latent=use_latent,
        use_cache=use_cache,
        cache_path=cache_path,
        num_workers=int(cfg.get("num_workers", 8)),
        pin_memory=bool(cfg.get("pin_memory", True)),
        distributed=(world_size > 1),
        rank=rank,
        world_size=world_size,
        latent_device=device,
    )

    # --- Checkpoint resume ---
    logger = Logger(workdir, cfg, rank)
    start_step = load_checkpoint(workdir, model, ema, optimizer, device)
    if is_main_process(rank):
        print(f"Resuming from step {start_step}")

    total_steps        = int(cfg.get("total_steps", 200000))
    save_per_step      = int(cfg.get("save_per_step", 5000))
    eval_per_step      = int(cfg.get("eval_per_step", 2000))
    eval_samples       = int(cfg.get("eval_samples", 5000))
    max_grad_norm      = float(cfg.get("max_grad_norm", 2.0))
    keep_every         = int(cfg.get("keep_every", 50000))
    keep_last          = int(cfg.get("keep_last", 2))
    mask_ratio_min     = float(cfg.get("mask_ratio_min", 0.5))
    mask_ratio_max     = float(cfg.get("mask_ratio_max", 0.5))
    finetune_last      = int(cfg.get("finetune_last_steps", 3000))
    warmup_finetune    = int(cfg.get("warmup_finetune", 1000))
    finetune_cls       = float(cfg.get("finetune_cls", 0.1))
    lambda_cls_base    = float(cfg.get("lambda_cls", 0.0))
    start_finetune     = total_steps - finetune_last

    train_iter = infinite_sampler(train_loader, start_step)
    pbar = (
        tqdm(range(start_step, total_steps), initial=start_step, total=total_steps)
        if is_main_process(rank)
        else range(start_step, total_steps)
    )

    start_time_all = time.time()
    for step in pbar:
        logger.set_step(step)
        model.train()

        # Compute lambda_cls with finetune ramp
        if step >= start_finetune and finetune_last > 0:
            frac = min(1.0, (step - start_finetune) / max(1, warmup_finetune))
            lambda_cls = finetune_cls * frac
        else:
            lambda_cls = lambda_cls_base

        # LR update
        lr = get_lr(step, warmup, base_lr, lr_sched, total_steps=total_steps)
        set_lr(optimizer, lr)

        # Fetch and preprocess batch
        t0 = time.time()
        batch = next(train_iter)
        processed = preprocess_fn(batch)
        images = processed["images"].to(device)
        labels = processed["labels"].to(device)
        t_data = time.time() - t0

        # Forward + loss
        t1 = time.time()
        raw_model = model.module if hasattr(model, "module") else model
        loss, metrics = raw_model(
            images,
            labels,
            lambda_cls=lambda_cls,
            mask_ratio_min=mask_ratio_min,
            mask_ratio_max=mask_ratio_max,
            train=True,
        )
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        g_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        ema.update(model_raw)
        t_train = time.time() - t1

        # Logging
        if step % log_every == 0 and is_main_process(rank):
            log_metrics = {k: v.mean().item() for k, v in metrics.items()}
            log_metrics["g_norm"]      = g_norm.item() if isinstance(g_norm, torch.Tensor) else float(g_norm)
            log_metrics["lr"]          = lr
            log_metrics["lambda_cls"]  = lambda_cls
            log_metrics["time/data"]   = t_data
            log_metrics["time/train"]  = t_train
            log_metrics["time/per_step"] = (time.time() - start_time_all) / (step - start_step + 1)
            log_metrics["kimg"]        = (step - start_step + 1) * images.shape[0] * world_size / 1000.0
            logger.log(log_metrics)

        # Eval
        if (step + 1) % eval_per_step == 0 and is_main_process(rank):
            eval_m = eval_loop(
                model, eval_loader, preprocess_fn, device, world_size, cfg,
                eval_samples=eval_samples, mask_ratio=mask_ratio_min,
                lambda_cls=lambda_cls_base, label="eval",
            )
            eval_nomask = eval_loop(
                model, eval_loader, preprocess_fn, device, world_size, cfg,
                eval_samples=eval_samples, mask_ratio=0.0,
                lambda_cls=lambda_cls_base, label="eval_nomask",
            )
            logger.log({**eval_m, **eval_nomask}, step=step + 1)
            # EMA eval
            ema_m = eval_loop(
                ema.shadow, eval_loader, preprocess_fn, device, world_size, cfg,
                eval_samples=eval_samples, mask_ratio=mask_ratio_min,
                lambda_cls=lambda_cls_base, label="eval_ema",
            )
            logger.log(ema_m, step=step + 1)
            model.train()

        # Checkpoint
        should_save = (
            (step + 1) % save_per_step == 0
            or (step + 1) == total_steps
            or (step + 1) == start_finetune
        )
        if should_save and is_main_process(rank):
            save_checkpoint(workdir, step + 1, model, ema, optimizer, cfg, keep_last, keep_every)

    if is_main_process(rank):
        logger.finish()
        print("MAE training complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet MAE pretraining (PyTorch)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--workdir", type=str, default="runs/mae", help="Working directory for checkpoints and logs.")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)

    # Override from env vars if not set in YAML
    if not cfg.get("imagenet_path"):
        cfg["imagenet_path"] = os.environ.get("IMAGENET_PATH", "")
    if not cfg.get("cache_path"):
        cfg["cache_path"] = os.environ.get("IMAGENET_CACHE_PATH", "")

    rank, world_size, device = setup_distributed()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    train_mae(cfg, args.workdir, rank, world_size, device)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
