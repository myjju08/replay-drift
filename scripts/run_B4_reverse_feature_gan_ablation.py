#!/usr/bin/env python3
"""Train one Phase-2 frozen-MAE feature-GAN ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_imagenet_gen import (  # noqa: E402
    is_main_process,
    load_yaml_config,
    setup_distributed,
    train_gen,
)


VARIANTS = {
    "s4_gr005": {
        "feature_gan_keys": ["layer4"],
        "feature_gan_gradient_ratio": 0.05,
    },
    "s4_gr010": {
        "feature_gan_keys": ["layer4"],
        "feature_gan_gradient_ratio": 0.10,
    },
    "s4_gr020": {
        "feature_gan_keys": ["layer4"],
        "feature_gan_gradient_ratio": 0.20,
    },
    "s34_gr010": {
        "feature_gan_keys": ["layer3", "layer4"],
        "feature_gan_gradient_ratio": 0.10,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/gen/B4_rev-drift_mae256.yaml"),
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--generated-epochs", type=float, default=40.0)
    parser.add_argument("--save-every-epochs", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.update(
        {
            "name": Path(args.workdir).name,
            "seed": int(args.seed),
            "batch_size": 10,
            "gen_per_label": 64,
            "pos_per_sample": 128,
            "neg_per_sample": 32,
            "total_generated_epochs": float(args.generated_epochs),
            "save_per_generated_epochs": float(args.save_every_epochs),
            "eval_at_start": False,
            "eval_per_step": 1_000_000_000,
            "keep_last": 20,
            "drift_matching": "rev-drift",
            "mix_alpha_adaptive": False,
            "stochastic_feature_stage_loss": False,
            "prune_skipped_feature_tensors": False,
            "feature_loss_group_weights": {"default": 1.0},
            "feature_loss_group_normalize": False,
            "layer_temperature_multipliers": {"default": 1.0},
            "rev_drift_top_p": 1.0,
            "drift_top_k_pos": 0,
            "drift_top_k_neg": 0,
            "topk_diagnostic_steps": 0,
            "R_list": [0.75],
            "rev_drift_affinity_kernel": "generalized_exponential",
            "rev_drift_kernel_shape": 2.0,
            "rev_drift_kernel_adaptive_k_pos": 32,
            "rev_drift_kernel_adaptive_k_neg": 24,
            "rev_drift_kernel_adaptive_margin": 1.05,
            "rev_drift_force_multiplier": 3.0,
            "rev_drift_kernel_temperature_mix": [],
            "rev_drift_kernel_temperature_mix_weights": [],
            # Keep the MAE metric fixed; only the small discriminator heads learn.
            "feature_adapter": False,
            "feature_gan": True,
            "feature_gan_hidden_channels": 128,
            "feature_gan_real_samples_per_class": 64,
            "feature_gan_lr": 2.0e-4,
            "feature_gan_weight_decay": 1.0e-4,
            "feature_gan_adam_b1": 0.0,
            "feature_gan_adam_b2": 0.9,
            "feature_gan_discriminator_max_grad_norm": 5.0,
            "feature_gan_warmup_steps": 1000,
            "feature_gan_gradient_calibration_freq": 10,
            "feature_gan_gradient_ratio_ema": 0.0,
            "feature_gan_scale_min": 0.0,
            "feature_gan_scale_max": 1000.0,
            "feature_gan_use_bf16": True,
        }
    )
    cfg.update(VARIANTS[args.variant])
    if args.max_steps > 0:
        cfg["train_max_step_exclusive"] = int(args.max_steps)
    if args.disable_wandb:
        cfg["use_wandb"] = False

    rank, world_size, device = setup_distributed()
    if is_main_process(rank):
        print(
            "[feature-gan-run] "
            f"variant={args.variant} keys={cfg['feature_gan_keys']} "
            f"target_gradient_ratio={cfg['feature_gan_gradient_ratio']:g} "
            f"seed={cfg['seed']}",
            flush=True,
        )
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
