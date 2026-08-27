#!/usr/bin/env python3
"""Train one Phase-1 real-supervised MAE feature-adapter ablation."""

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
    "frozen": {
        "feature_adapter": False,
        "feature_adapter_objective": "supcon",
        "feature_adapter_keys": ["layer3", "layer4"],
    },
    "s4_supcon": {
        "feature_adapter": True,
        "feature_adapter_objective": "supcon",
        "feature_adapter_keys": ["layer4"],
    },
    "s34_supcon": {
        "feature_adapter": True,
        "feature_adapter_objective": "supcon",
        "feature_adapter_keys": ["layer3", "layer4"],
    },
    "s34_supcon_ce": {
        "feature_adapter": True,
        "feature_adapter_objective": "supcon_ce",
        "feature_adapter_keys": ["layer3", "layer4"],
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
            "feature_adapter_bottleneck": 64,
            "feature_adapter_projection_dim": 128,
            "feature_adapter_dropout": 0.0,
            "feature_adapter_lr": 1.0e-4,
            "feature_adapter_weight_decay": 1.0e-4,
            "feature_adapter_update_freq": 1,
            "feature_adapter_samples_per_class": 8,
            "feature_adapter_temp": 0.1,
            "feature_adapter_reg_lambda": 0.01,
            "feature_adapter_loss_weight": 1.0,
            "feature_adapter_ce_weight": 0.1,
            "feature_adapter_max_grad_norm": 1.0,
            "feature_adapter_ema_decay": 0.999,
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
            "[feature-adapter-run] "
            f"variant={args.variant} enabled={cfg['feature_adapter']} "
            f"objective={cfg['feature_adapter_objective']} "
            f"keys={cfg['feature_adapter_keys']} seed={cfg['seed']}",
            flush=True,
        )
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
