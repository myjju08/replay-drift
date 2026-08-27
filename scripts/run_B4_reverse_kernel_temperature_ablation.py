#!/usr/bin/env python3
"""Train one full-feature B/4 reverse-kernel bandwidth ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_imagenet_gen import load_yaml_config, setup_distributed, train_gen  # noqa: E402


COMMON_KERNEL = {
    "rev_drift_kernel_shape": 1.0,
    "rev_drift_kernel_adaptive_k_pos": 32,
    "rev_drift_kernel_adaptive_k_neg": 24,
    "rev_drift_kernel_adaptive_margin": 1.05,
    # One unit-RMS field, scaled to match the nominal magnitude of the
    # original sum of three independently normalized temperature fields.
    "rev_drift_force_multiplier": 3.0,
}

VARIANTS = {
    "cauchy_r05": {
        "R_list": [0.5],
        "rev_drift_affinity_kernel": "student_t",
    },
    "cauchy_r20": {
        "R_list": [2.0],
        "rev_drift_affinity_kernel": "student_t",
    },
    "wendland_r15": {
        "R_list": [1.5],
        "rev_drift_affinity_kernel": "wendland",
    },
    "wendland_r20": {
        "R_list": [2.0],
        "rev_drift_affinity_kernel": "wendland",
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated-epochs", type=float, default=40.0)
    parser.add_argument("--save-every-epochs", type=float, default=10.0)
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
        }
    )
    cfg.update(COMMON_KERNEL)
    cfg.update(VARIANTS[args.variant])

    rank, world_size, device = setup_distributed()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
