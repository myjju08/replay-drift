#!/usr/bin/env python3
"""Train one full-feature B/4 Gaussian temperature/mixing ablation."""

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
    "gaussian_r045": {
        "R_list": [0.45],
        "rev_drift_kernel_temperature_mix": [],
        "rev_drift_kernel_temperature_mix_weights": [],
        "rev_drift_force_multiplier": 3.0,
    },
    "gaussian_r075": {
        "R_list": [0.75],
        "rev_drift_kernel_temperature_mix": [],
        "rev_drift_kernel_temperature_mix_weights": [],
        "rev_drift_force_multiplier": 3.0,
    },
    # Evaluate both Gaussian scales, mix their raw affinities, then perform
    # mutual normalization and force construction only once.
    "gaussian_kernelmix_r045_r075": {
        "R_list": [1.0],
        "rev_drift_kernel_temperature_mix": [0.45, 0.75],
        "rev_drift_kernel_temperature_mix_weights": [0.5, 0.5],
        "rev_drift_force_multiplier": 3.0,
    },
    # Conventional multitemperature control: independently normalize and
    # construct two unit-RMS fields. Multiplying their sum by 1.5 matches the
    # other variants' nominal total force magnitude of three.
    "gaussian_fieldmix_r045_r075": {
        "R_list": [0.45, 0.75],
        "rev_drift_kernel_temperature_mix": [],
        "rev_drift_kernel_temperature_mix_weights": [],
        "rev_drift_force_multiplier": 1.5,
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
            "rev_drift_affinity_kernel": "generalized_exponential",
            "rev_drift_kernel_shape": 2.0,
            "rev_drift_kernel_adaptive_k_pos": 32,
            "rev_drift_kernel_adaptive_k_neg": 24,
            "rev_drift_kernel_adaptive_margin": 1.05,
        }
    )
    cfg.update(VARIANTS[args.variant])

    rank, world_size, device = setup_distributed()
    if is_main_process(rank):
        print(
            "[gaussian-temperature-run] "
            f"variant={args.variant} R_list={cfg['R_list']} "
            f"kernel_mix={cfg['rev_drift_kernel_temperature_mix']} "
            f"kernel_mix_weights={cfg['rev_drift_kernel_temperature_mix_weights']} "
            f"force_multiplier={cfg['rev_drift_force_multiplier']}",
            flush=True,
        )
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
