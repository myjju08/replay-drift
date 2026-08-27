#!/usr/bin/env python3
"""Train the full-feature B/4 DualDrift/reverse-6 matched comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_imagenet_gen import load_yaml_config, setup_distributed, train_gen  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/gen/B4_dual-drift_mae256.yaml"),
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--variant", choices=("dual3x3", "reverse6"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated-epochs", type=float, default=40.0)
    parser.add_argument("--save-every-epochs", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    common = {
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
        "stochastic_feature_stage_loss": False,
        "prune_skipped_feature_tensors": False,
        "feature_loss_group_weights": {"default": 1.0},
        "feature_loss_group_normalize": False,
        "layer_temperature_multipliers": {"default": 1.0},
        "rev_drift_top_p": 1.0,
        "fwd_drift_top_p": 1.0,
        "drift_top_k_pos": 0,
        "drift_top_k_neg": 0,
        "topk_diagnostic_steps": 0,
    }
    cfg.update(common)

    if args.variant == "dual3x3":
        cfg.update(
            {
                "drift_matching": "dual-drift",
                "dual_drift_share_distances": True,
                "R_list": [0.4, 0.10, 0.04],
                "mix_baseline_R_list": [0.2, 0.05, 0.02],
                # Preserve the current forward-only adaptive initialization.
                "mix_alpha_adaptive": True,
                "mix_alpha_adaptive_mode": "hedge_no_time",
                "mix_alpha_adaptive_initial": 0.0,
                "mix_alpha_horizon_steps": 100_000,
                "mix_alpha_adaptive_gamma": 8.0,
                "mix_alpha_adaptive_eta": 1.0e-4,
                "mix_alpha_adaptive_decay": 1.0e-4,
                "mix_alpha_adaptive_gamma_warmup_steps": 0,
                "rev_drift_force_multiplier": 1.0,
            }
        )
    else:
        cfg.update(
            {
                "drift_matching": "rev-drift",
                "R_list": [0.4, 0.2, 0.10, 0.05, 0.04, 0.02],
                "mix_alpha_adaptive": False,
                # Six unit-normalized forces would have twice the nominal
                # amplitude of either three-temperature DualDrift expert.
                "rev_drift_force_multiplier": 0.5,
            }
        )

    rank, world_size, device = setup_distributed()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
