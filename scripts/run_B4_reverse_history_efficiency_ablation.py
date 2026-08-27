#!/usr/bin/env python3
"""Train one historical-replay support/mass efficiency ablation."""

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
    "h04_rho050": {"count": 4, "ratio": 0.50, "ramp": False},
    "h08_rho050": {"count": 8, "ratio": 0.50, "ramp": False},
    "h16_rho035": {"count": 16, "ratio": 0.35, "ramp": False},
    "h16_ramp050": {"count": 16, "ratio": 0.50, "ramp": True},
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
    parser.add_argument("--bank-count", type=int, default=16)
    parser.add_argument("--history-start-epoch", type=float, default=10.0)
    parser.add_argument("--ramp-start-step", type=int, default=10010)
    parser.add_argument("--ramp-end-step", type=int, default=20019)
    parser.add_argument("--generated-epochs", type=float, default=40.0)
    parser.add_argument("--save-every-epochs", type=float, default=10.0)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    count = int(variant["count"])
    ratio = float(variant["ratio"])
    ramp = bool(variant["ramp"])
    cfg = load_yaml_config(args.config)
    cfg.update(
        {
            "name": Path(args.workdir).name,
            "seed": int(args.seed),
            "seed_host_rng": True,
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
            "feature_adapter": False,
            "feature_gan": False,
            "historical_gen_replay": True,
            "historical_gen_replay_ratio": ratio,
            "historical_gen_replay_count": count,
            "historical_gen_replay_bank_count": int(args.bank_count),
            "historical_gen_replay_start_generated_epochs": float(
                args.history_start_epoch
            ),
            "historical_gen_replay_storage_dtype": "float16",
            "historical_gen_replay_source": "frozen_snapshot",
            "historical_gen_replay_ratio_start": 0.0 if ramp else None,
            "historical_gen_replay_ratio_ramp_start_step": (
                int(args.ramp_start_step) if ramp else 0
            ),
            "historical_gen_replay_ratio_ramp_end_step": (
                int(args.ramp_end_step) if ramp else 0
            ),
            "historical_gen_current_weight": None,
            "historical_gen_history_weight": None,
        }
    )
    if args.disable_wandb:
        cfg["use_wandb"] = False

    rank, world_size, device = setup_distributed()
    if is_main_process(rank):
        history_weight = ratio * int(cfg["gen_per_label"]) / count
        print(
            "[history-efficiency-run] "
            f"variant={args.variant} count={count} bank_count={args.bank_count} "
            f"ratio={ratio:g} static_history_weight={history_weight:g} "
            f"ramp={str(ramp).lower()} "
            f"ramp_steps={args.ramp_start_step}:{args.ramp_end_step} "
            f"shared_start_epoch=10 seed={cfg['seed']}",
            flush=True,
        )
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
