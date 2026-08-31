#!/usr/bin/env python3
"""Run one matched ICLR B/4 stage-3/4 reverse/replay drift variant."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/gen/B4_rev-drift_mae256.yaml"),
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument(
        "--variant",
        choices=("rev", "replay", "replay_k16", "replay_k64", "replay_proxy"),
        required=True,
    )
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--mae-checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument(
        "--generator-remat",
        choices=("config", "on", "off"),
        default="config",
    )
    parser.add_argument(
        "--mae-remat",
        choices=("config", "on", "off"),
        default="config",
    )
    parser.add_argument(
        "--throughput-opt-level", type=int, choices=(0, 1, 2, 3), default=0
    )
    parser.add_argument("--replay-ratio", type=float, default=0.5)
    parser.add_argument("--profile-train-step", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not 0.0 < args.replay_ratio <= 1.0:
        parser.error("--replay-ratio must be in (0, 1]")
    if args.variant == "replay_proxy" and args.max_steps <= 0:
        parser.error("replay_proxy is benchmark-only and requires --max-steps")

    replay = args.variant != "rev"
    replay_bank_count = 64 if args.variant == "replay_k64" else 16
    replay_proxy = args.variant == "replay_proxy"
    cfg = load_yaml_config(args.config)
    cfg.update(
        {
            "name": Path(args.workdir).name,
            "project": "ICLR - B4 drift",
            "entity": "a01065522071-kaist-digital-humanities-and-social-science",
            "use_wandb": not args.disable_wandb,
            "cache_path": str(Path(args.cache_path).resolve()),
            "cache_format": "hf_parquet",
            "mae_checkpoint": str(Path(args.mae_checkpoint).resolve()),
            "eval_enabled": False,
            "eval_at_start": False,
            "seed": 43,
            "seed_host_rng": True,
            "batch_size": int(args.batch_size),
            "loader_batch_size": 128,
            "pos_per_sample": 128,
            "neg_per_sample": 32,
            "gen_per_label": 64,
            "total_generated_epochs": 40.0,
            "save_per_generated_epochs": 10.0,
            "keep_last": 20,
            "drift_matching": "rev-drift",
            "R_list": [0.2, 0.05, 0.02],
            "rev_drift_affinity_kernel": "exponential",
            "rev_drift_kernel_shape": 1.0,
            "rev_drift_kernel_adaptive_k_pos": 0,
            "rev_drift_kernel_adaptive_k_neg": 0,
            "rev_drift_force_multiplier": 1.0,
            "rev_drift_top_p": 1.0,
            "drift_top_k_pos": 0,
            "drift_top_k_neg": 0,
            "layer_temperature_profile": "uniform",
            "feature_loss_profile": "no_stage12_norm_x2",
            "feature_loss_group_normalize": True,
            "prune_skipped_feature_tensors": True,
            "historical_gen_replay": replay,
            "historical_gen_replay_ratio": (
                float(args.replay_ratio) if replay else 0.0
            ),
            "historical_gen_replay_count": 16,
            "historical_gen_replay_bank_count": replay_bank_count,
            "historical_gen_replay_start_generated_epochs": (
                0.0001 if replay_proxy else 10.0
            ),
            "historical_gen_replay_storage_dtype": "float16",
            "historical_gen_replay_source": (
                "fresh_current" if replay_proxy else "frozen_snapshot"
            ),
            "historical_gen_replay_ratio_start": None,
            "historical_gen_replay_ratio_ramp_start_step": 0,
            "historical_gen_replay_ratio_ramp_end_step": 0,
            "historical_gen_current_weight": None,
            "historical_gen_history_weight": None,
            "feature_adapter": False,
            "feature_gan": False,
            "throughput_opt_level": int(args.throughput_opt_level),
            "profile_train_step": bool(args.profile_train_step),
        }
    )
    if args.generator_remat != "config":
        generator_remat = args.generator_remat == "on"
        raw_cfg = cfg.get("_raw")
        if not isinstance(raw_cfg, dict) or not isinstance(
            raw_cfg.get("model"), dict
        ):
            parser.error("base config does not contain a model section")
        # Generator construction reads the nested raw YAML, not the flattened
        # dictionary, so the runtime override must be applied here.
        raw_cfg["model"]["use_remat"] = generator_remat
        cfg["use_remat"] = generator_remat
    if args.mae_remat != "config":
        cfg["mae_use_remat"] = args.mae_remat == "on"
    if args.profile_train_step:
        cfg["log_every_k"] = 1
    if args.max_steps > 0:
        cfg["train_max_step_exclusive"] = int(args.max_steps)

    for path_name, path_value in (
        ("cache", Path(cfg["cache_path"])),
        ("MAE checkpoint", Path(cfg["mae_checkpoint"])),
    ):
        if not path_value.exists():
            parser.error(f"{path_name} path does not exist: {path_value}")

    rank, world_size, device = setup_distributed()
    if is_main_process(rank):
        print(
            "[iclr-b4-stage34] "
            f"variant={args.variant} world_size={world_size} seed=43 "
            f"B={args.batch_size} P=128 N=32 G=64 "
            "profile=no_stage12_norm_x2 "
            f"generator_remat={cfg.get('use_remat')} "
            f"mae_remat={cfg.get('mae_use_remat')} "
            f"throughput_opt_level={args.throughput_opt_level} "
            f"replay={str(replay).lower()} "
            f"rho={args.replay_ratio if replay else 0.0:g} "
            f"H=16 bank_count={replay_bank_count} "
            f"snapshot_epoch={0.0001 if replay_proxy else 10:g} "
            "target_epochs=40",
            flush=True,
        )

    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)

    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
