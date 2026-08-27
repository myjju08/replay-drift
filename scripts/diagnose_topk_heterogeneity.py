#!/usr/bin/env python3
"""Run a short dense B/4 trajectory and log top-k support heterogeneity."""

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
        default=str(ROOT / "configs/gen/B4_rev-drift_mae256.yaml"),
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--top-k-pos", type=int, default=16)
    parser.add_argument("--top-k-neg", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pos-per-sample", type=int, default=32)
    parser.add_argument("--neg-per-sample", type=int, default=16)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.update(
        {
            "name": Path(args.workdir).name,
            "seed": int(args.seed),
            "use_wandb": False,
            "console_log": True,
            "log_every_k": 1,
            "eval_at_start": False,
            "eval_per_step": 1_000_000_000,
            "save_per_step": 1_000_000_000,
            "keep_last": 0,
            "train_max_step_exclusive": int(args.steps),
            "batch_size": int(args.batch_size),
            "pos_per_sample": int(args.pos_per_sample),
            "neg_per_sample": int(args.neg_per_sample),
            # Follow the efficiency baseline while keeping the optimization
            # trajectory dense. The diagnostic computes hypothetical supports.
            "feature_loss_profile": "no_stage1",
            "layer_temperature_profile": "uniform",
            "rev_drift_top_p": 1.0,
            "drift_top_k_pos": 0,
            "drift_top_k_neg": 0,
            "topk_diagnostic_steps": int(args.steps),
            "topk_diagnostic_pos": int(args.top_k_pos),
            "topk_diagnostic_neg": int(args.top_k_neg),
            "compute_wpos_stats": False,
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
