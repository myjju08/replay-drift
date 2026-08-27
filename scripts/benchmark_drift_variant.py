#!/usr/bin/env python3
"""Short, matched multi-GPU throughput benchmark for drift-loss variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_imagenet_gen import (  # noqa: E402
    load_yaml_config,
    setup_distributed,
    train_gen,
)


VARIANTS = {
    "forward3": {
        "drift_matching": "fwd-drift",
        "R_list": [0.4, 0.10, 0.04],
    },
    "reverse3": {
        "drift_matching": "rev-drift",
        "R_list": [0.2, 0.05, 0.02],
    },
    "reverse6": {
        "drift_matching": "rev-drift",
        # Union of the three forward and three reverse temperatures.
        "R_list": [0.4, 0.2, 0.10, 0.05, 0.04, 0.02],
    },
    "dual3x3_shared": {
        "drift_matching": "dual-drift",
        "R_list": [0.4, 0.10, 0.04],
        "mix_baseline_R_list": [0.2, 0.05, 0.02],
        "dual_drift_share_distances": True,
        # Hold the benchmark mixture fixed so changing model state does not
        # introduce a schedule difference between repeated timing runs.
        "mix_alpha_adaptive": True,
        "mix_alpha_adaptive_mode": "hedge_no_time",
        "mix_alpha_adaptive_initial": 0.5,
        "mix_alpha_adaptive_gamma": 0.0,
        "mix_alpha_adaptive_eta": 0.0,
        "mix_alpha_adaptive_decay": 0.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--top-k-pos", type=int, default=0)
    parser.add_argument("--top-k-neg", type=int, default=0)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    cfg.update(VARIANTS[args.variant])
    cfg.update(
        {
            "name": f"benchmark_{args.variant}",
            "use_wandb": False,
            "console_log": True,
            "log_every_k": 1,
            "eval_at_start": False,
            "eval_per_step": 1_000_000_000,
            "save_per_step": 1_000_000_000,
            "keep_last": 0,
            "train_max_step_exclusive": int(args.steps),
            "compute_wpos_stats": False,
            "stochastic_feature_stage_loss": False,
            "prune_skipped_feature_tensors": False,
            # Synchronize timing boundaries and record true CUDA peak memory.
            "profile_train_step": True,
            "rev_drift_top_p": 1.0,
            "fwd_drift_top_p": 1.0,
            "drift_top_k_pos": int(args.top_k_pos),
            "drift_top_k_neg": int(args.top_k_neg),
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
