#!/usr/bin/env python3
"""Train one ESS-calibrated full-feature B/4 reverse-kernel ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.kernel_ess_spec import CANDIDATES  # noqa: E402
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
    parser.add_argument(
        "--calibration",
        default=str(ROOT / "runs/diagnostics/reverse_kernel_ess_seed42_refined.json"),
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--variant", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated-epochs", type=float, default=40.0)
    parser.add_argument("--save-every-epochs", type=float, default=10.0)
    args = parser.parse_args()

    calibration_path = Path(args.calibration).resolve()
    calibration = json.loads(calibration_path.read_text())
    selected = calibration["selected"][args.variant]
    expected = CANDIDATES[args.variant]
    if selected["kernel"] != expected["kernel"]:
        raise ValueError(
            f"calibration kernel mismatch for {args.variant}: {selected['kernel']}"
        )
    if float(selected["shape"]) != float(expected["shape"]):
        raise ValueError(
            f"calibration shape mismatch for {args.variant}: {selected['shape']}"
        )

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
            "R_list": [float(selected["r"])],
            "rev_drift_affinity_kernel": str(selected["kernel"]),
            "rev_drift_kernel_shape": float(selected["shape"]),
            "rev_drift_kernel_adaptive_k_pos": 32,
            "rev_drift_kernel_adaptive_k_neg": 24,
            "rev_drift_kernel_adaptive_margin": 1.05,
            "rev_drift_force_multiplier": 3.0,
            "kernel_ess_calibration_json": str(calibration_path),
            "kernel_ess_reference": "wendland_r15",
            "kernel_ess_group_log_rmse": float(selected["group_log_rmse"]),
        }
    )

    rank, world_size, device = setup_distributed()
    if is_main_process(rank):
        overall = selected["groups"]["overall"]
        print(
            "[kernel-ess-run] "
            f"variant={args.variant} kernel={selected['kernel']} "
            f"shape={float(selected['shape']):g} R={float(selected['r']):g} "
            f"pos_ess={float(overall['pos_ess']):.3f} "
            f"neg_ess={float(overall['neg_ess']):.3f} "
            f"group_log_rmse={float(selected['group_log_rmse']):.5f} "
            f"calibration={calibration_path}",
            flush=True,
        )
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
