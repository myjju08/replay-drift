#!/usr/bin/env python3
"""Calibrate reverse-kernel bandwidths to Wendland-R1.5 feature ESS."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_imagenet_gen as train_module  # noqa: E402
from drifting_core.imagenet_loss import (  # noqa: E402
    _adaptive_reverse_bandwidth,
    _cdist_batched,
    _ratio_of_means,
    _reverse_kernel_weights,
)
from scripts.kernel_ess_spec import CANDIDATES, REFERENCE  # noqa: E402


GROUPS = ("global", "norm_x", "stage1", "stage2", "stage3", "stage4", "overall")


def _candidate_entries():
    entries = [
        {
            "variant": REFERENCE["name"],
            "kernel": REFERENCE["kernel"],
            "shape": float(REFERENCE["shape"]),
            "r": float(REFERENCE["r"]),
            "reference": True,
        }
    ]
    for variant, spec in CANDIDATES.items():
        for r_value in spec["r_grid"]:
            entries.append(
                {
                    "variant": variant,
                    "kernel": spec["kernel"],
                    "shape": float(spec["shape"]),
                    "r": float(r_value),
                    "reference": False,
                }
            )
    return entries


def _group_ess(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mass = weights.sum(dim=2)
    square_mass = weights.square().sum(dim=2)
    ess = mass.square() / square_mass.clamp_min(1e-20)
    zero_fraction = mass.eq(0).float().mean()
    return ess.mean(), zero_fraction


@torch.no_grad()
def calibrate_from_features(
    *,
    gen_feats: Dict[str, torch.Tensor],
    pos_feats: Dict[str, torch.Tensor],
    neg_feats: Optional[Dict[str, torch.Tensor]],
    B: int,
    G: int,
    P: int,
    N: int,
    weight_neg: Optional[torch.Tensor],
    active_mask_pos: Optional[torch.Tensor],
    active_mask_neg: Optional[torch.Tensor],
    max_token_rows: int,
    output_path: Path,
) -> None:
    entries = _candidate_entries()
    device = next(iter(gen_feats.values())).device
    stats = torch.zeros(
        len(GROUPS), len(entries), 4, dtype=torch.float64, device=device
    )
    feature_counts = torch.zeros(len(GROUPS), dtype=torch.float64, device=device)
    group_to_index = {name: index for index, name in enumerate(GROUPS)}

    for name, gen_f in gen_feats.items():
        pos_f = pos_feats.get(name)
        if pos_f is None:
            continue
        group_name = train_module._feature_loss_group(name)
        if group_name not in group_to_index:
            continue
        group_indices = (group_to_index[group_name], group_to_index["overall"])
        T = gen_f.shape[1]
        gen_bt = rearrange(gen_f.detach().float(), "(b g) t d -> (b t) g d", b=B, g=G)
        pos_bt = rearrange(pos_f.detach().float(), "(b p) t d -> (b t) p d", b=B, p=P)
        if neg_feats is not None and name in neg_feats and N > 0:
            neg_bt = rearrange(
                neg_feats[name].detach().float(),
                "(b n) t d -> (b t) n d",
                b=B,
                n=N,
            )
        else:
            neg_bt = gen_bt.new_zeros(gen_bt.shape[0], 0, gen_bt.shape[2])

        old_gen = gen_bt
        targets = torch.cat([old_gen, neg_bt, pos_bt], dim=1)
        active_gen = gen_bt.new_ones(B, G)
        active_pos = (
            active_mask_pos.float()
            if active_mask_pos is not None
            else gen_bt.new_ones(B, P)
        )
        active_neg = (
            active_mask_neg.float()
            if active_mask_neg is not None
            else gen_bt.new_ones(B, N)
        )
        weights_neg = (
            weight_neg.float()
            if weight_neg is not None
            else gen_bt.new_ones(B, N)
        )
        target_weights = torch.cat(
            [active_gen, weights_neg * active_neg, active_pos], dim=1
        )
        target_weights = repeat(target_weights, "b m -> (b t) m", t=T)

        distances = _cdist_batched(old_gen, targets)
        scale = _ratio_of_means(
            distances * target_weights.unsqueeze(1),
            target_weights,
            use_global_stats=True,
        )
        normalized = distances / scale.clamp_min(1e-3)
        self_mask = F.pad(
            torch.eye(G, dtype=torch.float32, device=device),
            (0, N + P),
        ).unsqueeze(0)
        split_idx = G + N
        local_bandwidth = _adaptive_reverse_bandwidth(
            normalized,
            split_idx=split_idx,
            self_mask=self_mask,
            k_pos=32,
            k_neg=24,
            margin=1.05,
        )

        if normalized.shape[0] > max_token_rows:
            row_indices = torch.linspace(
                0,
                normalized.shape[0] - 1,
                steps=max_token_rows,
                device=device,
            ).round().long()
            normalized = normalized.index_select(0, row_indices)
            local_bandwidth = local_bandwidth.index_select(0, row_indices)
        base_scaled = normalized / local_bandwidth.clamp_min(1e-8)

        for candidate_index, candidate in enumerate(entries):
            scaled = base_scaled / float(candidate["r"])
            kernel_weights = _reverse_kernel_weights(
                scaled,
                kernel=str(candidate["kernel"]),
                shape=float(candidate["shape"]),
            ).masked_fill(self_mask.bool(), 0.0)
            neg_ess, neg_zero = _group_ess(kernel_weights[:, :, :split_idx])
            pos_ess, pos_zero = _group_ess(kernel_weights[:, :, split_idx:])
            values = torch.stack([pos_ess, neg_ess, pos_zero, neg_zero]).double()
            for group_index in group_indices:
                stats[group_index, candidate_index] += values
        for group_index in group_indices:
            feature_counts[group_index] += 1.0

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(feature_counts, op=dist.ReduceOp.SUM)
    stats = stats / feature_counts.clamp_min(1.0).view(-1, 1, 1)

    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    ref = stats[:, 0, :2]
    results = []
    for candidate_index, candidate in enumerate(entries):
        values = stats[:, candidate_index]
        ratios = torch.log(values[:, :2].clamp_min(1e-12) / ref.clamp_min(1e-12))
        group_score = float(ratios[:6].square().mean().sqrt().item())
        overall_score = float(ratios[6].square().mean().sqrt().item())
        result = dict(candidate)
        result.update(
            {
                "group_log_rmse": group_score,
                "overall_log_rmse": overall_score,
                "groups": {
                    group: {
                        "pos_ess": float(values[index, 0].item()),
                        "neg_ess": float(values[index, 1].item()),
                        "pos_zero_fraction": float(values[index, 2].item()),
                        "neg_zero_fraction": float(values[index, 3].item()),
                    }
                    for index, group in enumerate(GROUPS)
                },
            }
        )
        results.append(result)

    selected = {}
    for variant in CANDIDATES:
        choices = [row for row in results if row["variant"] == variant]
        best = min(choices, key=lambda row: row["group_log_rmse"])
        selected[variant] = {
            "kernel": best["kernel"],
            "shape": best["shape"],
            "r": best["r"],
            "group_log_rmse": best["group_log_rmse"],
            "overall_log_rmse": best["overall_log_rmse"],
            "groups": best["groups"],
        }

    payload = {
        "definition": "ESS=(sum_j K_ij)^2/sum_j K_ij^2 before mutual normalization",
        "reference": results[0],
        "selected": selected,
        "candidates": results[1:],
        "max_token_rows_per_feature_rank": int(max_token_rows),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[kernel-ess] JSON={output_path}", flush=True)
    for variant, best in selected.items():
        overall = best["groups"]["overall"]
        print(
            "[kernel-ess] "
            f"variant={variant} R={best['r']:g} "
            f"group_log_rmse={best['group_log_rmse']:.5f} "
            f"pos_ess={overall['pos_ess']:.3f} "
            f"neg_ess={overall['neg_ess']:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs/gen/B4_rev-drift_mae256.yaml"),
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-token-rows", type=int, default=64)
    args = parser.parse_args()

    cfg = train_module.load_yaml_config(args.config)
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
            "train_max_step_exclusive": 1,
            "batch_size": 10,
            "gen_per_label": 64,
            "pos_per_sample": 128,
            "neg_per_sample": 32,
            "drift_matching": "rev-drift",
            "compute_wpos_stats": False,
            "stochastic_feature_stage_loss": False,
            "feature_loss_group_weights": {"default": 1.0},
            "layer_temperature_multipliers": {"default": 1.0},
            "rev_drift_top_p": 1.0,
            "drift_top_k_pos": 0,
            "drift_top_k_neg": 0,
            "rev_drift_affinity_kernel": "wendland",
            "R_list": [1.5],
            "rev_drift_kernel_adaptive_k_pos": 32,
            "rev_drift_kernel_adaptive_k_neg": 24,
            "rev_drift_kernel_adaptive_margin": 1.05,
            "rev_drift_force_multiplier": 3.0,
        }
    )

    original_compute = train_module.compute_drift_loss_from_features
    state = {"complete": False}
    output_path = Path(args.output).resolve()

    def diagnostic_wrapper(*wrapper_args, **wrapper_kwargs):
        if not state["complete"]:
            calibrate_from_features(
                gen_feats=wrapper_kwargs["gen_feats"],
                pos_feats=wrapper_kwargs["pos_feats"],
                neg_feats=wrapper_kwargs["neg_feats"],
                B=wrapper_kwargs["B"],
                G=wrapper_kwargs["G"],
                P=wrapper_kwargs["P"],
                N=wrapper_kwargs["N"],
                weight_neg=wrapper_kwargs["weight_neg"],
                active_mask_pos=wrapper_kwargs["active_mask_pos"],
                active_mask_neg=wrapper_kwargs["active_mask_neg"],
                max_token_rows=int(args.max_token_rows),
                output_path=output_path,
            )
            state["complete"] = True
        return original_compute(*wrapper_args, **wrapper_kwargs)

    train_module.compute_drift_loss_from_features = diagnostic_wrapper
    rank, world_size, device = train_module.setup_distributed()
    Path(args.workdir).mkdir(parents=True, exist_ok=True)
    train_module.train_gen(cfg, args.workdir, rank, world_size, device)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
