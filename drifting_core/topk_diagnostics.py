"""Diagnostics for sharing reverse-drift top-k supports across features.

The production reverse-drift objective selects top-k independently for every
feature token and temperature.  This module asks a different, read-only
question: after pooling each feature over tokens and temperatures, how similar
are the selected particle sets across feature objectives and encoder stages?

It also measures the union of the *actual* token/temperature selections across
all generated queries.  That union is the relevant upper bound on how many
real candidates could be removed before a deeper feature-encoder stage.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .imagenet_loss import _cdist_batched, _ratio_of_means


_GROUPS = ("global", "norm_x", "stage1", "stage2", "stage3", "stage4")


def _feature_group(name: str) -> Optional[str]:
    if name in ("global", "norm_x"):
        return name
    if name == "conv1" or name.startswith("conv1_"):
        return "stage1"
    for stage_index in range(1, 5):
        prefix = f"layer{stage_index}"
        if name == prefix or name.startswith(f"{prefix}_"):
            return f"stage{stage_index}"
    return None


def _as_bt_samples(
    feature: torch.Tensor,
    *,
    batch_size: int,
    sample_count: int,
) -> Tuple[torch.Tensor, int]:
    """Convert ``(B*C,T,D)`` to the loss layout ``(B*T,C,D)``."""
    if feature.ndim != 3 or feature.shape[0] != batch_size * sample_count:
        raise ValueError(
            "Unexpected feature shape: "
            f"shape={tuple(feature.shape)}, B={batch_size}, C={sample_count}"
        )
    token_count, feature_dim = feature.shape[1:]
    reshaped = feature.reshape(
        batch_size, sample_count, token_count, feature_dim
    )
    reshaped = reshaped.permute(0, 2, 1, 3).reshape(
        batch_size * token_count, sample_count, feature_dim
    )
    return reshaped.detach().float(), int(token_count)


def _support_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    pool_size = scores.shape[-1]
    keep = min(max(int(k), 1), pool_size)
    indices = scores.topk(keep, dim=-1, largest=True, sorted=False).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)


def _chance_adjusted(overlap: torch.Tensor | float, k: int, pool_size: int) -> float:
    raw = float(overlap)
    chance = min(int(k), int(pool_size)) / max(int(pool_size), 1)
    if chance >= 1.0:
        return 1.0
    return (raw - chance) / (1.0 - chance)


def _pair_overlap(mask_a: torch.Tensor, mask_b: torch.Tensor, k: int) -> float:
    keep = min(max(int(k), 1), mask_a.shape[-1])
    return float((mask_a & mask_b).sum(dim=-1).float().mean().item() / keep)


def _collection_overlap(masks: Sequence[torch.Tensor], k: int) -> Optional[float]:
    """Mean equal-k support recall over every feature pair.

    The Gram matrix computes the mean intersection size over all ``B*G``
    queries without launching one CUDA reduction per feature pair.
    """
    if len(masks) < 2:
        return None
    stacked = torch.stack(tuple(masks), dim=0)
    feature_count, batch_size, query_count, pool_size = stacked.shape
    flat = stacked.float().reshape(feature_count, -1)
    mean_intersection = (flat @ flat.transpose(0, 1)) / float(
        batch_size * query_count
    )
    pair_indices = torch.triu_indices(
        feature_count, feature_count, offset=1, device=stacked.device
    )
    keep = min(max(int(k), 1), pool_size)
    return float(
        mean_intersection[pair_indices[0], pair_indices[1]].mean().item()
        / keep
    )


def _cross_collection_overlap(
    left: Sequence[torch.Tensor],
    right: Sequence[torch.Tensor],
    k: int,
) -> Optional[float]:
    if not left or not right:
        return None
    left_stack = torch.stack(tuple(left), dim=0)
    right_stack = torch.stack(tuple(right), dim=0)
    _, batch_size, query_count, pool_size = left_stack.shape
    left_flat = left_stack.float().reshape(len(left), -1)
    right_flat = right_stack.float().reshape(len(right), -1)
    mean_intersection = (left_flat @ right_flat.transpose(0, 1)) / float(
        batch_size * query_count
    )
    keep = min(max(int(k), 1), pool_size)
    return float(mean_intersection.mean().item() / keep)


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = tuple(float(value) for value in values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def _average_metrics_across_ranks(metrics: Dict[str, float]) -> Dict[str, float]:
    if not (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return metrics
    keys = tuple(sorted(metrics))
    if not keys:
        return metrics
    device = torch.device("cuda", torch.cuda.current_device())
    values = torch.tensor(
        [metrics[key] for key in keys], dtype=torch.float64, device=device
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= float(dist.get_world_size())
    return {key: float(value) for key, value in zip(keys, values.cpu().tolist())}


@torch.no_grad()
def diagnose_reverse_topk_heterogeneity(
    *,
    gen_feats: Dict[str, torch.Tensor],
    pos_feats: Dict[str, torch.Tensor],
    neg_feats: Optional[Dict[str, torch.Tensor]],
    batch_size: int,
    gen_count: int,
    pos_count: int,
    neg_count: int,
    weight_neg: Optional[torch.Tensor],
    R_list: Tuple[float, ...],
    top_k_pos: int,
    top_k_neg: int,
    active_mask_pos: Optional[torch.Tensor] = None,
    active_mask_neg: Optional[torch.Tensor] = None,
    global_scale_stats: bool = True,
    feature_temperature_multipliers: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Measure whether one early support can represent later feature supports.

    No tensors used by training are modified, and every input is detached.
    Returned metrics are averaged across DDP ranks in one collective.
    """
    if not R_list:
        raise ValueError("R_list must contain at least one temperature")
    if int(top_k_pos) <= 0 or int(top_k_neg) <= 0:
        raise ValueError("diagnostic top_k_pos and top_k_neg must be positive")
    if pos_count <= 0 or gen_count <= 0:
        raise ValueError("positive and generated candidate counts must be positive")

    device = next(iter(gen_feats.values())).device
    pos_k = min(int(top_k_pos), int(pos_count))
    repulsive_count = int(gen_count + neg_count)
    neg_k = min(int(top_k_neg), repulsive_count)

    feature_masks: Dict[str, Dict[str, torch.Tensor]] = {"pos": {}, "neg": {}}
    feature_token_unions: Dict[str, Dict[str, torch.Tensor]] = {
        "pos": {},
        "neg": {},
    }
    group_masks: Dict[str, Dict[str, List[torch.Tensor]]] = {
        pool: {group: [] for group in _GROUPS} for pool in ("pos", "neg")
    }
    group_score_sums: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        pool: {group: None for group in _GROUPS} for pool in ("pos", "neg")
    }
    group_score_counts = {group: 0 for group in _GROUPS}
    # Unioning these masks across feature objectives gives the exact candidate
    # coverage required to preserve every current token/R top-k decision in a
    # stage. Averaging per-feature coverage alone would underestimate this.
    group_token_unions: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
        pool: {group: None for group in _GROUPS} for pool in ("pos", "neg")
    }
    group_stats: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        pool: {
            group: {
                "temperature_overlap": [],
                "token_to_pooled_overlap": [],
                "pooled_query_union": [],
                "token_query_union": [],
                "pooled_real_neg_union": [],
                "token_real_neg_union": [],
            }
            for group in _GROUPS
        }
        for pool in ("pos", "neg")
    }

    names = tuple(name for name in gen_feats if name in pos_feats)
    for name in names:
        group = _feature_group(name)
        if group is None:
            continue

        gen_bt, token_count = _as_bt_samples(
            gen_feats[name], batch_size=batch_size, sample_count=gen_count
        )
        pos_bt, pos_token_count = _as_bt_samples(
            pos_feats[name], batch_size=batch_size, sample_count=pos_count
        )
        if pos_token_count != token_count:
            raise ValueError(f"Token mismatch for {name}: gen={token_count}, pos={pos_token_count}")

        if neg_count > 0:
            if neg_feats is None or name not in neg_feats:
                raise ValueError(f"Missing negative feature {name!r}")
            neg_bt, neg_token_count = _as_bt_samples(
                neg_feats[name], batch_size=batch_size, sample_count=neg_count
            )
            if neg_token_count != token_count:
                raise ValueError(
                    f"Token mismatch for {name}: gen={token_count}, neg={neg_token_count}"
                )
        else:
            neg_bt = gen_bt.new_empty(gen_bt.shape[0], 0, gen_bt.shape[-1])

        weight_gen = gen_bt.new_ones(batch_size, gen_count)
        weight_pos = gen_bt.new_ones(batch_size, pos_count)
        if neg_count > 0:
            if weight_neg is None:
                weight_neg_image = gen_bt.new_ones(batch_size, neg_count)
            else:
                weight_neg_image = weight_neg.detach().float().to(device)
        else:
            weight_neg_image = gen_bt.new_empty(batch_size, 0)

        mask_pos_image = (
            active_mask_pos.detach().float().to(device)
            if active_mask_pos is not None
            else gen_bt.new_ones(batch_size, pos_count)
        )
        mask_neg_image = (
            active_mask_neg.detach().float().to(device)
            if active_mask_neg is not None
            else gen_bt.new_ones(batch_size, neg_count)
        )
        targets_active_image = torch.cat(
            [
                gen_bt.new_ones(batch_size, gen_count),
                mask_neg_image,
                mask_pos_image,
            ],
            dim=1,
        )
        targets_weight_image = torch.cat(
            [weight_gen, weight_neg_image, weight_pos], dim=1
        ) * targets_active_image
        targets_weight = targets_weight_image[:, None].expand(
            batch_size, token_count, -1
        ).reshape(batch_size * token_count, -1)

        old_gen = gen_bt
        targets = torch.cat([old_gen, neg_bt, pos_bt], dim=1)
        distances = _cdist_batched(old_gen, targets)
        scale = _ratio_of_means(
            distances * targets_weight.unsqueeze(1),
            targets_weight,
            use_global_stats=global_scale_stats,
        )
        normalized = distances / scale.clamp_min(1e-3)
        diagonal = torch.eye(gen_count, dtype=normalized.dtype, device=device)
        normalized.add_(
            F.pad(diagonal, (0, neg_count + pos_count)).unsqueeze(0),
            alpha=100.0,
        )

        multiplier = 1.0
        if feature_temperature_multipliers:
            multiplier = float(
                feature_temperature_multipliers.get(
                    name,
                    feature_temperature_multipliers.get(
                        group,
                        feature_temperature_multipliers.get("default", 1.0),
                    ),
                )
            )
        feature_temperatures = tuple(float(R) * multiplier for R in R_list)
        pooled_score_sums = {
            "pos": gen_bt.new_zeros(batch_size, gen_count, pos_count),
            "neg": gen_bt.new_zeros(batch_size, gen_count, repulsive_count),
        }
        temperature_masks: Dict[str, List[torch.Tensor]] = {"pos": [], "neg": []}
        token_union_masks = {
            "pos": torch.zeros(batch_size, pos_count, dtype=torch.bool, device=device),
            "neg": torch.zeros(
                batch_size, repulsive_count, dtype=torch.bool, device=device
            ),
        }

        split_index = repulsive_count
        for temperature in feature_temperatures:
            logits = -normalized / temperature
            row_affinity = F.softmax(logits, dim=2)
            column_affinity = F.softmax(logits, dim=1)
            affinity = (row_affinity * column_affinity).clamp_min(1e-6).sqrt_()
            affinity.mul_(targets_weight.unsqueeze(1))
            pool_weights = {
                "neg": affinity[:, :, :split_index],
                "pos": affinity[:, :, split_index:],
            }

            for pool, pool_size, keep in (
                ("pos", pos_count, pos_k),
                ("neg", repulsive_count, neg_k),
            ):
                probabilities = pool_weights[pool] / pool_weights[pool].sum(
                    dim=2, keepdim=True
                ).clamp_min(1e-12)
                token_scores = probabilities.reshape(
                    batch_size, token_count, gen_count, pool_size
                )
                pooled_scores = token_scores.mean(dim=1)
                pooled_score_sums[pool].add_(pooled_scores)
                pooled_mask = _support_mask(pooled_scores, keep)
                temperature_masks[pool].append(pooled_mask)

                token_indices = token_scores.topk(
                    keep, dim=-1, largest=True, sorted=False
                ).indices
                pooled_expanded = pooled_mask[:, None].expand(
                    -1, token_count, -1, -1
                )
                token_agreement = pooled_expanded.gather(
                    dim=3, index=token_indices
                ).float().mean()
                group_stats[pool][group]["token_to_pooled_overlap"].append(
                    float(token_agreement.item())
                )

                selected_union = torch.zeros(
                    batch_size, pool_size, dtype=torch.bool, device=device
                )
                selected_union.scatter_(
                    1, token_indices.reshape(batch_size, -1), True
                )
                token_union_masks[pool].logical_or_(selected_union)

            del affinity, row_affinity, column_affinity, logits

        for pool, pool_size, keep in (
            ("pos", pos_count, pos_k),
            ("neg", repulsive_count, neg_k),
        ):
            pooled_scores = pooled_score_sums[pool] / len(feature_temperatures)
            feature_mask = _support_mask(pooled_scores, keep)
            feature_masks[pool][name] = feature_mask
            feature_token_unions[pool][name] = token_union_masks[pool].clone()
            group_masks[pool][group].append(feature_mask)
            if group_score_sums[pool][group] is None:
                group_score_sums[pool][group] = pooled_scores.clone()
            else:
                group_score_sums[pool][group].add_(pooled_scores)

            temp_overlap = _collection_overlap(temperature_masks[pool], keep)
            if temp_overlap is not None:
                group_stats[pool][group]["temperature_overlap"].append(temp_overlap)

            pooled_union = feature_mask.any(dim=1).float().mean()
            token_union = token_union_masks[pool].float().mean()
            group_stats[pool][group]["pooled_query_union"].append(
                float(pooled_union.item())
            )
            group_stats[pool][group]["token_query_union"].append(
                float(token_union.item())
            )
            if pool == "neg" and neg_count > 0:
                pooled_real_union = feature_mask[:, :, gen_count:].any(dim=1).float().mean()
                token_real_union = token_union_masks[pool][:, gen_count:].float().mean()
                group_stats[pool][group]["pooled_real_neg_union"].append(
                    float(pooled_real_union.item())
                )
                group_stats[pool][group]["token_real_neg_union"].append(
                    float(token_real_union.item())
                )
            if group_token_unions[pool][group] is None:
                group_token_unions[pool][group] = token_union_masks[pool].clone()
            else:
                group_token_unions[pool][group].logical_or_(
                    token_union_masks[pool]
                )

        group_score_counts[group] += 1
        del distances, normalized, targets, old_gen, gen_bt, pos_bt, neg_bt

    metrics: Dict[str, float] = {
        "topk_diag/feature_count": float(len(feature_masks["pos"])),
        "topk_diag/pos/k": float(pos_k),
        "topk_diag/pos/pool_size": float(pos_count),
        "topk_diag/neg/k": float(neg_k),
        "topk_diag/neg/pool_size": float(repulsive_count),
    }

    for pool, pool_size, keep in (
        ("pos", pos_count, pos_k),
        ("neg", repulsive_count, neg_k),
    ):
        all_masks = tuple(feature_masks[pool].values())
        all_overlap = _collection_overlap(all_masks, keep)
        if all_overlap is not None:
            metrics[f"topk_diag/{pool}/all_feature_overlap"] = all_overlap
            metrics[f"topk_diag/{pool}/all_feature_overlap_adjusted"] = (
                _chance_adjusted(all_overlap, keep, pool_size)
            )

        consensus_masks: Dict[str, torch.Tensor] = {}
        for group in _GROUPS:
            masks = group_masks[pool][group]
            if not masks:
                continue
            within = _collection_overlap(masks, keep)
            if within is not None:
                metrics[f"topk_diag/{pool}/within_{group}_overlap"] = within
                metrics[f"topk_diag/{pool}/within_{group}_overlap_adjusted"] = (
                    _chance_adjusted(within, keep, pool_size)
                )
            score_sum = group_score_sums[pool][group]
            assert score_sum is not None
            consensus_masks[group] = _support_mask(
                score_sum / max(group_score_counts[group], 1), keep
            )
            for stat_name, values in group_stats[pool][group].items():
                mean_value = _mean(values)
                if mean_value is not None:
                    metrics[f"topk_diag/{pool}/{stat_name}_{group}"] = mean_value
            exact_group_union = group_token_unions[pool][group]
            assert exact_group_union is not None
            metrics[
                f"topk_diag/{pool}/token_query_union_across_features_{group}"
            ] = float(exact_group_union.float().mean().item())
            if pool == "neg" and neg_count > 0:
                metrics[
                    f"topk_diag/{pool}/token_real_neg_union_across_features_{group}"
                ] = float(
                    exact_group_union[:, gen_count:].float().mean().item()
                )

        for left_group, right_group in combinations(consensus_masks, 2):
            overlap = _pair_overlap(
                consensus_masks[left_group], consensus_masks[right_group], keep
            )
            pair_name = f"{left_group}_to_{right_group}"
            metrics[f"topk_diag/{pool}/consensus_{pair_name}_overlap"] = overlap
            metrics[
                f"topk_diag/{pool}/consensus_{pair_name}_overlap_adjusted"
            ] = _chance_adjusted(overlap, keep, pool_size)

        for left_group, right_group in combinations(_GROUPS, 2):
            cross = _cross_collection_overlap(
                group_masks[pool][left_group],
                group_masks[pool][right_group],
                keep,
            )
            if cross is not None:
                pair_name = f"{left_group}_to_{right_group}"
                metrics[f"topk_diag/{pool}/cross_{pair_name}_overlap"] = cross

        anchor = feature_masks[pool].get("layer1")
        if anchor is not None:
            for stage in ("stage2", "stage3", "stage4"):
                target_masks = group_masks[pool][stage]
                if not target_masks:
                    continue
                anchor_overlap = _mean(
                    _pair_overlap(anchor, target, keep) for target in target_masks
                )
                assert anchor_overlap is not None
                metrics[f"topk_diag/{pool}/layer1_to_{stage}_overlap"] = (
                    anchor_overlap
                )
                metrics[
                    f"topk_diag/{pool}/layer1_to_{stage}_overlap_adjusted"
                ] = _chance_adjusted(anchor_overlap, keep, pool_size)

        # Base encoder outputs are natural candidates for a cheap selector.
        # Report their exact current token/R/query union separately from the
        # group average, which also contains mean/std/patch/block objectives.
        for base_name in ("global", "norm_x", "layer1", "layer2", "layer3", "layer4"):
            base_union = feature_token_unions[pool].get(base_name)
            if base_union is None:
                continue
            metrics[f"topk_diag/{pool}/token_query_union_{base_name}"] = float(
                base_union.float().mean().item()
            )
            if pool == "neg" and neg_count > 0:
                metrics[f"topk_diag/{pool}/token_real_neg_union_{base_name}"] = float(
                    base_union[:, gen_count:].float().mean().item()
                )

        # Oracle coverage: candidates needed by at least one downstream
        # objective. This is the loss-preserving ceiling for an early exit
        # after stage 1, even if a perfect selector were available.
        downstream_union: Optional[torch.Tensor] = None
        for group in ("stage2", "stage3", "stage4"):
            group_union = group_token_unions[pool][group]
            if group_union is None:
                continue
            if downstream_union is None:
                downstream_union = group_union.clone()
            else:
                downstream_union.logical_or_(group_union)
        if downstream_union is not None:
            metrics[
                f"topk_diag/{pool}/token_query_union_across_downstream_features"
            ] = float(downstream_union.float().mean().item())
            if pool == "neg" and neg_count > 0:
                metrics[
                    f"topk_diag/{pool}/token_real_neg_union_across_downstream_features"
                ] = float(
                    downstream_union[:, gen_count:].float().mean().item()
                )

    return _average_metrics_across_ranks(metrics)
