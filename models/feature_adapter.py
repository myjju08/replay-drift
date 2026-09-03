"""Small, teacher-stabilized adapters for frozen MAE feature maps.

The generator loss sees a frozen target copy of the adapter for the entire
generator step.  The online adapter is optimized separately, either from real
ImageNet supervision or from real positives plus detached generated samples.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_STAGES = ("stage1", "stage2", "stage3", "stage4")


def canonical_adapter_stages(keys: Iterable[str]) -> Tuple[str, ...]:
    """Normalize ``layer3``/``stage3`` spellings and preserve stage order."""
    selected = set()
    for raw in keys:
        key = str(raw).strip().lower()
        if key.startswith("layer"):
            key = f"stage{key[5:]}"
        if key not in _VALID_STAGES:
            raise ValueError(
                f"Unknown feature-adapter key {raw!r}; expected layer1..4 or stage1..4"
            )
        selected.add(key)
    return tuple(stage for stage in _VALID_STAGES if stage in selected)


class ResidualSpatialAdapter(nn.Module):
    """Identity-initialized 1x1 bottleneck adapter for a BCHW feature map."""

    def __init__(self, channels: int, bottleneck: int, dropout: float = 0.0):
        super().__init__()
        if channels <= 0 or bottleneck <= 0:
            raise ValueError("adapter channels and bottleneck must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("adapter dropout must be in [0, 1)")
        self.norm = nn.GroupNorm(1, channels, eps=1e-6)
        self.down = nn.Conv2d(channels, bottleneck, 1)
        self.dropout = (
            nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        )
        self.up = nn.Conv2d(bottleneck, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.dropout(F.silu(self.down(self.norm(x)))))
        return x + residual


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Local-batch supervised contrastive loss with self-pairs removed."""
    return multi_positive_info_nce_loss(
        embeddings,
        labels,
        embeddings,
        labels,
        temperature,
        exclude_matching_indices=True,
    )


def multi_positive_info_nce_loss(
    anchor_embeddings: torch.Tensor,
    anchor_labels: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_labels: torch.Tensor,
    temperature: float,
    *,
    exclude_matching_indices: bool = False,
) -> torch.Tensor:
    """Asymmetric local-batch multi-positive InfoNCE.

    For every anchor, all candidates with the same class label are positives
    and every valid candidate participates in the denominator.  The loss is
    the mean negative log-probability over the positive candidates, matching
    the supervised-contrastive multi-positive form.  ``exclude_matching_indices``
    removes the same-index candidate for every anchor. Candidates must begin
    with the embeddings corresponding to the anchors; extra candidates may
    follow that prefix.
    """
    if anchor_embeddings.ndim != 2:
        raise ValueError(
            f"anchor_embeddings must be rank 2, got {anchor_embeddings.shape}"
        )
    if candidate_embeddings.ndim != 2:
        raise ValueError(
            "candidate_embeddings must be rank 2, got "
            f"{candidate_embeddings.shape}"
        )
    if anchor_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("anchor and candidate embedding widths must match")
    if (
        anchor_labels.ndim != 1
        or anchor_labels.shape[0] != anchor_embeddings.shape[0]
    ):
        raise ValueError("anchor_labels must be rank 1 and match anchors")
    if (
        candidate_labels.ndim != 1
        or candidate_labels.shape[0] != candidate_embeddings.shape[0]
    ):
        raise ValueError("candidate_labels must be rank 1 and match candidates")
    if anchor_embeddings.shape[0] == 0 or candidate_embeddings.shape[0] == 0:
        raise ValueError("multi-positive InfoNCE needs non-empty inputs")
    if not float(temperature) > 0.0:
        raise ValueError("multi-positive InfoNCE temperature must be positive")

    anchors = F.normalize(anchor_embeddings.float(), dim=-1)
    candidates = F.normalize(candidate_embeddings.float(), dim=-1)
    logits = (anchors @ candidates.transpose(0, 1)) / float(temperature)
    valid = torch.ones_like(logits, dtype=torch.bool)
    if exclude_matching_indices:
        if anchor_embeddings.shape[0] > candidate_embeddings.shape[0]:
            raise ValueError(
                "exclude_matching_indices requires a candidate prefix at least "
                "as long as the anchor batch"
            )
        matching = torch.arange(logits.shape[0], device=logits.device)
        valid[matching, matching] = False
    if bool((valid.sum(dim=1) == 0).any()):
        raise ValueError("every InfoNCE anchor must have a valid candidate")

    positive = (
        anchor_labels[:, None].eq(candidate_labels[None, :]) & valid
    )
    positive_count = positive.sum(dim=1)
    if bool((positive_count == 0).any()):
        raise ValueError("every adapter example must have a same-label positive")

    logits_for_denominator = logits.masked_fill(~valid, float("-inf"))
    log_prob = logits - torch.logsumexp(
        logits_for_denominator, dim=1, keepdim=True
    )
    return -(
        log_prob.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.to(log_prob.dtype)
    ).mean()


class FeatureAdapterSystem(nn.Module):
    """Per-stage residual adapters plus small contrastive projection heads."""

    def __init__(
        self,
        stage_channels: Mapping[str, int],
        stages: Iterable[str],
        *,
        bottleneck: int = 64,
        projection_dim: int = 128,
        num_classes: int = 1000,
        dropout: float = 0.0,
        use_ce: bool = False,
        objective: str = "supcon",
    ):
        super().__init__()
        self.stages = canonical_adapter_stages(stages)
        if not self.stages:
            raise ValueError("feature adapter must select at least one stage")
        self.use_ce = bool(use_ce)
        self.objective = str(objective).lower().strip()
        if self.objective not in (
            "supcon",
            "supcon_ce",
            "real_anchor_multipos_infonce",
            "gen_real_multipos_infonce",
            "raw_drift_snr",
        ):
            raise ValueError(f"Unknown feature-adapter objective {objective!r}")
        self.adapters = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()
        for stage in self.stages:
            if stage not in stage_channels:
                raise ValueError(f"Missing channel width for {stage}")
            channels = int(stage_channels[stage])
            self.adapters[stage] = ResidualSpatialAdapter(
                channels, int(bottleneck), float(dropout)
            )
            if self.objective in (
                "real_anchor_multipos_infonce",
                "gen_real_multipos_infonce",
                "raw_drift_snr",
            ):
                # Make the contrastive gradient act directly on the adapted
                # MAE metric instead of letting an auxiliary projection head
                # absorb the task while remaining invisible to drift loss.
                self.projectors[stage] = nn.Identity()
            else:
                self.projectors[stage] = nn.Sequential(
                    nn.LayerNorm(channels),
                    nn.Linear(channels, int(projection_dim)),
                )
            if self.use_ce:
                self.classifiers[stage] = nn.Linear(
                    int(projection_dim), int(num_classes)
                )

    def forward(
        self,
        stage_features: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        *,
        batch_size: int,
        positive_count: int,
        samples_per_class: int,
        temperature: float,
        supcon_weight: float,
        ce_weight: float,
        reg_weight: float,
        generated_stage_features: Optional[Dict[str, torch.Tensor]] = None,
        generated_count: int = 0,
        generated_samples_per_class: int = 0,
        generated_anchor_weight: float = 1.0,
        real_anchor_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train on class-balanced subsets of detached MAE stage features."""
        take = min(int(samples_per_class), int(positive_count))
        if take < 2:
            raise ValueError("feature adapter needs at least two positives per class")
        if self.objective == "raw_drift_snr":
            if generated_stage_features is None:
                raise ValueError("raw_drift_snr requires generated stage features")
            generated_take = min(
                int(generated_samples_per_class), int(generated_count)
            )
            if take < 4 or generated_take < 4:
                raise ValueError(
                    "raw_drift_snr needs at least four real and generated "
                    "samples per class for two non-trivial banks"
                )
            if take % 2 != 0 or generated_take % 2 != 0:
                raise ValueError(
                    "raw_drift_snr real/generated sample counts must be even"
                )

            adapted_real: Dict[str, torch.Tensor] = {}
            adapted_generated: Dict[str, torch.Tensor] = {}
            residuals: Dict[str, list[torch.Tensor]] = {
                stage: [] for stage in self.stages
            }
            for name, raw_real in stage_features.items():
                stage = None
                for candidate in self.stages:
                    layer = f"layer{candidate[-1]}"
                    if name == layer or name.startswith(f"{layer}_"):
                        stage = candidate
                        break
                if stage is None:
                    continue
                if name not in generated_stage_features:
                    raise KeyError(
                        f"MAE did not emit matching generated raw map {name}"
                    )
                raw_generated = generated_stage_features[name]
                expected_real = batch_size * positive_count
                expected_generated = batch_size * generated_count
                if raw_real.shape[0] < expected_real:
                    raise ValueError(
                        f"{name} has {raw_real.shape[0]} real examples, "
                        f"expected at least {expected_real}"
                    )
                if raw_generated.shape[0] < expected_generated:
                    raise ValueError(
                        f"{name} has {raw_generated.shape[0]} generated examples, "
                        f"expected at least {expected_generated}"
                    )
                real = raw_real.detach()[:expected_real].reshape(
                    batch_size, positive_count, *raw_real.shape[1:]
                )[:, :take]
                generated = raw_generated.detach()[:expected_generated].reshape(
                    batch_size, generated_count, *raw_generated.shape[1:]
                )[:, :generated_take]
                real = real.reshape(-1, *real.shape[2:])
                generated = generated.reshape(-1, *generated.shape[2:])
                combined = torch.cat([real, generated], dim=0)
                adapted = self.adapters[stage](combined)
                adapted_real[name] = adapted[: real.shape[0]]
                adapted_generated[name] = adapted[real.shape[0] :]
                with torch.no_grad():
                    residuals[stage].append(
                        (adapted.float() - combined.float()).square().mean()
                        / combined.float().square().mean().clamp_min(1.0e-6)
                    )

            if not adapted_real:
                raise ValueError("raw_drift_snr received no selected stage maps")
            metrics: Dict[str, torch.Tensor] = {}
            for stage, values in residuals.items():
                if values:
                    metrics[f"adapter/{stage}_residual_ratio"] = torch.stack(
                        values
                    ).mean()
            return (adapted_real, adapted_generated), metrics

        repeated_labels = labels[:, None].expand(batch_size, take).reshape(-1)
        use_generated = self.objective in (
            "real_anchor_multipos_infonce",
            "gen_real_multipos_infonce",
        )
        generated_take = 0
        generated_labels: Optional[torch.Tensor] = None
        if use_generated:
            if generated_stage_features is None:
                raise ValueError(
                    f"{self.objective} requires generated stage features"
                )
            generated_take = min(
                int(generated_samples_per_class), int(generated_count)
            )
            if generated_take < 1:
                raise ValueError(
                    "feature adapter needs at least one generated sample per class"
                )
            if float(generated_anchor_weight) < 0.0:
                raise ValueError("generated_anchor_weight must be non-negative")
            if float(real_anchor_weight) < 0.0:
                raise ValueError("real_anchor_weight must be non-negative")
            if float(generated_anchor_weight) + float(real_anchor_weight) <= 0.0:
                raise ValueError("at least one adapter anchor weight must be positive")
            if self.objective == "real_anchor_multipos_infonce" and (
                float(generated_anchor_weight) != 0.0
                or float(real_anchor_weight) != 1.0
            ):
                raise ValueError(
                    "real_anchor_multipos_infonce fixes generated_anchor_weight=0 "
                    "and real_anchor_weight=1"
                )
            generated_labels = labels[:, None].expand(
                batch_size, generated_take
            ).reshape(-1)
        total = torch.zeros((), device=labels.device, dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {}

        for stage in self.stages:
            layer = f"layer{stage[-1]}"
            if layer not in stage_features:
                raise KeyError(f"MAE did not emit required adapter feature {layer}")
            real_feature = stage_features[layer].detach()
            if real_feature.shape[0] < batch_size * positive_count:
                raise ValueError(
                    f"{layer} has {real_feature.shape[0]} examples, expected "
                    f"at least {batch_size * positive_count}"
                )
            real_feature = real_feature[: batch_size * positive_count]
            real_feature = real_feature.reshape(
                batch_size, positive_count, *real_feature.shape[1:]
            )[:, :take]
            real_feature = real_feature.reshape(-1, *real_feature.shape[2:])
            adapted_real = self.adapters[stage](real_feature)
            pooled_real = adapted_real.float().mean(dim=(2, 3))
            projected_real = self.projectors[stage](pooled_real)
            if float(reg_weight) > 0.0:
                real_base_power = (
                    real_feature.float().square().mean().clamp_min(1e-6)
                )
                real_residual_ratio = (
                    (adapted_real.float() - real_feature.float()).square().mean()
                    / real_base_power
                )
            else:
                # Keep the diagnostic without retaining a large residual graph
                # when the experiment explicitly disables this regularizer.
                with torch.no_grad():
                    real_base_power = (
                        real_feature.float().square().mean().clamp_min(1e-6)
                    )
                    real_residual_ratio = (
                        (adapted_real.float() - real_feature.float()).square().mean()
                        / real_base_power
                    )
            residual_ratio = real_residual_ratio

            if use_generated:
                assert generated_stage_features is not None
                assert generated_labels is not None
                if layer not in generated_stage_features:
                    raise KeyError(
                        f"MAE did not emit required generated adapter feature {layer}"
                    )
                generated_feature = generated_stage_features[layer].detach()
                if generated_feature.shape[0] < batch_size * generated_count:
                    raise ValueError(
                        f"generated {layer} has {generated_feature.shape[0]} examples, "
                        f"expected at least {batch_size * generated_count}"
                    )
                generated_feature = generated_feature[: batch_size * generated_count]
                generated_feature = generated_feature.reshape(
                    batch_size, generated_count, *generated_feature.shape[1:]
                )[:, :generated_take]
                generated_feature = generated_feature.reshape(
                    -1, *generated_feature.shape[2:]
                )
                adapted_generated = self.adapters[stage](generated_feature)
                pooled_generated = adapted_generated.float().mean(dim=(2, 3))
                projected_generated = self.projectors[stage](pooled_generated)
                if self.objective == "real_anchor_multipos_infonce":
                    combined_candidates = torch.cat(
                        [projected_real, projected_generated], dim=0
                    )
                    combined_candidate_labels = torch.cat(
                        [repeated_labels, generated_labels], dim=0
                    )
                    contrastive = multi_positive_info_nce_loss(
                        projected_real,
                        repeated_labels,
                        combined_candidates,
                        combined_candidate_labels,
                        temperature,
                        exclude_matching_indices=True,
                    )
                else:
                    generated_to_real = multi_positive_info_nce_loss(
                        projected_generated,
                        generated_labels,
                        projected_real,
                        repeated_labels,
                        temperature,
                    )
                    if float(real_anchor_weight) > 0.0:
                        real_to_real = multi_positive_info_nce_loss(
                            projected_real,
                            repeated_labels,
                            projected_real,
                            repeated_labels,
                            temperature,
                            exclude_matching_indices=True,
                        )
                    else:
                        real_to_real = generated_to_real.new_zeros(())
                    anchor_weight_sum = (
                        float(generated_anchor_weight) + float(real_anchor_weight)
                    )
                    contrastive = (
                        float(generated_anchor_weight) * generated_to_real
                        + float(real_anchor_weight) * real_to_real
                    ) / anchor_weight_sum
                residual_context = (
                    torch.enable_grad()
                    if float(reg_weight) > 0.0
                    else torch.no_grad()
                )
                with residual_context:
                    generated_base_power = (
                        generated_feature.float().square().mean().clamp_min(1e-6)
                    )
                    generated_residual_ratio = (
                        (
                            adapted_generated.float()
                            - generated_feature.float()
                        ).square().mean()
                        / generated_base_power
                    )
                residual_ratio = 0.5 * (
                    real_residual_ratio + generated_residual_ratio
                )
                metrics[
                    f"adapter/{stage}_generated_residual_ratio"
                ] = generated_residual_ratio.detach()
                metrics[f"adapter/{stage}_infonce"] = contrastive.detach()
                if self.objective == "real_anchor_multipos_infonce":
                    metrics[
                        f"adapter/{stage}_real_anchor_infonce"
                    ] = contrastive.detach()
                    with torch.no_grad():
                        normalized_real = F.normalize(
                            projected_real.float(), dim=-1
                        )
                        normalized_generated = F.normalize(
                            projected_generated.float(), dim=-1
                        )
                        normalized_candidates = torch.cat(
                            [normalized_real, normalized_generated], dim=0
                        )
                        candidate_labels = torch.cat(
                            [repeated_labels, generated_labels], dim=0
                        )
                        cosine = normalized_real @ normalized_candidates.transpose(
                            0, 1
                        )
                        valid_pairs = torch.ones_like(cosine, dtype=torch.bool)
                        matching = torch.arange(
                            normalized_real.shape[0], device=cosine.device
                        )
                        valid_pairs[matching, matching] = False
                        positive_pairs = (
                            repeated_labels[:, None].eq(candidate_labels[None, :])
                            & valid_pairs
                        )
                        negative_pairs = (
                            repeated_labels[:, None].ne(candidate_labels[None, :])
                            & valid_pairs
                        )
                        generated_columns = torch.arange(
                            candidate_labels.shape[0], device=cosine.device
                        ).ge(normalized_real.shape[0])
                        generated_positive_pairs = (
                            positive_pairs & generated_columns[None, :]
                        )
                        real_positive_pairs = (
                            positive_pairs & ~generated_columns[None, :]
                        )
                        generated_negative_pairs = (
                            negative_pairs & generated_columns[None, :]
                        )
                        real_negative_pairs = (
                            negative_pairs & ~generated_columns[None, :]
                        )
                        metrics[f"adapter/{stage}_positive_cosine"] = cosine[
                            positive_pairs
                        ].mean()
                        metrics[
                            f"adapter/{stage}_same_class_generated_cosine"
                        ] = cosine[generated_positive_pairs].mean()
                        negative_cosines = cosine[negative_pairs]
                        metrics[f"adapter/{stage}_negative_cosine"] = (
                            negative_cosines.mean()
                            if negative_cosines.numel() > 0
                            else cosine.new_zeros(())
                        )
                        metrics[f"adapter/{stage}_positives_per_anchor"] = (
                            positive_pairs.sum(dim=1).float().mean()
                        )
                        metrics[
                            f"adapter/{stage}_real_positives_per_anchor"
                        ] = real_positive_pairs.sum(dim=1).float().mean()
                        metrics[
                            f"adapter/{stage}_generated_positives_per_anchor"
                        ] = generated_positive_pairs.sum(dim=1).float().mean()
                        metrics[f"adapter/{stage}_negatives_per_anchor"] = (
                            negative_pairs.sum(dim=1).float().mean()
                        )
                        metrics[
                            f"adapter/{stage}_real_negatives_per_anchor"
                        ] = real_negative_pairs.sum(dim=1).float().mean()
                        metrics[
                            f"adapter/{stage}_generated_negatives_per_anchor"
                        ] = generated_negative_pairs.sum(dim=1).float().mean()
                        metrics[f"adapter/{stage}_candidate_count"] = (
                            cosine.new_tensor(float(candidate_labels.shape[0]))
                        )
                        metrics[
                            f"adapter/{stage}_valid_candidates_per_anchor"
                        ] = cosine.new_tensor(float(candidate_labels.shape[0] - 1))
                else:
                    metrics[
                        f"adapter/{stage}_gen_to_real_infonce"
                    ] = generated_to_real.detach()
                    metrics[
                        f"adapter/{stage}_real_to_real_infonce"
                    ] = real_to_real.detach()
                    with torch.no_grad():
                        normalized_generated = F.normalize(
                            projected_generated.float(), dim=-1
                        )
                        normalized_real = F.normalize(
                            projected_real.float(), dim=-1
                        )
                        cosine = normalized_generated @ normalized_real.transpose(
                            0, 1
                        )
                        positive_pairs = generated_labels[:, None].eq(
                            repeated_labels[None, :]
                        )
                        metrics[f"adapter/{stage}_positive_cosine"] = cosine[
                            positive_pairs
                        ].mean()
                        negative_cosines = cosine[~positive_pairs]
                        metrics[f"adapter/{stage}_negative_cosine"] = (
                            negative_cosines.mean()
                            if negative_cosines.numel() > 0
                            else cosine.new_zeros(())
                        )
                        positives_per_anchor = positive_pairs.sum(dim=1).float()
                        metrics[f"adapter/{stage}_positives_per_anchor"] = (
                            positives_per_anchor.mean()
                        )
                        metrics[f"adapter/{stage}_real_candidate_count"] = (
                            cosine.new_tensor(float(projected_real.shape[0]))
                        )
            else:
                contrastive = supervised_contrastive_loss(
                    projected_real, repeated_labels, temperature
                )
                metrics[f"adapter/{stage}_supcon"] = contrastive.detach()

            stage_loss = float(supcon_weight) * contrastive
            if self.use_ce:
                ce = F.cross_entropy(
                    self.classifiers[stage](projected_real.float()), repeated_labels
                )
                stage_loss = stage_loss + float(ce_weight) * ce
                metrics[f"adapter/{stage}_ce"] = ce.detach()
            if float(reg_weight) > 0.0:
                stage_loss = stage_loss + float(reg_weight) * residual_ratio
            total = total + stage_loss
            metrics[
                f"adapter/{stage}_real_residual_ratio"
            ] = real_residual_ratio.detach()
            metrics[f"adapter/{stage}_residual_ratio"] = residual_ratio.detach()

        total = total / float(len(self.stages))
        metrics["adapter/loss"] = total.detach()
        return total, metrics


@torch.no_grad()
def update_adapter_ema(
    target: FeatureAdapterSystem,
    online: nn.Module,
    decay: float,
) -> None:
    """EMA-update a target adapter from a raw or DDP-wrapped online module."""
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("feature adapter EMA decay must be in [0, 1)")
    source = online.module if hasattr(online, "module") else online
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.lerp_(source_param.detach(), 1.0 - float(decay))
    for target_buffer, source_buffer in zip(target.buffers(), source.buffers()):
        target_buffer.copy_(source_buffer.detach())
