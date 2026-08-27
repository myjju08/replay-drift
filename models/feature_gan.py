"""Lightweight adversarial heads on frozen MAE feature maps.

The discriminator operates on terminal MAE stage maps that are already
computed by the drifting objective.  It never updates the MAE backbone.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_VALID_STAGES = ("stage1", "stage2", "stage3", "stage4")


def canonical_feature_gan_stages(stages: Iterable[str]) -> Tuple[str, ...]:
    """Normalize ``layerN``/``stageN`` names and preserve depth order."""
    selected = set()
    for raw_name in stages:
        name = str(raw_name).lower().strip()
        if name.startswith("layer") and name[5:].isdigit():
            name = f"stage{name[5:]}"
        if name not in _VALID_STAGES:
            raise ValueError(
                f"Unknown feature-GAN stage {raw_name!r}; expected layer1..4 "
                "or stage1..4"
            )
        selected.add(name)
    if not selected:
        raise ValueError("At least one feature-GAN stage must be selected")
    return tuple(stage for stage in _VALID_STAGES if stage in selected)


class _ConditionalStageHead(nn.Module):
    """Small spatial discriminator with a projection-label term."""

    def __init__(self, channels: int, hidden_channels: int, num_classes: int):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.norm = nn.GroupNorm(1, channels)
        self.input = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.spatial = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.patch_score = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.class_embedding = nn.Embedding(num_classes, hidden_channels)

        nn.init.normal_(self.class_embedding.weight, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                f"Expected BCHW feature map, got shape={tuple(features.shape)}"
            )
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                "Feature-GAN labels must have shape (batch,), got "
                f"features={tuple(features.shape)} labels={tuple(labels.shape)}"
            )
        hidden = F.silu(self.input(self.norm(features.float())))
        hidden = F.silu(self.spatial(hidden))
        patch_score = self.patch_score(hidden).mean(dim=(1, 2, 3))
        pooled = hidden.mean(dim=(2, 3))
        projection = (
            pooled * self.class_embedding(labels).to(pooled.dtype)
        ).sum(dim=1) / math.sqrt(self.hidden_channels)
        return patch_score + projection


class FrozenFeatureDiscriminator(nn.Module):
    """Class-conditional discriminator over one or more frozen-MAE stages."""

    def __init__(
        self,
        stage_channels: Dict[str, int],
        stages: Sequence[str],
        *,
        hidden_channels: int = 128,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.stages = canonical_feature_gan_stages(stages)
        self.heads = nn.ModuleDict(
            {
                stage: _ConditionalStageHead(
                    int(stage_channels[stage]), int(hidden_channels), int(num_classes)
                )
                for stage in self.stages
            }
        )

        # These buffers make gradient-ratio calibration resumable and keep it
        # synchronized with discriminator checkpoints.
        self.register_buffer("drift_grad_ema", torch.tensor(0.0))
        self.register_buffer("adversarial_grad_ema", torch.tensor(0.0))
        self.register_buffer("gradient_unit_scale", torch.tensor(1.0))
        self.register_buffer("gradient_ratio_updates", torch.tensor(0, dtype=torch.long))

    def forward_per_stage(
        self, features: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        missing = [stage for stage in self.stages if stage not in features]
        if missing:
            raise KeyError(f"Missing feature-GAN stage maps: {missing}")
        return {
            stage: self.heads[stage](features[stage], labels)
            for stage in self.stages
        }

    def forward(
        self, features: Dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        scores = tuple(self.forward_per_stage(features, labels).values())
        return torch.stack(scores, dim=0).mean(dim=0)

    @torch.no_grad()
    def update_gradient_calibration(
        self,
        drift_norm: torch.Tensor | float,
        adversarial_norm: torch.Tensor | float,
        *,
        ema_decay: float = 0.9,
        eps: float = 1.0e-12,
    ) -> None:
        """Track the scale mapping adversarial feature gradients to drift gradients."""
        decay = float(ema_decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"ema_decay must be in [0, 1), got {decay}")
        drift = torch.as_tensor(
            drift_norm, device=self.drift_grad_ema.device, dtype=torch.float32
        )
        adversarial = torch.as_tensor(
            adversarial_norm,
            device=self.adversarial_grad_ema.device,
            dtype=torch.float32,
        )
        if int(self.gradient_ratio_updates.item()) == 0:
            self.drift_grad_ema.copy_(drift)
            self.adversarial_grad_ema.copy_(adversarial)
        else:
            self.drift_grad_ema.mul_(decay).add_(drift, alpha=1.0 - decay)
            self.adversarial_grad_ema.mul_(decay).add_(
                adversarial, alpha=1.0 - decay
            )
        self.gradient_unit_scale.copy_(
            self.drift_grad_ema / self.adversarial_grad_ema.clamp_min(eps)
        )
        self.gradient_ratio_updates.add_(1)


def discriminator_hinge_loss(
    real_scores: torch.Tensor, fake_scores: torch.Tensor
) -> torch.Tensor:
    """Standard discriminator hinge objective."""
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()


def generator_hinge_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """Generator-side non-saturating hinge objective."""
    return -fake_scores.mean()


__all__ = [
    "FrozenFeatureDiscriminator",
    "canonical_feature_gan_stages",
    "discriminator_hinge_loss",
    "generator_hinge_loss",
]
