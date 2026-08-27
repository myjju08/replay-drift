"""Small, teacher-stabilized adapters for frozen MAE feature maps.

The generator loss must see a slowly moving metric.  The online module in this
file is therefore trained only from real ImageNet examples; a separate EMA
copy is threaded through the frozen MAE when drift features are constructed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

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
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be rank 2, got {embeddings.shape}")
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise ValueError("labels must be rank 1 and match embeddings")
    if embeddings.shape[0] < 2:
        raise ValueError("supervised contrastive loss needs at least two examples")
    if not float(temperature) > 0.0:
        raise ValueError("supervised contrastive temperature must be positive")

    z = F.normalize(embeddings.float(), dim=-1)
    logits = z @ z.transpose(0, 1)
    logits = logits / float(temperature)
    eye = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(eye, float("-inf"))
    positive = labels[:, None].eq(labels[None, :]) & ~eye
    positive_count = positive.sum(dim=1)
    if bool((positive_count == 0).any()):
        raise ValueError("every adapter example must have a same-label positive")
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(
        log_prob.masked_fill(~positive, 0.0).sum(dim=1)
        / positive_count.to(log_prob.dtype)
    ).mean()


class FeatureAdapterSystem(nn.Module):
    """Per-stage residual adapters plus small heads used only for real supervision."""

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
    ):
        super().__init__()
        self.stages = canonical_adapter_stages(stages)
        if not self.stages:
            raise ValueError("feature adapter must select at least one stage")
        self.use_ce = bool(use_ce)
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train on a small class-balanced subset of detached real features."""
        take = min(int(samples_per_class), int(positive_count))
        if take < 2:
            raise ValueError("feature adapter needs at least two positives per class")
        repeated_labels = labels[:, None].expand(batch_size, take).reshape(-1)
        total = torch.zeros((), device=labels.device, dtype=torch.float32)
        metrics: Dict[str, torch.Tensor] = {}

        for stage in self.stages:
            layer = f"layer{stage[-1]}"
            if layer not in stage_features:
                raise KeyError(f"MAE did not emit required adapter feature {layer}")
            feature = stage_features[layer].detach()
            if feature.shape[0] < batch_size * positive_count:
                raise ValueError(
                    f"{layer} has {feature.shape[0]} examples, expected "
                    f"at least {batch_size * positive_count}"
                )
            feature = feature[: batch_size * positive_count]
            feature = feature.reshape(
                batch_size, positive_count, *feature.shape[1:]
            )[:, :take]
            feature = feature.reshape(-1, *feature.shape[2:])
            adapted = self.adapters[stage](feature)
            pooled = adapted.float().mean(dim=(2, 3))
            projected = self.projectors[stage](pooled)
            supcon = supervised_contrastive_loss(
                projected, repeated_labels, temperature
            )
            base_power = feature.float().square().mean().clamp_min(1e-6)
            residual_ratio = (
                (adapted.float() - feature.float()).square().mean() / base_power
            )
            stage_loss = float(supcon_weight) * supcon
            if self.use_ce:
                ce = F.cross_entropy(
                    self.classifiers[stage](projected.float()), repeated_labels
                )
                stage_loss = stage_loss + float(ce_weight) * ce
                metrics[f"adapter/{stage}_ce"] = ce.detach()
            stage_loss = stage_loss + float(reg_weight) * residual_ratio
            total = total + stage_loss
            metrics[f"adapter/{stage}_supcon"] = supcon.detach()
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
