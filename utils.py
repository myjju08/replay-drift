"""Utilities for ImageNet drifting training: EMA, LR scheduling, checkpointing."""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        foreach: bool = False,
    ):
        self.decay = decay
        self.foreach = bool(foreach)
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        if self.foreach:
            ema_params = list(self.shadow.parameters())
            model_params = list(model.parameters())
            torch._foreach_mul_(ema_params, self.decay)
            torch._foreach_add_(ema_params, model_params, alpha=1.0 - self.decay)
            return
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, Any]:
        return self.shadow.state_dict()

    def load_state_dict(self, sd: Dict[str, Any]):
        self.shadow.load_state_dict(sd)
