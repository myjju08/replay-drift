#!/usr/bin/env python3
"""Compare OLD (commit 4f87011) vs NEW self-mask behavior in drift_loss_imagenet_colwise.

Uses the same training setup as the running experiment: same YAML config + real
ImageNet latents pushed into the memory bank + optional checkpoint for the generator.

Captures a real (gen, fixed_pos, fixed_neg, weight_neg) batch by monkey-patching
drift_loss_imagenet_colwise to record its arguments on the first call, then
re-runs the loss with OLD and NEW self-mask strategies on the same tensors.
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory_bank import ArrayMemoryBank  # noqa: E402
from models.imagenet_generator import build_ditgen_from_config  # noqa: E402
from train.train_data import create_imagenet_split  # noqa: E402
import drifting_core.imagenet_loss as il  # noqa: E402
from drifting_core.imagenet_loss import _cdist_batched  # noqa: E402
import train_imagenet_gen as tig  # noqa: E402
from train_imagenet_gen import (  # noqa: E402
    build_optimizer,
    load_mae,
    load_yaml_config,
    train_step,
)


CAPTURED_LIST: List[Dict[str, Any]] = []
# Toggle for the patched loss. When PATCH_MODE is None the patch is pass-through
# (call original loss). Otherwise it's a dict {self_mask_on_raw, scale_override}.
SELF_MASK_ON_RAW: Optional[bool] = None
PATCH_SCALE_OVERRIDE: Optional[str] = None


def _patched_colwise(*args, **kwargs):
    """Record every call's args. If SELF_MASK_ON_RAW is set, route to the toggleable
    implementation so backprop uses the chosen self-mask strategy; otherwise call
    the original loss (capture-only mode)."""
    gen = kwargs.get("gen", args[0] if args else None)
    fixed_pos = kwargs.get("fixed_pos", args[1] if len(args) > 1 else None)
    fixed_neg = kwargs.get("fixed_neg", args[2] if len(args) > 2 else None)
    weight_gen = kwargs.get("weight_gen")
    weight_pos = kwargs.get("weight_pos")
    weight_neg = kwargs.get("weight_neg")
    R_list = kwargs.get("R_list", (0.02, 0.05, 0.2))
    coupling = kwargs.get("coupling", False)

    if SELF_MASK_ON_RAW is None:
        CAPTURED_LIST.append({
            "gen": gen.detach().clone(),
            "fixed_pos": fixed_pos.detach().clone(),
            "fixed_neg": fixed_neg.detach().clone() if fixed_neg is not None else None,
            "weight_gen": weight_gen.detach().clone() if weight_gen is not None else None,
            "weight_pos": weight_pos.detach().clone() if weight_pos is not None else None,
            "weight_neg": weight_neg.detach().clone() if weight_neg is not None else None,
            "R_list": tuple(R_list),
            "coupling": bool(coupling),
        })
        return _original_colwise(*args, **kwargs)

    # Grad-check mode: delegate to the faithful clone with the self-mask toggle.
    # This supports ALL the extra kwargs (active_mask, per_sample_fnorm, etc.)
    # that the real training loop might pass, so it's a true drop-in replacement.
    loss, info = colwise_full_toggleable(
        gen=gen, fixed_pos=fixed_pos, fixed_neg=fixed_neg,
        weight_gen=weight_gen, weight_pos=weight_pos, weight_neg=weight_neg,
        R_list=R_list, coupling=coupling,
        compute_wpos_stats=kwargs.get("compute_wpos_stats", False),
        per_sample_fnorm=kwargs.get("per_sample_fnorm", False),
        active_mask_pos=kwargs.get("active_mask_pos"),
        active_mask_neg=kwargs.get("active_mask_neg"),
        decouple_weight_from_coupling=kwargs.get("decouple_weight_from_coupling", False),
        self_mask_on_raw=SELF_MASK_ON_RAW,
        scale_override=PATCH_SCALE_OVERRIDE,
    )
    return loss, info


_original_colwise = il.drift_loss_imagenet_colwise


def colwise_full_toggleable(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    coupling: bool = False,
    compute_wpos_stats: bool = False,
    per_sample_fnorm: bool = False,
    active_mask_pos: Optional[torch.Tensor] = None,
    active_mask_neg: Optional[torch.Tensor] = None,
    decouple_weight_from_coupling: bool = False,
    # EXTRA KNOBS for ablation:
    self_mask_on_raw: bool = False,
    scale_override: Optional[str] = None,   # None | "clean" | "polluted"
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Faithful clone of `drifting_core.imagenet_loss.drift_loss_imagenet_colwise`
    with TWO extra flags:

    - self_mask_on_raw:
        False (default) → current behavior (+100 on normalized dist, after scale)
        True            → commit 4f87011 behavior (+100 on raw dist, before scale)

    - scale_override (for ablation):
        None        → scale follows self_mask_on_raw (natural mode)
        "clean"     → force scale to be computed WITHOUT the +100 (NEW-style scale)
        "polluted"  → force scale to be computed WITH the +100 (OLD-style scale)

      This lets us run "OLD mask + NEW scale" and "NEW mask + OLD scale" to
      prove whether scale pollution is the entire cause of OLD ≠ NEW.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen        = gen.float()
    fixed_pos  = fixed_pos.float()
    fixed_neg  = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    if active_mask_pos is None:
        active_mask_pos = gen.new_ones(B, C_p)
    else:
        active_mask_pos = active_mask_pos.to(gen.device).float()
    if active_mask_neg is None:
        active_mask_neg = gen.new_ones(B, C_n)
    else:
        active_mask_neg = active_mask_neg.to(gen.device).float()
    active_mask_gen = gen.new_ones(B, C_g)
    neg_targets_active = torch.cat([active_mask_gen, active_mask_neg], dim=1)

    old_gen = gen.detach()
    neg_targets = torch.cat([old_gen, fixed_neg], dim=1)

    dist_pos = _cdist_batched(old_gen, fixed_pos)
    dist_neg_raw = _cdist_batched(old_gen, neg_targets)

    diag = torch.eye(C_g, dtype=torch.float32, device=gen.device)
    neg_diag_mask = F.pad(diag, (0, C_n)) * 100.0

    # Compute both scale variants so we can pick independently of self_mask_on_raw.
    neg_targets_w = torch.cat([weight_gen, weight_neg], dim=1) * neg_targets_active
    eff_weight_pos = weight_pos * active_mask_pos
    all_targets_w = torch.cat([eff_weight_pos, neg_targets_w], dim=1)
    weighted_dist_pos = dist_pos * eff_weight_pos.unsqueeze(1)

    #   clean scale = what NEW uses (raw dist only; no +100)
    dist_neg_clean = dist_neg_raw
    weighted_dist_neg_clean = dist_neg_clean * neg_targets_w.unsqueeze(1)
    wd_all_clean = torch.cat([weighted_dist_pos, weighted_dist_neg_clean], dim=2)
    scale_clean = wd_all_clean.mean() / all_targets_w.mean().clamp(min=1e-8)

    #   polluted scale = what OLD uses (+100 on diagonal enters the mean)
    dist_neg_pol = dist_neg_raw + neg_diag_mask.unsqueeze(0)
    weighted_dist_neg_pol = dist_neg_pol * neg_targets_w.unsqueeze(1)
    wd_all_pol = torch.cat([weighted_dist_pos, weighted_dist_neg_pol], dim=2)
    scale_polluted = wd_all_pol.mean() / all_targets_w.mean().clamp(min=1e-8)

    # Select scale according to self_mask_on_raw + optional override.
    if scale_override == "clean":
        scale = scale_clean
    elif scale_override == "polluted":
        scale = scale_polluted
    elif self_mask_on_raw:
        scale = scale_polluted
    else:
        scale = scale_clean
    scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)

    dist_pos_n = dist_pos / scale.clamp(min=1e-3)
    if self_mask_on_raw:
        # Mask baked into raw dist before normalization: dist_neg_n = (raw + 100) / scale
        dist_neg_n = dist_neg_pol / scale.clamp(min=1e-3)
    else:
        # Mask added on top of normalized dist: dist_neg_n = raw/scale + 100
        dist_neg_n = dist_neg_raw / scale.clamp(min=1e-3)
        dist_neg_n = dist_neg_n + neg_diag_mask.unsqueeze(0)

    old_gen_sc    = old_gen   / scale_inputs
    fixed_pos_sc  = fixed_pos / scale_inputs
    neg_targets_sc = neg_targets / scale_inputs
    all_targets_sc = torch.cat([fixed_pos_sc, neg_targets_sc], dim=1)

    info: Dict[str, float] = {"scale": float(scale.item())}
    force_across_R = torch.zeros_like(old_gen_sc)

    for R in R_list:
        logit_pos = -dist_pos_n / R
        logit_neg = -dist_neg_n / R

        A_pos_raw = F.softmax(logit_pos, dim=1)
        A_neg_raw = F.softmax(logit_neg, dim=1)
        A_pos = A_pos_raw * eff_weight_pos.unsqueeze(1)
        A_neg = A_neg_raw * neg_targets_w.unsqueeze(1)

        if coupling:
            if decouple_weight_from_coupling:
                m_neg = A_neg_raw.sum(dim=2, keepdim=True)
                m_pos = A_pos_raw.sum(dim=2, keepdim=True)
            else:
                m_neg = A_neg.sum(dim=2, keepdim=True)
                m_pos = A_pos.sum(dim=2, keepdim=True)
            W_pos = A_pos * m_neg
            W_neg = A_neg * m_pos
        else:
            W_pos = A_pos
            W_neg = A_neg

        R_coeff = torch.cat([W_pos, -W_neg], dim=2)
        total_force = torch.bmm(R_coeff, all_targets_sc)
        total_coeffs = R_coeff.sum(dim=2)
        total_force = total_force - total_coeffs.unsqueeze(-1) * old_gen_sc

        if per_sample_fnorm:
            f_norm_val = (total_force ** 2).mean(dim=(-1, -2), keepdim=True)
            info[f"loss_{R}"] = float(f_norm_val.mean().item())
            force_scale = f_norm_val.sqrt().clamp(min=1e-8)
            force_across_R = force_across_R + total_force / force_scale
        else:
            f_norm_val = (total_force ** 2).mean()
            info[f"loss_{R}"] = float(f_norm_val.item())
            force_scale = f_norm_val.sqrt().clamp(min=1e-8)
            force_across_R = force_across_R + total_force / force_scale

    goal_scaled = (old_gen_sc + force_across_R).detach()
    gen_scaled = gen / scale_inputs
    loss = ((gen_scaled - goal_scaled) ** 2).mean(dim=(-1, -2))
    return loss, info


def colwise_with_self_mask(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    coupling: bool = True,
    self_mask_on_raw: bool = False,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """drift_loss_imagenet_colwise with a toggle for self-mask position.

    self_mask_on_raw=True  → OLD (commit 4f87011): +100 on raw dist BEFORE scale.
    self_mask_on_raw=False → NEW (current):         +100 on normalized dist AFTER scale.

    Also returns the intermediate tensors needed for diagnostics (A_neg, scale,
    logits, etc.) for the smallest R value only.
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]
    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = gen.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = gen.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    neg_targets = torch.cat([old_gen, fixed_neg], dim=1)

    dist_pos = _cdist_batched(old_gen, fixed_pos)
    dist_neg = _cdist_batched(old_gen, neg_targets)

    # Self-mask at raw distance level (OLD style, commit 4f87011)
    diag = torch.eye(C_g, dtype=torch.float32, device=gen.device)
    neg_diag_mask = F.pad(diag, (0, C_n)) * 100.0  # [C_g, C_g+C_n]

    if self_mask_on_raw:
        # OLD: add +100 to raw dist before any scale / normalization
        dist_neg_for_scale = dist_neg + neg_diag_mask.unsqueeze(0)
    else:
        dist_neg_for_scale = dist_neg

    # Scale computation (weighted)
    neg_targets_w = torch.cat([weight_gen, weight_neg], dim=1)
    eff_weight_pos = weight_pos
    all_targets_w = torch.cat([eff_weight_pos, neg_targets_w], dim=1)
    weighted_dist_pos = dist_pos * eff_weight_pos.unsqueeze(1)
    weighted_dist_neg = dist_neg_for_scale * neg_targets_w.unsqueeze(1)
    weighted_dist_all = torch.cat([weighted_dist_pos, weighted_dist_neg], dim=2)
    scale = weighted_dist_all.mean() / all_targets_w.mean().clamp(min=1e-8)
    scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)

    dist_pos_n = dist_pos / scale.clamp(min=1e-3)
    if self_mask_on_raw:
        # OLD: dist_neg already has +100 baked in, so normalized version carries it too
        dist_neg_n = dist_neg_for_scale / scale.clamp(min=1e-3)
    else:
        # NEW: compute normalized dist from RAW, then add +100 at normalized level
        dist_neg_n = dist_neg / scale.clamp(min=1e-3)
        dist_neg_n = dist_neg_n + neg_diag_mask.unsqueeze(0)

    old_gen_sc = old_gen / scale_inputs
    fixed_pos_sc = fixed_pos / scale_inputs
    neg_targets_sc = neg_targets / scale_inputs
    all_targets_sc = torch.cat([fixed_pos_sc, neg_targets_sc], dim=1)

    info: Dict[str, Any] = {"scale": float(scale.item())}
    force_across_R = torch.zeros_like(old_gen_sc)

    diag_logits_per_R: Dict[float, Dict[str, float]] = {}
    aneg_diag_per_R: Dict[float, Dict[str, float]] = {}
    loss_per_R: Dict[float, float] = {}

    for R in R_list:
        logit_pos = -dist_pos_n / R
        logit_neg = -dist_neg_n / R

        A_pos_raw = F.softmax(logit_pos, dim=1)
        A_neg_raw = F.softmax(logit_neg, dim=1)
        A_pos = A_pos_raw * eff_weight_pos.unsqueeze(1)
        A_neg = A_neg_raw * neg_targets_w.unsqueeze(1)

        if coupling:
            m_neg = A_neg.sum(dim=2, keepdim=True)
            m_pos = A_pos.sum(dim=2, keepdim=True)
            W_pos = A_pos * m_neg
            W_neg = A_neg * m_pos
        else:
            W_pos = A_pos
            W_neg = A_neg

        R_coeff = torch.cat([W_pos, -W_neg], dim=2)
        total_force = torch.bmm(R_coeff, all_targets_sc)
        total_coeffs = R_coeff.sum(dim=2)
        total_force = total_force - total_coeffs.unsqueeze(-1) * old_gen_sc

        f_norm_val = (total_force ** 2).mean()
        loss_per_R[float(R)] = float(f_norm_val.item())
        force_scale = f_norm_val.sqrt().clamp(min=1e-8)
        force_across_R = force_across_R + total_force / force_scale

        # --- Diagnostics at this R ---
        # Diagonal = the "self" slot: for gen sample i, column i in [old_gen | fixed_neg]
        #   so logit_neg[:, i, i] is the self-logit.
        # Off-diagonal averages (excluding the C_g×C_g self block).
        with torch.no_grad():
            diag_idx = torch.arange(C_g, device=gen.device)
            # Self logits: logit_neg[:, i, i]
            self_logit = logit_neg[:, diag_idx, diag_idx]          # [B, C_g]
            # Off-self-block logits (columns in fixed_neg part, any row)
            off_logit_neg_part = logit_neg[:, :, C_g:]              # [B, C_g, C_n]
            # Other-gen (non-self) off-diag in the gen block
            gen_block = logit_neg[:, :, :C_g]                       # [B, C_g, C_g]
            eye_mask = torch.eye(C_g, dtype=torch.bool, device=gen.device)
            off_logit_gen = gen_block.masked_fill(eye_mask.unsqueeze(0), float("nan"))

            self_aneg = A_neg_raw[:, diag_idx, diag_idx]            # [B, C_g]
            off_aneg_gen = A_neg_raw[:, :, :C_g].masked_fill(
                eye_mask.unsqueeze(0), float("nan")
            )
            off_aneg_neg = A_neg_raw[:, :, C_g:]

            diag_logits_per_R[float(R)] = {
                "self_logit_mean": float(self_logit.mean().item()),
                "self_logit_min": float(self_logit.min().item()),
                "self_logit_max": float(self_logit.max().item()),
                "offdiag_gen_logit_mean": float(torch.nanmean(off_logit_gen).item()),
                "offneg_logit_mean": float(off_logit_neg_part.mean().item()),
            }
            aneg_diag_per_R[float(R)] = {
                "A_neg_raw[self] mean": float(self_aneg.mean().item()),
                "A_neg_raw[self] max": float(self_aneg.max().item()),
                "A_neg_raw[off-gen] mean": float(torch.nanmean(off_aneg_gen).item()),
                "A_neg_raw[off-neg] mean": float(off_aneg_neg.mean().item()),
                "A_neg[self]*m_pos (repulsion on self, version_b)": float(
                    (A_neg[:, diag_idx, diag_idx] * m_pos.squeeze(-1)).mean().item()
                ) if coupling else float("nan"),
            }

    goal_scaled = (old_gen_sc + force_across_R).detach()
    gen_scaled = gen / scale_inputs
    loss = ((gen_scaled - goal_scaled) ** 2).mean(dim=(-1, -2))

    info["loss_per_R"] = loss_per_R
    info["diag_logits_per_R"] = diag_logits_per_R
    info["aneg_per_R"] = aneg_diag_per_R
    info["loss"] = float(loss.mean().item())
    # Goal direction the optimizer effectively chases: force_across_R (summed over R),
    # in feature space. Returning it lets us compare OLD vs NEW gradient direction.
    info["goal_scaled"] = goal_scaled.detach()
    info["gen_scaled"] = gen_scaled.detach()
    info["force_across_R"] = force_across_R.detach()
    return loss, info


# ---------------------------------------------------------------------------
#  Bank fill + one train_step (same as profile script but with monkey-patch)
# ---------------------------------------------------------------------------

def _fill_banks_and_batch(cfg: dict, device: torch.device):
    imagenet_path = cfg.get("imagenet_path") or ""
    cache_path = cfg.get("cache_path") or ""
    resolution = int(cfg.get("resolution", 256))
    batch_size = int(cfg.get("batch_size", 24))
    use_latent = bool(cfg.get("use_latent", True))
    use_cache = bool(cfg.get("use_cache", True))
    use_aug = bool(cfg.get("use_aug", False))

    train_loader, preprocess_fn, _ = create_imagenet_split(
        imagenet_path=imagenet_path,
        resolution=resolution,
        batch_size=batch_size,
        split="train",
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        cache_path=cache_path,
        num_workers=min(4, int(cfg.get("num_workers", 8))),
        pin_memory=bool(cfg.get("pin_memory", True)),
        distributed=False,
        rank=0,
        world_size=1,
        latent_device=device,
    )

    positive_bank_size = int(cfg.get("positive_bank_size", 128))
    negative_bank_size = int(cfg.get("negative_bank_size", 1000))
    num_classes = int(cfg.get("num_classes", 1000))
    pos_bank = ArrayMemoryBank(num_classes=num_classes, max_size=positive_bank_size)
    neg_bank = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    pos_per_sample = int(cfg.get("pos_per_sample", 16))
    neg_per_sample = int(cfg.get("neg_per_sample", 16))
    push_goal = max(batch_size * 6, pos_per_sample * 6, 128)

    it = iter(train_loader)
    n_pushed = 0
    last_labels = None
    while n_pushed < push_goal:
        batch = next(it)
        processed = preprocess_fn(batch)
        images = processed["images"]
        labels_b = processed["labels"]
        if isinstance(images, torch.Tensor):
            images_np = images.cpu().numpy()
        else:
            images_np = np.array(images)
        if isinstance(labels_b, torch.Tensor):
            labels_np = labels_b.cpu().numpy()
        else:
            labels_np = np.array(labels_b)
        pos_bank.add(images_np, labels_np)
        neg_bank.add(images_np, np.zeros_like(labels_np))
        n_pushed += images_np.shape[0]
        last_labels = labels_np

    labels_np = last_labels[:batch_size]
    rng_idx = np.random.choice(len(labels_np), size=min(batch_size, len(labels_np)), replace=False)
    labels_sel = labels_np[rng_idx]
    labels_t = torch.from_numpy(labels_sel).long().to(device)
    pos_smp = pos_bank.sample(labels_sel, n_samples=pos_per_sample, device=device)
    neg_smp = neg_bank.sample(np.zeros(len(labels_sel), dtype=np.int64), n_samples=neg_per_sample, device=device)
    return labels_t, pos_smp, neg_smp


def _fmt(v, fmt="{:+.4e}"):
    if isinstance(v, float):
        return fmt.format(v)
    return str(v)


def _classify_feature(name: str) -> str:
    """Group feature names into coarse buckets for aggregate reporting."""
    if name == "norm_x":
        return "norm_x"
    # _mean / _std without patch-size suffix → global pool (T=1)
    if name.endswith("_mean") or name.endswith("_std"):
        return "global_pool"
    # _mean_2/_mean_4/_std_2/_std_4 → patch-level pool (T=H*W/s²)
    if any(name.endswith(f"_{kind}_{s}") for kind in ("mean", "std") for s in (2, 4)):
        return "patch_pool"
    # Everything else (raw layer tokens, conv1, blk)
    return "patch_level"


# Module-level slot for the fresh-random-init state dict (populated in main()).
# Used when a ckpt path is the sentinel "step0" / "<init>" to restore the
# pre-training random weights across multiple grad-check passes.
_INIT_STATE_DICT: Optional[Dict[str, torch.Tensor]] = None


def _is_init_sentinel(path: str) -> bool:
    return path.strip().lower() in ("step0", "<init>", "random", "random_init", "init")


def _load_gen_state(gen: nn.Module, ckpt_path: str, device: torch.device) -> Tuple[int, int]:
    """Load a checkpoint's weights into gen (in place). Returns (missing, unexpected) counts.
    Sentinel paths like 'step0' restore the module-level init state dict instead."""
    if _is_init_sentinel(ckpt_path):
        if _INIT_STATE_DICT is None:
            raise SystemExit("step0 sentinel used but _INIT_STATE_DICT was never captured")
        gen.load_state_dict(_INIT_STATE_DICT)
        return 0, 0
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = None
    for key in ("ema", "model", "generator", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            sd = ckpt[key]
            break
    if sd is None and isinstance(ckpt, dict) and "params" in ckpt:
        sd = ckpt["params"]
    if sd is None:
        raise SystemExit(f"Could not find state dict in checkpoint {ckpt_path}")
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    return len(missing), len(unexpected)


def _analyze_captured(
    captured: List[Dict[str, Any]],
    names: List[str],
) -> Dict[str, Any]:
    """Run OLD/NEW self-mask on every captured call and aggregate stats."""
    total_old = 0.0
    total_new = 0.0
    rows: List[Dict[str, Any]] = []
    for name, cap in zip(names, captured):
        with torch.no_grad():
            _, info_new = colwise_with_self_mask(
                gen=cap["gen"], fixed_pos=cap["fixed_pos"], fixed_neg=cap["fixed_neg"],
                weight_gen=cap["weight_gen"], weight_pos=cap["weight_pos"], weight_neg=cap["weight_neg"],
                R_list=cap["R_list"], coupling=cap["coupling"], self_mask_on_raw=False,
            )
            _, info_old = colwise_with_self_mask(
                gen=cap["gen"], fixed_pos=cap["fixed_pos"], fixed_neg=cap["fixed_neg"],
                weight_gen=cap["weight_gen"], weight_pos=cap["weight_pos"], weight_neg=cap["weight_neg"],
                R_list=cap["R_list"], coupling=cap["coupling"], self_mask_on_raw=True,
            )
        total_old += info_old["loss"]
        total_new += info_new["loss"]

        # Cosine similarity on the force vector (per-sample, then mean over B*T)
        f_o = info_old["force_across_R"].reshape(info_old["force_across_R"].shape[0], -1)
        f_n = info_new["force_across_R"].reshape(info_new["force_across_R"].shape[0], -1)
        cos_force = F.cosine_similarity(f_o, f_n, dim=-1).mean().item()

        rows.append({
            "name": name,
            "shape": tuple(cap["gen"].shape),
            "scale_old": info_old["scale"],
            "scale_new": info_new["scale"],
            "loss_old": info_old["loss"],
            "loss_new": info_new["loss"],
            "cos_force": cos_force,
            "group": _classify_feature(name),
        })

    # Aggregate by group
    groups: Dict[str, Dict[str, List[float]]] = {}
    for r in rows:
        g = groups.setdefault(r["group"], {"cos": [], "loss_o": [], "loss_n": []})
        g["cos"].append(r["cos_force"])
        g["loss_o"].append(r["loss_old"])
        g["loss_n"].append(r["loss_new"])

    group_stats: Dict[str, Dict[str, float]] = {}
    for g, d in groups.items():
        group_stats[g] = {
            "count": len(d["cos"]),
            "cos_mean": float(np.mean(d["cos"])),
            "cos_min": float(np.min(d["cos"])),
            "loss_old_sum": float(np.sum(d["loss_o"])),
            "loss_new_sum": float(np.sum(d["loss_n"])),
        }

    return {
        "rows": rows,
        "total_old": total_old,
        "total_new": total_new,
        "cos_mean": float(np.mean([r["cos_force"] for r in rows])),
        "cos_min": float(np.min([r["cos_force"] for r in rows])),
        "cos_p5": float(np.percentile([r["cos_force"] for r in rows], 5)),
        "groups": group_stats,
    }


def _grad_direction_check(
    gen: nn.Module,
    mae: Any,
    ckpt_path: str,
    device: torch.device,
    labels_t: torch.Tensor,
    pos_smp: torch.Tensor,
    neg_smp: torch.Tensor,
    cfg: dict,
    seed: int,
    run_ablations: bool = True,
) -> Dict[str, Any]:
    """Run forward+backward end-to-end for multiple self-mask variants and compare
    the resulting parameter-gradient directions.

    Variants:
      OLD        = self_mask_on_raw=True                              (4f87011 original)
      NEW        = self_mask_on_raw=False                             (current)
      OLD+clean  = self_mask_on_raw=True,  scale_override="clean"     (OLD mask path, NEW's scale)
      NEW+pol    = self_mask_on_raw=False, scale_override="polluted"  (NEW mask path, OLD's scale)

    If the OLD vs NEW gap is purely due to scale pollution, then:
      cos(OLD, NEW+pol)  → ~1.0
      cos(NEW, OLD+clean) → ~1.0
    """
    global SELF_MASK_ON_RAW, PATCH_SCALE_OVERRIDE
    _load_gen_state(gen, ckpt_path, device)

    opt = build_optimizer(gen, cfg)
    orig_step = opt.step
    orig_zero = opt.zero_grad
    opt.step = lambda *a, **k: None
    opt.zero_grad = lambda *a, **k: None

    il.drift_loss_imagenet_colwise = _patched_colwise
    tig.drift_loss_imagenet_colwise = _patched_colwise

    def _run_once(mask_mode: bool, scale_ov: Optional[str]) -> Tuple[torch.Tensor, float]:
        global SELF_MASK_ON_RAW, PATCH_SCALE_OVERRIDE
        SELF_MASK_ON_RAW = mask_mode
        PATCH_SCALE_OVERRIDE = scale_ov
        for p in gen.parameters():
            if p.grad is not None:
                p.grad.zero_()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        loss, _, _ = train_step(gen, mae, opt, labels_t, pos_smp, neg_smp, device, 0, cfg)
        grads = []
        for p in gen.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().clone().reshape(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads), float(loss.detach().mean().item())

    try:
        g_old, loss_old = _run_once(True,  None)
        g_new, loss_new = _run_once(False, None)
        if run_ablations:
            g_old_clean, _ = _run_once(True,  "clean")      # OLD mask + NEW scale
            g_new_pol,   _ = _run_once(False, "polluted")   # NEW mask + OLD scale
        else:
            g_old_clean = g_new_pol = None
    finally:
        opt.step = orig_step
        opt.zero_grad = orig_zero
        SELF_MASK_ON_RAW = None
        PATCH_SCALE_OVERRIDE = None
        il.drift_loss_imagenet_colwise = _original_colwise
        tig.drift_loss_imagenet_colwise = _original_colwise

    def _cos(a, b):
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()

    out = {
        "cos_param_grad": _cos(g_old, g_new),            # main OLD vs NEW
        "grad_norm_old": float(g_old.norm().item()),
        "grad_norm_new": float(g_new.norm().item()),
        "loss_old_sum": loss_old,
        "loss_new_sum": loss_new,
    }
    if run_ablations:
        # If scale-pollution is the ONLY cause of OLD≠NEW, these two should be ~1.
        out["cos_NEW_vs_OLDcleanScale"] = _cos(g_new, g_old_clean)       # should be ~1 if hypothesis holds
        out["cos_OLD_vs_NEWpollutedScale"] = _cos(g_old, g_new_pol)      # should be ~1 if hypothesis holds
        # Sanity: OLD should agree with OLD+clean path only via scale change, etc.
        out["cos_OLDclean_vs_NEWpol"] = _cos(g_old_clean, g_new_pol)
        out["grad_norm_old_clean"] = float(g_old_clean.norm().item())
        out["grad_norm_new_pol"] = float(g_new_pol.norm().item())
    return out


def _run_one_checkpoint(
    gen: nn.Module,
    mae: Any,
    ckpt_path: str,
    device: torch.device,
    labels_t: torch.Tensor,
    pos_smp: torch.Tensor,
    neg_smp: torch.Tensor,
    cfg: dict,
) -> Dict[str, Any]:
    """Load checkpoint → run 1 train_step (capturing loss calls) → analyze."""
    missing, unexpected = _load_gen_state(gen, ckpt_path, device)

    # Reset captures
    CAPTURED_LIST.clear()
    capture_names: List[str] = []

    # Monkey-patch
    il.drift_loss_imagenet_colwise = _patched_colwise
    tig.drift_loss_imagenet_colwise = _patched_colwise
    original_compute = tig.compute_drift_loss_from_features

    def _compute_wrapped(*args, **kwargs):
        gen_feats = kwargs.get("gen_feats", args[0] if args else None)
        if gen_feats is not None:
            capture_names.extend(list(gen_feats.keys()))
        return original_compute(*args, **kwargs)

    tig.compute_drift_loss_from_features = _compute_wrapped

    optimizer = build_optimizer(gen, cfg)
    try:
        _ = train_step(gen, mae, optimizer, labels_t, pos_smp, neg_smp, device, 0, cfg)
    finally:
        il.drift_loss_imagenet_colwise = _original_colwise
        tig.drift_loss_imagenet_colwise = _original_colwise
        tig.compute_drift_loss_from_features = original_compute

    # Align names with captures
    aligned_names: List[str] = []
    idx = 0
    for name in capture_names:
        if idx < len(CAPTURED_LIST):
            aligned_names.append(name)
            idx += 1
    while len(aligned_names) < len(CAPTURED_LIST):
        aligned_names.append(f"layer_{len(aligned_names)}")

    result = _analyze_captured(list(CAPTURED_LIST), aligned_names)
    result["ckpt"] = ckpt_path
    result["missing_keys"] = missing
    result["unexpected_keys"] = unexpected

    # Free the big captured tensors before moving on
    CAPTURED_LIST.clear()
    torch.cuda.empty_cache()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="configs/gen/latent_small_version_b_nocfgw_gpu4567_posneg16_gen16.yaml")
    p.add_argument("--checkpoints", type=str,
                   default=",".join([
                       "/data/runs/gen_small_version_b_nocfgw_gpu4567_posneg16_gen16/checkpoints/ckpt_step_0010000.pt",
                       "/data/runs/gen_small_version_b_nocfgw_gpu4567_posneg16_gen16/checkpoints/ckpt_step_0020000.pt",
                       "/data/runs/gen_small_version_b_nocfgw_gpu4567_posneg16_gen16/checkpoints/ckpt_step_0024000.pt",
                       "/data/runs/gen_small_version_b_nocfgw_gpu4567_posneg16_gen16/checkpoints/ckpt_step_0026000.pt",
                   ]),
                   help="Comma-separated checkpoint paths to sweep.")
    p.add_argument("--mae_checkpoint", type=str,
                   default="/data/imagenet/mae_latent_256/ckpt_latest.pt")
    p.add_argument("--gpu", type=str, default="4")
    p.add_argument("--batch_size", type=int, default=24, help="Training batch size (same as real run)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per_layer", action="store_true",
                   help="Also dump the full per-layer table for the LAST checkpoint.")
    args = p.parse_args()

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(ROOT, config_path)
    cfg = load_yaml_config(config_path)
    if args.mae_checkpoint:
        cfg["mae_checkpoint"] = args.mae_checkpoint
    cfg.setdefault("imagenet_path", os.environ.get("IMAGENET_PATH", "")
                   or "/data/imagenet/ILSVRC2012")
    cfg.setdefault("cache_path", os.environ.get("IMAGENET_CACHE_PATH", "")
                   or "/data/imagenet/latent_cache_sd_vae_mse")
    cfg["batch_size"] = args.batch_size
    cfg["use_wandb"] = False
    cfg["memory_efficient_bidir"] = False
    cfg["compute_wpos_stats"] = False

    # --- Build generator once (state dict swapped per checkpoint) ---
    raw = cfg.get("_raw", {})
    gen = build_ditgen_from_config(raw.get("model", cfg), raw.get("dataset", cfg)).to(device)
    mae = load_mae(cfg.get("mae_checkpoint", ""), cfg, device)

    # Snapshot fresh-random-init state so "step0" sentinel paths can restore it.
    global _INIT_STATE_DICT
    _INIT_STATE_DICT = {k: v.detach().clone() for k, v in gen.state_dict().items()}

    # --- Fill memory banks once, reuse same (labels, pos, neg) across checkpoints ---
    print(f"[debug] filling memory banks (batch_size={args.batch_size}) ...")
    labels_t, pos_smp, neg_smp = _fill_banks_and_batch(cfg, device)
    print(f"[debug] labels shape {labels_t.shape}, pos {pos_smp.shape}, neg {neg_smp.shape}")

    cfg_w = copy.deepcopy(cfg)
    cfg_w["drift_matching"] = "version_b"
    cfg_w["profile_train_step"] = False
    cfg_w["use_wandb"] = False
    cfg_w["compute_wpos_stats"] = False

    # --- Sweep checkpoints ---
    ckpt_paths = [s.strip() for s in args.checkpoints.split(",") if s.strip()]
    results: List[Dict[str, Any]] = []
    for ckpt_path in ckpt_paths:
        if not _is_init_sentinel(ckpt_path) and not os.path.exists(ckpt_path):
            print(f"[debug] SKIP missing checkpoint: {ckpt_path}")
            continue
        label = os.path.basename(ckpt_path)
        print(f"\n[debug] === {label} ===")
        # 1. Feature-space cos (per-layer force vectors; cheap capture-mode pass)
        r = _run_one_checkpoint(gen, mae, ckpt_path, device, labels_t, pos_smp, neg_smp, cfg_w)
        print(f"[debug]   features: {len(r['rows'])}  cos(force) mean={r['cos_mean']:.4f}  "
              f"min={r['cos_min']:.4f}  p5={r['cos_p5']:.4f}")
        print(f"[debug]   total loss OLD={r['total_old']:.3e}  NEW={r['total_new']:.3e}  "
              f"ratio={r['total_old']/max(r['total_new'],1e-12):.4f}x")

        # 2. End-to-end model-parameter gradient cos (forward+backward twice)
        print(f"[debug]   running grad-direction check (2× forward+backward) ...")
        g = _grad_direction_check(gen, mae, ckpt_path, device, labels_t, pos_smp, neg_smp, cfg_w, args.seed)
        print(f"[debug]   cos(param_grad) = {g['cos_param_grad']:.6f}   "
              f"||g_old||={g['grad_norm_old']:.3e}   ||g_new||={g['grad_norm_new']:.3e}   "
              f"(loss_sum OLD={g['loss_old_sum']:.3e}  NEW={g['loss_new_sum']:.3e})")
        r["cos_param_grad"] = g["cos_param_grad"]
        r["grad_norm_old"] = g["grad_norm_old"]
        r["grad_norm_new"] = g["grad_norm_new"]
        # Optional ablation outputs (present only when _grad_direction_check ran
        # with run_ablations=True). Currently unused by the trajectory table.
        for k in ("cos_NEW_vs_OLDcleanScale", "cos_OLD_vs_NEWpollutedScale",
                  "cos_OLDclean_vs_NEWpol", "grad_norm_old_clean", "grad_norm_new_pol"):
            if k in g:
                r[k] = g[k]
        results.append(r)

    # --- Trajectory table across checkpoints ---
    print()
    print("=" * 118)
    print(f"  SELF-MASK TRAJECTORY  (batch_size={args.batch_size}, same batch reused across checkpoints, seed={args.seed})")
    print("  OLD = commit 4f87011 (raw + 100 BEFORE scale)  /  NEW = current (normalized + 100 AFTER scale)")
    print("=" * 118)
    print(f"  {'checkpoint':<24} | {'cos(force)':>10} {'cos(param_grad)':>17} | "
          f"{'||g_old||':>10} {'||g_new||':>10} | {'loss OLD':>11} {'loss NEW':>11} {'OLD/NEW':>8}")
    print("  " + "-" * 116)
    for r in results:
        pg_cos = r.get("cos_param_grad", float("nan"))
        gno = r.get("grad_norm_old", float("nan"))
        gnn = r.get("grad_norm_new", float("nan"))
        print(f"  {os.path.basename(r['ckpt']):<24} | "
              f"{r['cos_mean']:>10.4f} {pg_cos:>17.6f} | "
              f"{gno:>10.3e} {gnn:>10.3e} | "
              f"{r['total_old']:>11.3e} {r['total_new']:>11.3e} "
              f"{r['total_old']/max(r['total_new'],1e-12):>8.3f}")
    print("  " + "-" * 116)

    # --- Per-group breakdown (for each ckpt, show cos_mean per feature group) ---
    if results:
        groups_seen = sorted({g for r in results for g in r["groups"].keys()})
        print()
        print("  Feature-group breakdown (cos(force) mean):")
        header = "  {:<24} | ".format("checkpoint")
        header += " ".join(f"{g:>14}" for g in groups_seen)
        print(header)
        print("  " + "-" * (26 + 15 * len(groups_seen)))
        for r in results:
            row = "  {:<24} | ".format(os.path.basename(r["ckpt"]))
            for g in groups_seen:
                if g in r["groups"]:
                    gs = r["groups"][g]
                    row += f"{gs['cos_mean']:>10.4f}(n={gs['count']:<2})"
                else:
                    row += f"{'-':>14}"
            print(row)

    # --- Optional full per-layer table for the LAST checkpoint ---
    if args.per_layer and results:
        r = results[-1]
        print()
        print(f"  Full per-layer breakdown for {os.path.basename(r['ckpt'])}:")
        print(f"  {'feature':<22} {'D':>4} {'group':>12} | {'scale OLD':>10} {'scale NEW':>10} | "
              f"{'loss OLD':>11} {'loss NEW':>11} {'OLD/NEW':>8} | {'cos(force)':>10}")
        print("  " + "-" * 114)
        for row in r["rows"]:
            D = row["shape"][2]
            ratio = row["loss_old"] / row["loss_new"] if row["loss_new"] > 0 else float("nan")
            print(f"  {row['name']:<22} {D:>4} {row['group']:>12} | "
                  f"{row['scale_old']:>10.4f} {row['scale_new']:>10.4f} | "
                  f"{row['loss_old']:>11.3e} {row['loss_new']:>11.3e} {ratio:>8.3f} | "
                  f"{row['cos_force']:>10.4f}")

    print()
    print("  cos(force) is per-sample cosine similarity between OLD and NEW 'force_across_R'")
    print("  vectors — directly proxies gradient-direction similarity seen by the optimizer.")
    print("  cos → 1 means OLD and NEW are equivalent; lower means meaningfully different.")
    print()


if __name__ == "__main__":
    main()
