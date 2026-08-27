"""Indexed weighted reductions used by sparse drift-force construction.

The CUDA path avoids materializing ``[batch, generated, top_k, feature]``
gathers.  That tensor can be several GiB for early, high-resolution MAE
features.  A small PyTorch reference path is retained for CPU correctness
tests.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Optional

import torch

try:  # Triton is optional for CPU-only tooling and unit tests.
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _indexed_weighted_sum_kernel(
        weights_ptr,
        indices_ptr,
        target0_ptr,
        target1_ptr,
        output_ptr,
        n_gen: tl.constexpr,
        n_target0,
        n_features,
        stride_wb,
        stride_wg,
        stride_wk,
        stride_ib,
        stride_ig,
        stride_ik,
        stride_t0b,
        stride_t0c,
        stride_t0s,
        stride_t1b,
        stride_t1c,
        stride_t1s,
        stride_ob,
        stride_og,
        stride_os,
        alpha,
        beta,
        TOP_K: tl.constexpr,
        HAS_TARGET1: tl.constexpr,
        LOAD_OUTPUT: tl.constexpr,
        BLOCK_G: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        gen_offsets = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
        feature_offsets = tl.program_id(2) * BLOCK_S + tl.arange(0, BLOCK_S)
        gen_mask = gen_offsets < n_gen
        feature_mask = feature_offsets < n_features

        accumulator = tl.zeros((BLOCK_G, BLOCK_S), dtype=tl.float32)
        for k_start in range(0, TOP_K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            gk_mask = gen_mask[:, None] & (k_offsets[None, :] < TOP_K)
            weight_ptrs = (
                weights_ptr
                + batch_idx * stride_wb
                + gen_offsets[:, None] * stride_wg
                + k_offsets[None, :] * stride_wk
            )
            index_ptrs = (
                indices_ptr
                + batch_idx * stride_ib
                + gen_offsets[:, None] * stride_ig
                + k_offsets[None, :] * stride_ik
            )
            weights = tl.load(weight_ptrs, mask=gk_mask, other=0.0).to(tl.float32)
            indices = tl.load(index_ptrs, mask=gk_mask, other=0).to(tl.int64)

            target0_mask = gk_mask & (indices < n_target0)
            target0_ptrs = (
                target0_ptr
                + batch_idx * stride_t0b
                + indices[:, :, None] * stride_t0c
                + feature_offsets[None, None, :] * stride_t0s
            )
            values0 = tl.load(
                target0_ptrs,
                mask=target0_mask[:, :, None] & feature_mask[None, None, :],
                other=0.0,
            ).to(tl.float32)

            if HAS_TARGET1:
                indices1 = indices - n_target0
                target1_mask = gk_mask & (indices >= n_target0)
                target1_ptrs = (
                    target1_ptr
                    + batch_idx * stride_t1b
                    + indices1[:, :, None] * stride_t1c
                    + feature_offsets[None, None, :] * stride_t1s
                )
                values1 = tl.load(
                    target1_ptrs,
                    mask=target1_mask[:, :, None] & feature_mask[None, None, :],
                    other=0.0,
                ).to(tl.float32)
                values0 += values1

            accumulator += tl.sum(weights[:, :, None] * values0, axis=1)

        output_ptrs = (
            output_ptr
            + batch_idx * stride_ob
            + gen_offsets[:, None] * stride_og
            + feature_offsets[None, :] * stride_os
        )
        output_mask = gen_mask[:, None] & feature_mask[None, :]
        if LOAD_OUTPUT:
            previous = tl.load(output_ptrs, mask=output_mask, other=0.0).to(tl.float32)
            accumulator = accumulator * alpha + previous * beta
        else:
            accumulator *= alpha
        tl.store(output_ptrs, accumulator, mask=output_mask)


def _torch_indexed_weighted_sum(
    weights: torch.Tensor,
    indices: torch.Tensor,
    target0: torch.Tensor,
    target1: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Small reference implementation; unsuitable for full ImageNet CUDA tensors."""
    targets = target0 if target1 is None else torch.cat([target0, target1], dim=1)
    batch, n_gen, top_k = indices.shape
    n_features = targets.shape[2]
    expanded_targets = targets.unsqueeze(1).expand(-1, n_gen, -1, -1)
    selected = torch.gather(
        expanded_targets,
        dim=2,
        index=indices.unsqueeze(-1).expand(batch, n_gen, top_k, n_features),
    )
    result = (weights.unsqueeze(-1) * selected).sum(dim=2)
    result.mul_(float(alpha))
    if out is not None:
        result.add_(out, alpha=float(beta))
    return result


def indexed_weighted_sum(
    weights: torch.Tensor,
    indices: torch.Tensor,
    target0: torch.Tensor,
    target1: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute a row-wise indexed weighted target sum.

    Args:
        weights: ``[B, G, K]`` compact weights.
        indices: ``[B, G, K]`` target indices.  When ``target1`` is supplied,
            indices address ``cat([target0, target1], dim=1)`` without actually
            materializing that concatenation.
        target0: ``[B, C0, S]`` target features.
        target1: optional ``[B, C1, S]`` continuation of the target pool.
        out: optional ``[B, G, S]`` accumulation tensor.
        alpha, beta: return ``alpha * indexed_sum + beta * out``.
    """
    if weights.ndim != 3 or indices.shape != weights.shape:
        raise ValueError("weights and indices must have identical [B, G, K] shapes")
    if target0.ndim != 3 or target0.shape[0] != weights.shape[0]:
        raise ValueError("target0 must have shape [B, C, S] with matching B")
    if target1 is not None and (
        target1.ndim != 3
        or target1.shape[0] != target0.shape[0]
        or target1.shape[2] != target0.shape[2]
    ):
        raise ValueError("target1 must have shape [B, C1, S] matching target0")
    expected_out_shape = (weights.shape[0], weights.shape[1], target0.shape[2])
    if out is not None and tuple(out.shape) != expected_out_shape:
        raise ValueError(f"out has shape {tuple(out.shape)}, expected {expected_out_shape}")
    if indices.dtype != torch.int64:
        indices = indices.to(dtype=torch.int64)

    if not weights.is_cuda:
        return _torch_indexed_weighted_sum(
            weights, indices, target0, target1, out, alpha, beta
        )
    if not _TRITON_AVAILABLE:  # Avoid a catastrophic CUDA gather fallback.
        raise RuntimeError("CUDA top-k drift accumulation requires Triton")

    # Conda's compiler package intentionally installs a target-prefixed binary
    # instead of ``gcc``. Triton's launcher builder only probes gcc/clang unless
    # CC is explicit, so discover the compiler beside the active interpreter.
    if not os.environ.get("CC") and not (
        shutil.which("gcc") or shutil.which("clang")
    ):
        conda_cc = Path(sys.executable).resolve().parent / "x86_64-conda-linux-gnu-cc"
        if conda_cc.is_file():
            os.environ["CC"] = str(conda_cc)

    batch, n_gen, top_k = weights.shape
    n_features = target0.shape[2]
    if out is None:
        output = torch.empty(
            expected_out_shape,
            device=weights.device,
            dtype=torch.float32,
        )
    else:
        output = out
    if output.dtype != torch.float32:
        raise ValueError("indexed weighted-sum output must be float32")

    # A valid placeholder pointer is required by the compiled signature even
    # when HAS_TARGET1=False; masking prevents it from being read.
    target1_arg = target0 if target1 is None else target1
    block_g = 4
    block_s = 32
    block_k = 8
    grid = (
        batch,
        triton.cdiv(n_gen, block_g),
        triton.cdiv(n_features, block_s),
    )
    _indexed_weighted_sum_kernel[grid](
        weights,
        indices,
        target0,
        target1_arg,
        output,
        n_gen=n_gen,
        n_target0=target0.shape[1],
        n_features=n_features,
        stride_wb=weights.stride(0),
        stride_wg=weights.stride(1),
        stride_wk=weights.stride(2),
        stride_ib=indices.stride(0),
        stride_ig=indices.stride(1),
        stride_ik=indices.stride(2),
        stride_t0b=target0.stride(0),
        stride_t0c=target0.stride(1),
        stride_t0s=target0.stride(2),
        stride_t1b=target1_arg.stride(0),
        stride_t1c=target1_arg.stride(1),
        stride_t1s=target1_arg.stride(2),
        stride_ob=output.stride(0),
        stride_og=output.stride(1),
        stride_os=output.stride(2),
        alpha=float(alpha),
        beta=float(beta),
        TOP_K=top_k,
        HAS_TARGET1=target1 is not None,
        LOAD_OUTPUT=out is not None,
        BLOCK_G=block_g,
        BLOCK_K=block_k,
        BLOCK_S=block_s,
        num_warps=4,
    )
    return output
