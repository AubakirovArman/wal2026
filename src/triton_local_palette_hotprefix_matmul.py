from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
    ],
    key=["m_size", "n_size", "k_size", "palette_size"],
)
@triton.jit
def _local_palette_hotprefix_linear_kernel(
    x_ptr,
    local_idx_ptr,
    palette_ptr,
    row_scale_ptr,
    out_ptr,
    m_size,
    n_size,
    k_size,
    palette_size,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HOT_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    hot_offs = tl.arange(0, HOT_SIZE)
    rho = tl.load(row_scale_ptr + offs_n * stride_rn, mask=offs_n < n_size, other=1.0)
    hot_vals = tl.load(palette_ptr + hot_offs, mask=hot_offs < palette_size, other=0.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k_size, BLOCK_K):
        k_idx = k_start + offs_k
        x_mask = (offs_m[:, None] < m_size) & (k_idx[None, :] < k_size)
        w_mask = (offs_n[:, None] < n_size) & (k_idx[None, :] < k_size)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        )
        idx = tl.load(
            local_idx_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk,
            mask=w_mask,
            other=0,
        ).to(tl.int32)
        cold_mask = w_mask & (idx >= HOT_SIZE)
        weights = tl.load(palette_ptr + idx, mask=cold_mask, other=0.0).to(x.dtype)
        for hot_idx in range(HOT_SIZE):
            weights = tl.where(idx == hot_idx, tl.load(palette_ptr + hot_idx).to(x.dtype), weights)
        weights = weights * rho[:, None].to(x.dtype)
        acc += tl.dot(x, tl.trans(weights))

    out_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=out_mask,
    )


@triton.jit
def _local_palette_hotprefix_linear_kernel_fixed(
    x_ptr,
    local_idx_ptr,
    palette_ptr,
    row_scale_ptr,
    out_ptr,
    m_size,
    n_size,
    k_size,
    palette_size,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_rn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HOT_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    rho = tl.load(row_scale_ptr + offs_n * stride_rn, mask=offs_n < n_size, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, k_size, BLOCK_K):
        k_idx = k_start + offs_k
        x_mask = (offs_m[:, None] < m_size) & (k_idx[None, :] < k_size)
        w_mask = (offs_n[:, None] < n_size) & (k_idx[None, :] < k_size)
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk,
            mask=x_mask,
            other=0.0,
        )
        idx = tl.load(
            local_idx_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk,
            mask=w_mask,
            other=0,
        ).to(tl.int32)
        cold_mask = w_mask & (idx >= HOT_SIZE)
        weights = tl.load(palette_ptr + idx, mask=cold_mask, other=0.0).to(x.dtype)
        for hot_idx in range(HOT_SIZE):
            weights = tl.where(idx == hot_idx, tl.load(palette_ptr + hot_idx).to(x.dtype), weights)
        weights = weights * rho[:, None].to(x.dtype)
        acc += tl.dot(x, tl.trans(weights))

    out_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=out_mask,
    )


def local_palette_hotprefix_linear_matmul(
    x: torch.Tensor,
    local_idx: torch.Tensor,
    palette: torch.Tensor,
    row_scale: torch.Tensor,
    hot_size: int = 32,
    *,
    autotuned: bool = False,
    block_m: int = 32,
    block_n: int = 32,
    block_k: int = 32,
) -> torch.Tensor:
    if not x.is_cuda or not local_idx.is_cuda or not palette.is_cuda or not row_scale.is_cuda:
        raise ValueError("local_palette_hotprefix_linear_matmul expects CUDA tensors")
    if x.shape[-1] != local_idx.shape[1]:
        raise ValueError(f"Feature mismatch: {x.shape[-1]} != {local_idx.shape[1]}")
    if local_idx.shape[0] != row_scale.numel():
        raise ValueError(f"Row-scale mismatch: {local_idx.shape[0]} != {row_scale.numel()}")
    if hot_size < 1:
        raise ValueError("hot_size must be >= 1")

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    idx_2d = local_idx.contiguous()
    palette_1d = palette.reshape(-1).contiguous()
    row_1d = row_scale.reshape(-1).contiguous()
    m_size, k_size = x_2d.shape
    n_size = idx_2d.shape[0]
    out = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    if autotuned:
        grid = lambda meta: (triton.cdiv(m_size, meta["BLOCK_M"]), triton.cdiv(n_size, meta["BLOCK_N"]))
        _local_palette_hotprefix_linear_kernel[grid](
            x_2d,
            idx_2d,
            palette_1d,
            row_1d,
            out,
            m_size,
            n_size,
            k_size,
            palette_1d.numel(),
            x_2d.stride(0),
            x_2d.stride(1),
            idx_2d.stride(0),
            idx_2d.stride(1),
            row_1d.stride(0),
            out.stride(0),
            out.stride(1),
            HOT_SIZE=hot_size,
        )
    else:
        grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
        _local_palette_hotprefix_linear_kernel_fixed[grid](
            x_2d,
            idx_2d,
            palette_1d,
            row_1d,
            out,
            m_size,
            n_size,
            k_size,
            palette_1d.numel(),
            x_2d.stride(0),
            x_2d.stride(1),
            idx_2d.stride(0),
            idx_2d.stride(1),
            row_1d.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            HOT_SIZE=hot_size,
            num_warps=4,
            num_stages=2,
        )
    return out.reshape(*original_shape[:-1], n_size)