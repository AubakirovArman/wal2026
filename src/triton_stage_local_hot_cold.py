from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_local_hot_cold_kernel(
    x_ptr, stage_ids_ptr, full_codebooks_ptr, hot_codebooks_ptr, hot_lut_ptr, row_scale_ptr, out_ptr,
    m_size, n_size, blocks_per_row,
    stride_xm, stride_xk, stride_ids_s, stride_ids_n, stride_ids_b, stride_full_s, stride_full_c, stride_full_k,
    stride_hot_s, stride_hot_c, stride_hot_k, stride_lut_s, stride_lut_c, stride_rn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_SIZE)
    row_mask = offs_n < n_size
    rho = tl.load(row_scale_ptr + offs_n * stride_rn, mask=row_mask, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_idx in range(0, blocks_per_row):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (block_idx * BLOCK_SIZE + offs_k[None, :]) * stride_xk, mask=offs_m[:, None] < m_size, other=0.0)
        weight_block = tl.zeros((BLOCK_N, BLOCK_SIZE), dtype=tl.float32)
        for stage_idx in range(NUM_STAGES):
            ids = tl.load(stage_ids_ptr + stage_idx * stride_ids_s + offs_n * stride_ids_n + block_idx * stride_ids_b, mask=row_mask, other=0)
            hot_slots = tl.load(hot_lut_ptr + stage_idx * stride_lut_s + ids.to(tl.int32) * stride_lut_c, mask=row_mask, other=-1)
            hot_mask = row_mask & (hot_slots >= 0)
            cold_mask = row_mask & ~hot_mask
            hot_safe = tl.where(hot_mask, hot_slots, 0).to(tl.int32)
            hot_stage = tl.load(hot_codebooks_ptr + stage_idx * stride_hot_s + hot_safe[:, None] * stride_hot_c + offs_k[None, :] * stride_hot_k, mask=hot_mask[:, None], other=0.0)
            cold_stage = tl.load(full_codebooks_ptr + stage_idx * stride_full_s + ids[:, None].to(tl.int32) * stride_full_c + offs_k[None, :] * stride_full_k, mask=cold_mask[:, None], other=0.0)
            weight_block += hot_stage.to(tl.float32) + cold_stage.to(tl.float32)
        acc += tl.dot(x, tl.trans(weight_block.to(x.dtype)))
    acc = acc * rho[None, :].to(acc.dtype)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size))


@triton.jit
def _stage_local_hot_palette_kernel(
    x_ptr, stage_ids_ptr, full_codebooks_ptr, hot_ids_ptr, hot_codebooks_ptr, row_scale_ptr, out_ptr,
    m_size, n_size, blocks_per_row,
    stride_xm, stride_xk, stride_ids_s, stride_ids_n, stride_ids_b, stride_full_s, stride_full_c, stride_full_k,
    stride_hot_ids_s, stride_hot_ids_h, stride_hot_s, stride_hot_c, stride_hot_k, stride_rn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr, HOT_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_SIZE)
    hot_offs = tl.arange(0, HOT_SIZE)
    row_mask = offs_n < n_size
    rho = tl.load(row_scale_ptr + offs_n * stride_rn, mask=row_mask, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_idx in range(0, blocks_per_row):
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + (block_idx * BLOCK_SIZE + offs_k[None, :]) * stride_xk, mask=offs_m[:, None] < m_size, other=0.0)
        weight_block = tl.zeros((BLOCK_N, BLOCK_SIZE), dtype=tl.float32)
        for stage_idx in range(NUM_STAGES):
            ids = tl.load(stage_ids_ptr + stage_idx * stride_ids_s + offs_n * stride_ids_n + block_idx * stride_ids_b, mask=row_mask, other=0)
            hot_ids = tl.load(hot_ids_ptr + stage_idx * stride_hot_ids_s + hot_offs * stride_hot_ids_h, mask=hot_offs < HOT_SIZE, other=-1)
            hot_stage = tl.load(hot_codebooks_ptr + stage_idx * stride_hot_s + hot_offs[:, None] * stride_hot_c + offs_k[None, :] * stride_hot_k, mask=hot_offs[:, None] < HOT_SIZE, other=0.0).to(tl.float32)
            stage_block = tl.zeros((BLOCK_N, BLOCK_SIZE), dtype=tl.float32)
            matched = tl.zeros((BLOCK_N,), dtype=tl.int32)
            for hot_idx in range(HOT_SIZE):
                is_hot = row_mask & (tl.load(hot_ids_ptr + stage_idx * stride_hot_ids_s + hot_idx * stride_hot_ids_h) >= 0) & (ids == tl.load(hot_ids_ptr + stage_idx * stride_hot_ids_s + hot_idx * stride_hot_ids_h))
                stage_block = tl.where(is_hot[:, None], tl.load(hot_codebooks_ptr + stage_idx * stride_hot_s + hot_idx * stride_hot_c + offs_k * stride_hot_k)[None, :], stage_block)
                matched = tl.where(is_hot, 1, matched)
            cold_mask = row_mask & (matched == 0)
            cold_ids = tl.where(cold_mask, ids, 0).to(tl.int32)
            cold_stage = tl.load(full_codebooks_ptr + stage_idx * stride_full_s + cold_ids[:, None] * stride_full_c + offs_k[None, :] * stride_full_k, mask=cold_mask[:, None], other=0.0)
            weight_block += stage_block + cold_stage.to(tl.float32)
        acc += tl.dot(x, tl.trans(weight_block.to(x.dtype)))
    acc = acc * rho[None, :].to(acc.dtype)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size))


@triton.jit
def _stage_local_hot_palette_b2_kernel(
    x_ptr, stage_ids_ptr, full_codebooks_ptr, hot_codebooks_ptr, hot_lut_ptr, row_scale_ptr, out_ptr,
    m_size, n_size, blocks_per_row,
    stride_xm, stride_xk, stride_ids_s, stride_ids_n, stride_ids_b, stride_full_s, stride_full_c, stride_full_k,
    stride_hot_s, stride_hot_c, stride_hot_k, stride_lut_s, stride_lut_c, stride_rn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_SIZE: tl.constexpr, NUM_STAGES: tl.constexpr, HOT_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_SIZE)
    hot_offs = tl.arange(0, HOT_SIZE)
    row_mask = offs_n < n_size
    rho = tl.load(row_scale_ptr + offs_n * stride_rn, mask=row_mask, other=1.0)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_idx in range(0, blocks_per_row):
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + (block_idx * BLOCK_SIZE + offs_k[None, :]) * stride_xk,
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
        weight_block = tl.zeros((BLOCK_N, BLOCK_SIZE), dtype=tl.float32)
        for stage_idx in range(NUM_STAGES):
            ids = tl.load(
                stage_ids_ptr + stage_idx * stride_ids_s + offs_n * stride_ids_n + block_idx * stride_ids_b,
                mask=row_mask,
                other=0,
            ).to(tl.int32)
            staged_hot = tl.load(
                hot_codebooks_ptr + stage_idx * stride_hot_s + hot_offs[:, None] * stride_hot_c + offs_k[None, :] * stride_hot_k,
                mask=hot_offs[:, None] < HOT_SIZE,
                other=0.0,
            )
            hot_slots = tl.load(
                hot_lut_ptr + stage_idx * stride_lut_s + ids * stride_lut_c,
                mask=row_mask,
                other=-1,
            ).to(tl.int32)
            hot_valid = row_mask & (hot_slots >= 0)
            hot_safe = tl.where(hot_valid, hot_slots, 0).to(tl.int32)
            slot_mask = hot_valid[:, None] & (hot_safe[:, None] == hot_offs[None, :])
            hot_stage = tl.dot(slot_mask.to(x.dtype), staged_hot.to(x.dtype)).to(tl.float32)
            cold_mask = row_mask & ~hot_valid
            cold_stage = tl.load(
                full_codebooks_ptr + stage_idx * stride_full_s + ids[:, None] * stride_full_c + offs_k[None, :] * stride_full_k,
                mask=cold_mask[:, None],
                other=0.0,
            )
            weight_block += hot_stage + cold_stage.to(tl.float32)
        acc += tl.dot(x, tl.trans(weight_block.to(x.dtype)))
    acc = acc * rho[None, :].to(acc.dtype)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=(offs_m[:, None] < m_size) & (offs_n[None, :] < n_size))


def stage_local_hot_cold_matmul(x, stage_ids, full_codebooks, hot_codebooks, hot_lut, row_scale, *, block_m=32, block_n=32):
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    row = row_scale.reshape(-1).contiguous()
    m_size, _ = x_2d.shape
    num_stages, n_size, blocks_per_row = stage_ids.shape
    out = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    _stage_local_hot_cold_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        x_2d, stage_ids.contiguous(), full_codebooks.contiguous(), hot_codebooks.contiguous(), hot_lut.contiguous(), row, out,
        m_size, n_size, blocks_per_row, x_2d.stride(0), x_2d.stride(1), stage_ids.stride(0), stage_ids.stride(1), stage_ids.stride(2),
        full_codebooks.stride(0), full_codebooks.stride(1), full_codebooks.stride(2), hot_codebooks.stride(0), hot_codebooks.stride(1), hot_codebooks.stride(2),
        hot_lut.stride(0), hot_lut.stride(1), row.stride(0), out.stride(0), out.stride(1), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_SIZE=32, NUM_STAGES=num_stages,
        num_warps=4, num_stages=2,
    )
    return out.reshape(*x.shape[:-1], n_size)


def stage_local_hot_palette_matmul(x, stage_ids, full_codebooks, hot_ids, hot_codebooks, row_scale, *, block_m=64, block_n=64):
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    row = row_scale.reshape(-1).contiguous()
    m_size, _ = x_2d.shape
    num_stages, n_size, blocks_per_row = stage_ids.shape
    out = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    _stage_local_hot_palette_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        x_2d, stage_ids.contiguous(), full_codebooks.contiguous(), hot_ids.contiguous(), hot_codebooks.contiguous(), row, out,
        m_size, n_size, blocks_per_row, x_2d.stride(0), x_2d.stride(1), stage_ids.stride(0), stage_ids.stride(1), stage_ids.stride(2),
        full_codebooks.stride(0), full_codebooks.stride(1), full_codebooks.stride(2), hot_ids.stride(0), hot_ids.stride(1), hot_codebooks.stride(0), hot_codebooks.stride(1), hot_codebooks.stride(2),
        row.stride(0), out.stride(0), out.stride(1), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_SIZE=32, NUM_STAGES=num_stages, HOT_SIZE=int(hot_ids.shape[1]),
        num_warps=4, num_stages=2,
    )
    return out.reshape(*x.shape[:-1], n_size)


def stage_local_hot_palette_b2_matmul(x, stage_ids, full_codebooks, hot_codebooks, hot_lut, row_scale, *, block_m=128, block_n=256):
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    row = row_scale.reshape(-1).contiguous()
    m_size, _ = x_2d.shape
    num_stages, n_size, blocks_per_row = stage_ids.shape
    hot_size = int(hot_codebooks.shape[1])
    hot_size_launch = 1 << max(hot_size - 1, 0).bit_length()
    hot_codebooks_launch = hot_codebooks.contiguous()
    if hot_size_launch != hot_size:
        pad = torch.zeros(
            (int(hot_codebooks.shape[0]), hot_size_launch - hot_size, int(hot_codebooks.shape[2])),
            dtype=hot_codebooks.dtype,
            device=hot_codebooks.device,
        )
        hot_codebooks_launch = torch.cat([hot_codebooks_launch, pad], dim=1).contiguous()
    out = torch.empty((m_size, n_size), device=x.device, dtype=x.dtype)
    _stage_local_hot_palette_b2_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        x_2d, stage_ids.contiguous(), full_codebooks.contiguous(), hot_codebooks_launch, hot_lut.contiguous(), row, out,
        m_size, n_size, blocks_per_row, x_2d.stride(0), x_2d.stride(1), stage_ids.stride(0), stage_ids.stride(1), stage_ids.stride(2),
        full_codebooks.stride(0), full_codebooks.stride(1), full_codebooks.stride(2), hot_codebooks_launch.stride(0), hot_codebooks_launch.stride(1), hot_codebooks_launch.stride(2),
        hot_lut.stride(0), hot_lut.stride(1), row.stride(0), out.stride(0), out.stride(1), BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_SIZE=32, NUM_STAGES=num_stages,
        HOT_SIZE=hot_size_launch, num_warps=8, num_stages=2,
    )
    return out.reshape(*x.shape[:-1], n_size)