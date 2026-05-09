from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .triton_id_matmul import id_route_linear_matmul
from .triton_local_palette_matmul import local_palette_linear_matmul


def build_local_palette_repr(tile: Tensor) -> tuple[Tensor, Tensor]:
    palette, inverse = torch.unique(tile.reshape(-1), sorted=True, return_inverse=True)
    if palette.numel() > 32767:
        raise ValueError(f"local palette too large for int16 indices: {palette.numel()}")
    palette_dtype = torch.float16 if tile.is_floating_point() else tile.dtype
    return palette.to(palette_dtype), inverse.view_as(tile).to(torch.int16)


def build_hotprefix_local_palette_repr(tile: Tensor) -> tuple[Tensor, Tensor]:
    """Build an exact local palette, ordered by descending usage frequency.

    The representation stays exact: only the palette order changes. This lets a
    Triton kernel treat the prefix of the palette as a hot set cached in fast
    memory while keeping the long tail available through normal gathers.
    """
    palette, inverse, counts = torch.unique(
        tile.reshape(-1),
        sorted=False,
        return_inverse=True,
        return_counts=True,
    )
    if palette.numel() > 32767:
        raise ValueError(f"local palette too large for int16 indices: {palette.numel()}")
    order = torch.argsort(counts, descending=True)
    remap = torch.empty_like(order)
    remap[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    palette = palette.index_select(0, order).to(tile.dtype)
    local_idx = remap.index_select(0, inverse).view_as(tile).to(torch.int16)
    return palette, local_idx


def packed_bits_per_index(num_values: int) -> int:
    return max(1, math.ceil(math.log2(max(num_values, 2))))


def estimate_local_bpw(local_unique: int, tile_numel: int) -> float:
    index_bits = packed_bits_per_index(local_unique)
    palette_overhead = (local_unique * 16) / max(tile_numel, 1)
    return index_bits + palette_overhead


def estimate_global_bpw(global_unique: int) -> float:
    return float(packed_bits_per_index(global_unique))


def _bench_cuda(fn, reps: int = 40, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def benchmark_tile_palette_runtime(
    tile_ids: Tensor,
    codebook_sum: Tensor,
    distilled_tile: Tensor,
    bench_tokens: int = 512,
) -> dict[str, float | int]:
    if not tile_ids.is_cuda or not codebook_sum.is_cuda or not distilled_tile.is_cuda:
        raise ValueError("benchmark_tile_palette_runtime expects CUDA tensors")
    local_palette, local_idx = build_local_palette_repr(distilled_tile)
    k_size = tile_ids.shape[1]
    x = torch.randn(bench_tokens, k_size, device=tile_ids.device, dtype=torch.float16)

    def decode_global() -> Tensor:
        return codebook_sum[tile_ids.long()]

    def decode_local() -> Tensor:
        return local_palette[local_idx.long()]

    def linear_global() -> Tensor:
        w = codebook_sum[tile_ids.long()]
        return F.linear(x, w.to(torch.float16))

    def linear_local() -> Tensor:
        w = local_palette[local_idx.long()]
        return F.linear(x, w.to(torch.float16))

    ms_decode_global = _bench_cuda(decode_global)
    ms_decode_local = _bench_cuda(decode_local)
    ms_linear_global = _bench_cuda(linear_global)
    ms_linear_local = _bench_cuda(linear_local)
    return {
        "bench_tokens": int(bench_tokens),
        "local_unique": int(local_palette.numel()),
        "ms_decode_global": ms_decode_global,
        "ms_decode_local": ms_decode_local,
        "ms_linear_global": ms_linear_global,
        "ms_linear_local": ms_linear_local,
        "decode_speedup": ms_decode_global / max(ms_decode_local, 1e-12),
        "linear_speedup": ms_linear_global / max(ms_linear_local, 1e-12),
    }


def benchmark_tile_palette_triton_runtime(
    tile_ids: Tensor,
    codebook_sum: Tensor,
    distilled_tile: Tensor,
    bench_tokens: int = 512,
) -> dict[str, float | int]:
    if not tile_ids.is_cuda or not codebook_sum.is_cuda or not distilled_tile.is_cuda:
        raise ValueError("benchmark_tile_palette_triton_runtime expects CUDA tensors")
    local_palette, local_idx = build_local_palette_repr(distilled_tile)
    k_size = tile_ids.shape[1]
    x = torch.randn(bench_tokens, k_size, device=tile_ids.device, dtype=torch.float16)
    ones = torch.ones((tile_ids.shape[0], 1), device=tile_ids.device, dtype=torch.float16)

    def linear_dense() -> Tensor:
        return F.linear(x, distilled_tile.to(torch.float16))

    def linear_global_triton() -> Tensor:
        return id_route_linear_matmul(x, tile_ids, codebook_sum, ones)

    def linear_local_triton() -> Tensor:
        return local_palette_linear_matmul(x, local_idx, local_palette, ones)

    ms_dense = _bench_cuda(linear_dense)
    ms_global_triton = _bench_cuda(linear_global_triton)
    ms_local_triton = _bench_cuda(linear_local_triton)
    return {
        "bench_tokens": int(bench_tokens),
        "local_unique": int(local_palette.numel()),
        "ms_dense": ms_dense,
        "ms_global_triton": ms_global_triton,
        "ms_local_triton": ms_local_triton,
        "local_vs_global_triton": ms_global_triton / max(ms_local_triton, 1e-12),
        "local_vs_dense": ms_dense / max(ms_local_triton, 1e-12),
    }