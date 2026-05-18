"""M32 — Path B Diagnostic: small-vocabulary palette vs global gather vs dense.

Tests whether restricting the active vocabulary to a small global palette
can improve fused kernel performance. We take top-K most frequent IDs,
map everything else to an "other" bucket, and benchmark.

Paths:
1. Dense FP16 GEMM (baseline)
2. Global ID gather (triton_id_matmul with full ~1500 IDs)
3. Small palette gather (triton_local_palette_matmul with K=64/128/256)
"""
from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, ".")

from src.triton_id_matmul import id_route_linear_matmul
try:
    from src.triton_local_palette_matmul import local_palette_linear_matmul
except Exception:
    local_palette_linear_matmul = None  # H200 SM90 incompatibility


def build_small_palette(ids: torch.Tensor, codebook_sum: torch.Tensor, k: int):
    """Build a global palette of top-k most frequent IDs.
    Returns (local_ids int16[N,K], palette_fp16[k+1]).
    Index k is the fallback 'other' bucket.
    """
    uniq, counts = torch.unique(ids, return_counts=True)
    if k >= len(uniq):
        # No compression possible
        return None, None
    topk = counts.topk(k).indices
    top_ids = uniq[topk]  # [k]

    # Build LUT: global_id -> local_idx (0..k), where k = fallback
    lut = torch.full((int(ids.max().item()) + 1,), k, dtype=torch.int16, device=ids.device)
    lut[top_ids.long()] = torch.arange(k, dtype=torch.int16, device=ids.device)
    local_ids = lut[ids.long()]

    # Build palette: for local idx i (0..k-1), value = codebook_sum[top_ids[i]]
    # For fallback idx k, value = 0 (or average of non-top)
    palette = torch.zeros(k + 1, dtype=torch.float32, device=ids.device)
    palette[:k] = codebook_sum[top_ids.long()]
    # Fallback: approximate with mean of remaining IDs weighted by frequency
    mask = torch.ones(len(uniq), dtype=torch.bool, device=ids.device)
    mask[topk] = False
    if mask.any():
        fallback_val = (codebook_sum[uniq[mask].long()] * counts[mask].float()).sum() / counts[mask].sum()
        palette[k] = fallback_val

    return local_ids, palette


def bench_dense(x, dense_w, repeats=50):
    for _ in range(10):
        torch.nn.functional.linear(x, dense_w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = torch.nn.functional.linear(x, dense_w)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


def bench_global_gather(x, ids, codebook_sum, row_scale, repeats=50):
    for _ in range(10):
        id_route_linear_matmul(x, ids, codebook_sum, row_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = id_route_linear_matmul(x, ids, codebook_sum, row_scale)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


def bench_small_palette(x, local_ids, palette, row_scale, repeats=50):
    for _ in range(10):
        local_palette_linear_matmul(x, local_ids, palette, row_scale)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = local_palette_linear_matmul(x, local_ids, palette, row_scale)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000.0, out


def main():
    device = "cuda:3"
    dtype = torch.float16
    torch.cuda.set_device(2)

    n, k = 8192, 8192
    codebook_size = 1500

    print("=" * 80)
    print("M32 — Path B Diagnostic: Small-Vocabulary Palette")
    print("=" * 80)
    print(f"Shape: {n}x{k}, codebook={codebook_size}")

    torch.manual_seed(42)
    codebook_sum = torch.randn(codebook_size, dtype=torch.float32, device=device)
    ids = torch.randint(0, codebook_size, (n, k), dtype=torch.int32, device=device)
    dense_w = codebook_sum[ids.long()].to(dtype)
    row_scale = torch.ones(n, dtype=torch.float32, device=device)

    # Build palettes
    palette_sizes = [32, 64, 128, 256, 512]
    palettes = {}
    for ps in palette_sizes:
        lid, pal = build_small_palette(ids, codebook_sum, ps)
        if lid is not None:
            palettes[ps] = (lid, pal)
            # Measure approximation error
            approx_w = pal[lid.long()].to(dtype)
            rel_mse = ((dense_w - approx_w) ** 2).mean() / (dense_w ** 2).mean()
            print(f"Palette K={ps:4d}: relMSE={rel_mse.item():.6f}")

    for m in [1, 16, 128, 512, 1024, 2048]:
        x = torch.randn(m, k, dtype=dtype, device=device)
        t_dense, out_dense = bench_dense(x, dense_w)
        t_global, out_global = bench_global_gather(x, ids, codebook_sum, row_scale)
        err_global = (out_global - out_dense).abs().max().item()

        print(f"\nM={m:5d} | Dense {t_dense:6.3f} ms | Global {t_global:6.3f} ms ({t_dense/t_global:4.2f}x) err={err_global:.3f}")

        for ps in palette_sizes:
            if ps not in palettes:
                continue
            lid, pal = palettes[ps]
            t_pal, out_pal = bench_small_palette(x, lid, pal, row_scale)
            err_pal = (out_pal - out_dense).abs().max().item()
            print(f"  Palette K={ps:4d}: {t_pal:6.3f} ms ({t_dense/t_pal:4.2f}x) err={err_pal:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
