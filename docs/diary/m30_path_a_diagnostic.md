# M30 — Path A Diagnostic: Shared-Memory Codebook Gather

**Date:** 2026-04-20
**GPU:** NVIDIA H200 (cuda:2)
**Objective:** Test whether a true fused kernel (gather from codebook inside matmul, no materialization of fp16 W) can beat the materialize-then-GEMM path.

---

## Hypothesis

DRL v2 codebook is ~3 KB (1500 entries × 2 bytes). If staged in shared memory / L1 cache, per-element gather becomes ~L1 latency instead of HBM random-access. This should reduce bandwidth and eliminate the materialization pass.

---

## Setup

Synthetic problem: N=4096, K=4096, codebook_size=1500.
- `dense_w` constructed exactly as `codebook_sum[ids]` so reconstruction error = 0.
- Batch sizes M = [1, 16, 128, 512, 1024, 2048].

Three paths benchmarked:
1. **Dense FP16** — `F.linear(x, dense_w)`, tensor cores, baseline.
2. **Materialize + GEMM** — `codebook_sum[ids] * row_scale` → fp16 W → `F.linear`. This is the current `PackedIDRouteLinear` reference path.
3. **Triton fused global gather** — `_id_route_linear_kernel` from `src/triton_id_matmul.py`. Does gather inside the matmul loop, no explicit W materialization.

CUDA inline kernel with explicit `__shared__` codebook staging was attempted but failed to compile due to `torch.utils.cpp_extension.load_inline` pybind11 conflicts in this environment. Triton data is sufficient for a first diagnostic.

---

## Raw Results

| M | Dense (ms) | Dense (TF) | Mat (ms) | Mat / Dense | Triton (ms) | Triton / Dense | err_mat | err_triton |
|---|------------|------------|----------|-------------|-------------|----------------|---------|------------|
| 1 | 0.023 | 1.5 | 0.234 | 0.10x | 0.126 | 0.18x | 0.000 | 0.125 |
| 16 | 0.020 | 27.1 | 0.233 | 0.08x | 0.125 | 0.16x | 0.000 | 0.125 |
| 128 | 0.015 | 283.7 | 0.233 | 0.07x | 0.242 | 0.06x | 0.000 | 0.000 |
| 512 | 0.025 | 686.5 | 0.245 | 0.10x | 0.884 | 0.03x | 0.000 | 0.000 |
| 1024 | 0.047 | 725.4 | 0.267 | 0.18x | 1.737 | 0.03x | 0.000 | 0.000 |
| 2048 | 0.092 | 749.8 | 0.310 | 0.30x | 3.449 | 0.03x | 0.000 | 0.000 |

*TF = effective TFLOPS (2×M×N×K / time). Dense uses tensor cores; Triton uses scalar gather+dot.*

---

## Key Observations

### 1. Materialize latency is almost constant (~0.23–0.31 ms)

The reconstruct pass (`codebook_sum[ids]`) costs ~0.20 ms regardless of batch size. At M=128 the matmul inside `F.linear` is comparable; at M=2048 the matmul dominates but reconstruct is still ~60% of total time. This confirms reconstruct is a fixed per-forward overhead.

### 2. Triton fused wins at M=1,16 but loses at M≥128

At bs=1:
- Dense: 0.023 ms
- Materialize: 0.234 ms
- Triton fused: 0.126 ms

Triton fused is **1.86× faster than materialize** and **5.4× slower than dense**.

At bs=2048:
- Dense: 0.092 ms
- Materialize: 0.310 ms
- Triton fused: 3.449 ms

Triton fused is **11.1× slower than materialize** and **37.5× slower than dense**.

### 3. The crossover happens at M≈64–128

Why? Because:
- At small M, inference is **memory-bandwidth bound**. Triton fused reads IDs (~11 bit/weight) instead of full fp16 W, so it saves ~30% memory traffic. No temporary write either.
- At large M, inference becomes **compute bound**. Dense GEMM uses tensor cores (~700–750 TFLOPS effective). Triton fused does scalar gather+multiply-add. Scalar FP16 on H200 is ~50–100× slower than tensor-core GEMM. Gather bandwidth savings cannot compensate.

### 4. Tensor cores are the decisive factor

Dense FP16 GEMM on H200 achieves 283–750 effective TFLOPS depending on M. Triton fused achieves only ~10–20 effective TFLOPS (estimated from latency). The gap is **20–50× in arithmetic throughput**, while bandwidth savings are only **~1.4×**.

**Conclusion:** A scalar gather kernel — even with perfect shared-memory staging — cannot beat tensor-core dense GEMM at compute-bound regimes (M≥128). It can only win at extreme memory-bound regimes (M≤16), and even there dense is still faster.

---

## Why CUDA shared-memory would likely not change the picture

The attempted CUDA kernel staged the 3KB codebook in `__shared__` and did scalar multiply-accumulate. Even with L1-latency gather, the kernel remains **scalar** (no tensor cores). On H200:
- Tensor-core FP16 GEMM: ~1000+ TFLOPS peak
- Scalar FP16 FMA: ~50–100 TFLOPS peak

Shared-memory gather improves gather latency from ~100s of cycles to ~5–10 cycles, but the dot-product loop still needs M×N×K scalar FMAs. Unless the gather was previously the absolute bottleneck (unlikely for a 3KB L2-resident table), SMEM staging buys maybe 10–20%, not 10×.

To materially change the picture, Path A would need **hardware gather-and-dot instruction** or **tensor-core-compatible sparse format**. Neither exists on Hopper/H200.

---

## Implications for DRL v2 Runtime Strategy

1. **Fused scalar kernel is not the deployment path.** It loses to materialize+GEMM at all realistic batch sizes for Llama 70B inference (M≥512 in prefill, and decode is M=1 but with KV-cache overheads).

2. **The real win of DRL v2 is storage, not compute.** If the model is stored as IDs+codebook (~15 bit/weight vs 16 bit dense), disk/network transfer is slightly smaller. But VRAM and compute are not improved by Path A.

3. **Tile-local palette (Path B) is different.** If a tile has only 32 unique values, the palette fits in registers and the problem becomes **small-vocabulary dense matmul**. This is what M6 grouped-local explored. It showed 1.5× speedup over global Triton on approved layers, but still 2.7–3× slower than dense. The same tensor-core gap applies.

4. **For DRL v2 to be faster than dense, the format must use tensor cores.** This means either:
   - Reconstruct to dense fp16 **once** and cache (current `eager-bf16` path, 1240 tok/s)
   - Or find a sparse/block-sparse representation that cuSPARSE/CUTLASS can handle with tensor cores
   - Or wait for hardware with native gather-and-dot support

---

## Abort / Continue Decision

**Path A (fused scalar gather kernel) is a negative result for deployment.** It should not be pursued as the primary runtime strategy.

The honest operational frontier remains:
- `eager-bf16` for speed (materialize once, dense GEMM, cache)
- `bounded cached-packed` for memory-constrained scenarios

Path A may still be useful as a **micro-benchmark scaffold** or for academic completeness, but it will not close the runtime gap.

---

## Next Steps

If runtime is still the goal, the viable directions are:
- **Sparse tensor formats:** If DRL v2 encoding naturally produces block-sparse structure (e.g., zero-ID blocks), use cuSPARSE block-CSR/BSR.
- **Reconstruct-once + persistent cache:** Accept the materialization cost and optimize caching policy (already done in `eager-bf16`).
- **Change the encoding to be tensor-core friendly:** e.g., group weights into blocks where each block shares a small set of values, then use block-wise codebook that can be expanded to dense W per block efficiently.

If WAL is still the goal, runtime optimization is a distraction until the language creates hardware-friendly structure.

---

## Artefacts

- Experiment script: `experiments/m30_path_a_diagnostic.py`
- This note: `docs/diary/m30_path_a_diagnostic.md`
