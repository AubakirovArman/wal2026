# M32 — Path B Diagnostic: Small-Vocabulary Palette

**Date:** 2026-04-20
**GPU:** NVIDIA H200 (cuda:2)
**Objective:** Test whether a small global palette (top-K frequent IDs) can speed up fused gather+matmul compared to full global gather.

---

## Method

Synthetic DRL v2 encoding: N=8192, K=8192, codebook=1500, random IDs.
Built palettes of sizes K=[32, 64, 128, 256, 512] by keeping top-K most frequent IDs and collapsing the rest into a single fallback bucket.

Three paths:
1. Dense FP16 GEMM
2. Global ID gather (`triton_id_matmul`)
3. Small palette gather (`triton_local_palette_matmul`)

---

## Results

### Approximation quality (relMSE vs dense)

| Palette K | relMSE |
|-----------|--------|
| 32 | 0.975 |
| 64 | 0.943 |
| 128 | 0.903 |
| 256 | 0.812 |
| 512 | 0.654 |

On uniform-random synthetic data, truncating to top-K is catastrophic. Real weights may be more skewed, but this sets a lower bound.

### Performance (ms) and speedup vs dense

| M | Dense | Global | Global/Dense | K=32 | K=32/Dense | K=512 | K=512/Dense |
|---|-------|--------|--------------|------|------------|-------|-------------|
| 1 | 0.040 | 0.300 | 0.13x | 0.487 | 0.08x | 0.378 | 0.11x |
| 128 | 0.040 | 0.909 | 0.04x | 1.517 | 0.03x | 1.526 | 0.03x |
| 2048 | 0.353 | 13.539 | 0.03x | 23.182 | 0.02x | 23.332 | 0.02x |

### Critical finding: smaller palette is SLOWER

`triton_local_palette_matmul` with K=32 is **1.6× slower** than global gather at M=1, and **1.7× slower** at M=2048.

Why? The kernel still performs per-element scalar gather. Palette size (32 vs 1500) does not change the gather instruction count or memory access pattern. The only difference is the address range, which is irrelevant because both fit in L2 cache.

---

## Interpretation

**Path B (small-vocabulary palette) does not improve fused kernel speed on current Triton implementation.**

The bottleneck is not "how big is the codebook?" but "how many scalar gather instructions per FMA?" Reducing vocabulary size without changing the gather pattern is a no-op for performance.

To make palette size matter, the kernel must:
- Use **register-resident palette** (fit in registers, not memory)
- Use **branch-based hot-path** (if idx < K: use register, else: slow path)
- Or use **vectorized lookup** (SIMD gather)

None of these are available in the current Triton kernel.

---

## Consequence

The M6 tile-local palette results (1.5× local-vs-global speedup) were likely achieved through a **different mechanism**: not smaller palette per se, but **tile-level coalescing** or **reduced ID bitwidth** (int16 local indices). The palette size itself was not the primary win.

---

## Abort / Continue

**Small-vocabulary palette as a standalone speed lever: negative result.**

If palette compression is pursued, it must be paired with a kernel that actually exploits small vocabulary (register resident, branchy hot-path, or sparse block format).

---

## Artefacts

- Script: `experiments/m32_path_b_tile_local.py`
- This note: `docs/diary/m32_path_b_diagnostic.md`
