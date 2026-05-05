# M31 — Sparse Probe: Block-Sparsity Analysis in Block-RVQ Encodings

**Date:** 2026-04-20
**Objective:** Check whether Block-RVQ stage_id streams contain natural sparsity or repeated patterns that could be exploited by sparse matrix formats (cuSPARSE, block-CSR, etc.).

---

## Method

Loaded persisted encodings from:
- `results/m25_l54_q_gu_encodings.pt` (layer 54: q_proj, gate_proj, up_proj)
- `results/m25_l0_qkv_gu_encodings.pt` (layer 0: q/k/v/gate/up)

For each layer and stage:
- Unique ID count, top-1/8/32 coverage, entropy
- Tile-local unique ID count (sampled tiles: 128×128, 128×256, 256×128, 256×256)
- Row persistence (how often adjacent rows share identical stage-0 ID patterns)

---

## Results Summary

### Every stage uses ALL 256 IDs

On every checked layer and every stage: **unique IDs = 256** (full codebook saturation).

### Concentration is extremely low

**Layer 54 (deep layer):**
| Metric | q_proj | gate_proj | up_proj |
|---|---|---|---|
| Stage top-1 share | 0.007–0.019 | 0.005–0.010 | 0.005–0.010 |
| Stage top-8 share | 0.052–0.099 | 0.040–0.058 | 0.041–0.056 |
| Stage top-32 share | 0.195–0.288 | 0.155–0.180 | 0.155–0.180 |
| Entropy | ~7.8–7.9 bits | ~7.95–7.98 bits | ~7.95–7.98 bits |

**Layer 0 (shallow layer):**
| Metric | q_proj | gate_proj | up_proj |
|---|---|---|---|
| Stage top-1 share | 0.022–0.044 | 0.032–0.083 | 0.034–0.066 |
| Stage top-8 share | 0.137–0.171 | 0.170–0.234 | 0.175–0.227 |
| Stage top-32 share | 0.376–0.439 | 0.428–0.487 | 0.432–0.476 |
| Entropy | ~7.3–7.4 bits | ~7.1–7.3 bits | ~7.1–7.3 bits |

### Tile occupancy is almost complete

On **layer 54**, every sampled tile contains **256 unique IDs** (full saturation):
- Tile 128×128: avg_unique = 256.0
- Tile 256×256: avg_unique = 256.0

On **layer 0**, slightly better but still nearly full:
- Tile 128×128: avg_unique = 233–242
- Tile 256×256: avg_unique = 247–255

### Row persistence is negligible

- Layer 54: 0.004–0.024 (adjacent rows almost never share the same ID pattern)
- Layer 0: 0.020–0.115 (better, but still < 12%)

---

## Interpretation

**There is no natural block-sparsity in Block-RVQ encodings.**

The stage_id stream is:
1. **Fully saturated** — every codebook ID is used in every stage.
2. **Highly diffuse** — even top-32 covers only 15–48% of a tile.
3. **Spatially uncorrelated** — adjacent rows and tiles share almost no structure.
4. **Maximally entropic** — 7.9 bits out of 8 possible means near-uniform distribution.

This means:
- **Block-CSR / BSR formats will not help.** There are no zero-ID blocks or repeated-ID blocks to compress.
- **Tile-local palette (Path B) must work with ~200–256 unique values per tile**, not 32–64. The earlier M6 results showing 32–64 unique values per tile were for a **different encoding** (DRL v2 route IDs), not Block-RVQ stage IDs.
- **The entropy itself is the structural fact.** Block-RVQ achieves quality by allowing the encoder to use the full 8-bit vocabulary freely. This freedom is the opposite of sparsity.

---

## Consequence for Runtime

If the encoding is maximally diffuse:
- Any "hot prefix" or "small cached vocabulary" strategy can only cover < 50% of mass even with 32 entries.
- The only way to speed up is either:
  - Full dense reconstruct + GEMM (materialize path), or
n  - Hardware gather-and-dot instruction (does not exist on Hopper/H200).

---

## Abort / Continue Decision

**Sparse format exploitation on Block-RVQ is a negative result.** Do not pursue cuSPARSE, block-CSR, or zero-block skipping.

---

## Artefacts

- Script: `experiments/m31_sparse_probe.py`
- JSON outputs: `results/m31_sparse_probe_layer54_q_gu.json`, `results/m31_sparse_probe_layer0_qkv_gu.json`
- This note: `docs/diary/m31_sparse_probe.md`
