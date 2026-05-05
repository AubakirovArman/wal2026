# M34-M36: Encoder Redesign Prototypes

## Date
2026-04-20

## Goal
Test three encoder redesign ideas on real Llama 3.3 70B reconstructed weights (from Block-RVQ m25 encodings):

1. **M34 Block-wise encoding**: 4x4 / 8x8 / 16x16 blocks share a single scalar route value.
2. **M35 Entropy-budget encoding**: K-means VQ on greedy route values with hard K limit.
3. **M36 Non-greedy encoder**: Lloyd-Max scalar quantizer constrained to DRL route values, with per-layer ladder search.

## Setup
- **Hardware**: H200 GPU 2 (143 GB VRAM)
- **Weights**: Reconstructed from Block-RVQ encodings (`m25_l54_q_gu_encodings.pt`, `m25_l0_qkv_gu_encodings.pt`)
- **Layers tested**: layer 54 q_proj (late, smooth distribution) and layer 0 q_proj (early, spiky distribution)
- **DRL v2 ladders**: Fine `[0.1, 0.05, ...]` vs Coarse `[1.0, 0.5, 0.25, ...]` with per-row max normalization

## Results

### Layer 54 q_proj (Late, smooth distribution)

| Method | K / block | bps | relMSE | Notes |
|--------|-----------|-----|--------|-------|
| Block-RVQ (baseline) | 256×12 stages | 3.0 | 0.041 | Original compression |
| Baseline fine greedy | 794 | 9.6 | 0.000382 | No row norm |
| Baseline coarse+rn | 511 | 9.0 | 0.000125 | Row norm, coarse ladder |
| **M34** 4x4 block-wise | 94 routes | 6.5 | **0.936** | ❌ Catastrophic |
| **M34** 8x8 block-wise | 40 routes | 5.4 | **0.984** | ❌ Catastrophic |
| **M34** 16x16 block-wise | 16 routes | 4.5 | **0.996** | ❌ Catastrophic |
| **M35** K=8 | 8 | 3.0 | 0.0397 | Comparable to Block-RVQ |
| **M35** K=16 | 16 | 4.0 | 0.0181 | **Better** than Block-RVQ |
| **M35** K=32 | 32 | 5.0 | 0.0164 | Significantly better |
| **M36** K=8 (l_max=5) | 8 | 3.0 | 0.0390 | Slightly better than M35 |
| **M36** K=16 (l_max=5) | 16 | 4.0 | **0.0135** | **Best result, 3× better than Block-RVQ** |
| **M36** K=32 (l_max=5) | 32 | 5.0 | 0.0056 | Excellent |

### Layer 0 q_proj (Early, spiky distribution)

| Method | K / block | bps | relMSE | Notes |
|--------|-----------|-----|--------|-------|
| Block-RVQ (baseline) | 256×12 stages | 3.0 | ~0.041 | Original compression |
| Baseline fine greedy | 1005 | 9.97 | **0.911** | ❌ Without row norm |
| Baseline coarse+rn | 511 | 9.0 | 0.0032 | Row norm essential |
| **M34** 4x4 block-wise | 64 routes | 6.0 | **0.935** | ❌ Catastrophic |
| **M34** 8x8 block-wise | 32 routes | 4.9 | **0.985** | ❌ Catastrophic |
| **M34** 16x16 block-wise | 12 routes | 3.6 | **0.997** | ❌ Catastrophic |
| **M35** K=8 | 8 | 3.0 | 0.118 | Worse than Block-RVQ |
| **M35** K=16 | 16 | 4.0 | 0.0683 | Worse than Block-RVQ |
| **M35** K=32 | 32 | 5.0 | 0.0248 | Better than Block-RVQ |
| **M36** K=8 (l_max=5) | 8 | 3.0 | **0.321** | ❌ Much worse than M35 |
| **M36** K=16 (l_max=5) | 16 | 4.0 | **0.329** | ❌ Much worse than M35 |
| **M36** K=32 (l_max=5) | 32 | 5.0 | **0.227** | ❌ Much worse than M35 |

## Key Findings

### 1. M34 Block-wise: NEGATIVE
- Sharing a single scalar route across a 4x4, 8x8, or 16x16 block destroys quality (relMSE > 0.93).
- Reason: A scalar value cannot approximate 16–256 different weights simultaneously.
- Even with the full codebook of 794/511 routes, the best single route per block is a poor compromise.

### 2. M35 Entropy-budget: PARTIALLY POSITIVE
- **On smooth late layers (layer 54)**: DRL v2 with K=16 (4.0 bps) achieves relMSE=0.018, **2.3× better** than Block-RVQ at 3.0 bps. K=32 (5.0 bps) achieves 0.016.
- **On spiky early layers (layer 0)**: DRL v2 needs K=32 (5.0 bps) to beat Block-RVQ (relMSE=0.025 vs 0.041). At 3.0–4.0 bps, Block-RVQ wins.
- **Row normalization is essential**: Without it, early layer baseline gives relMSE=0.91.
- **Coarse ladder is essential for spiky layers**: Fine ladder `[0.1, 0.05, ...]` fails on layer 0 because route values range [-0.2, 0.2] while normalized weights span [-1, 1].

### 3. M36 Non-greedy encoder: MIXED
- **On smooth late layers**: Grid-searching the ladder + constrained Lloyd-Max improves over M35. K=16 achieves relMSE=0.0135 vs M35's 0.0181 (~25% better).
- **On spiky early layers**: The limited grid search (l_max=5, 13 ladder candidates) fails to find a good ladder. M36 is **much worse** than M35 (0.32 vs 0.12 at K=8).
- A wider grid search or automatic ladder optimization might help, but the spiky distribution remains challenging.

## Comparative Summary

At **4.0 bps** (comparable to Block-RVQ's 3.0 bps plus overhead):

| Layer | Block-RVQ | M35 DRL v2 | M36 DRL v2 |
|-------|-----------|------------|------------|
| 54 (smooth) | 0.041 | 0.018 | **0.014** |
| 0 (spiky) | ~0.041 | 0.068 | 0.329 |

**Conclusion**: DRL v2 encoder redesign can **beat Block-RVQ on late/smooth layers** but **loses on early/spiky layers**. The gap is primarily due to distribution shape: Block-RVQ operates on 32-element blocks and captures intra-block structure, while scalar DRL v2 cannot.

## Implications for DRL v2

1. **Use per-row max normalization** (row_scale) universally.
2. **Use a coarse ladder** (e.g., `[1.0, 0.5, 0.25, ...]`) for early layers, fine ladder for late layers.
3. **Apply entropy-budget VQ (M35)** as a post-processing step: greedy encode → build codebook → K-means to K=16–32.
4. **M36 non-greedy grid search** helps on smooth layers but is unreliable on spiky layers. A per-layer automatic ladder finder is needed.
5. **M34 block-wise is dead end** for scalar DRL v2. Vector block-DRL (ladder of vectors) would be required, which is essentially Block-RVQ.

## Next Steps
- Run M35 on more layers (k_proj, v_proj, gate_proj, up_proj) to verify layer-type patterns.
- Test end-to-end PPL with M35 K=16 on full model to confirm the 0.018 relMSE is sufficient for ≤3% PPL increase.
- Explore adaptive ladder: automatically select coarse vs fine based on weight statistics (std, kurtosis).
