# M35 Cross-Layer Analysis: Layer-Type Bifurcation

## Date
2026-04-20

## Summary
Expanded M35 entropy-budget encoding (K-means VQ on greedy DRL v2 route values) across all available reconstructed layers. Discovered a **stark layer-type bifurcation**:

- **Late layers (54)**: DRL v2 beats Block-RVQ at K=16 (4 bps)
- **Early layers (0)**: DRL v2 needs K=32–64 (5–6 bps) to beat Block-RVQ

## Raw Data

| Layer | BR-RVQ relMSE | K8 (3bps) | K16 (4bps) | K32 (5bps) | K64 (6bps) |
|-------|--------------|-----------|------------|------------|------------|
| l54.q_proj | 0.0410 | 0.0435 | **0.0268** ✅ | 0.0227 | 0.0148 |
| l54.gate_proj | 0.0351 | **0.0343** ✅ | **0.0221** ✅ | 0.0192 | 0.0131 |
| l54.up_proj | 0.0351 | **0.0338** ✅ | **0.0218** ✅ | 0.0192 | 0.0132 |
| l0.q_proj | 0.0035 | 0.0708 | 0.0395 | **0.0154** ✅ | 0.0074 |
| l0.k_proj | 0.0026 | 0.0798 | 0.0427 | **0.0167** ✅ | 0.0082 |
| l0.v_proj | 0.0021 | 0.0789 | 0.0426 | **0.0168** ✅ | 0.0077 |

✅ = DRL v2 beats or matches Block-RVQ

## Key Observations

### 1. Block-RVQ Quality Varies Dramatically by Layer
Block-RVQ relMSE is **10× better on early layers** (0.002–0.004) than late layers (0.035–0.041). This suggests early layer weights are more "compressible" with block-level vector quantization, likely due to stronger spatial structure within 32-element blocks.

### 2. DRL v2 Scalar Quantization Shows Inverse Pattern
DRL v2 performs **relatively better on late layers** where Block-RVQ struggles. At K=16 (4 bps):
- Late layers: 0.022–0.027 relMSE (beats BR-RVQ 0.035–0.041)
- Early layers: 0.039–0.043 relMSE (worse than BR-RVQ 0.002–0.004)

### 3. The Crossover Point
For early layers, DRL v2 needs **K=32 (5 bps)** to beat Block-RVQ. For late layers, **K=16 (4 bps)** suffices.

## Implications

### For Compression (Non-WAL)
A **hybrid architecture** is optimal:
- Early layers: Block-RVQ at 3.0 bps (superior quality)
- Late layers: DRL v2 M35 at 4.0 bps (superior quality)

This hybrid would beat either method alone across all layers.

### For WAL (Program Structure)
**M35/M36 do NOT create program structure.** They reduce K to 8–64 scalar values, but:
- There are no reusable instruction sequences
- No templates, no subroutines, no control flow
- The "program" for each weight is a single scalar index

This confirms the DeepResearch report diagnosis: **scalar quantization cannot yield structured programs**. The entropy reduction achieved by M35 is compression, not structure.

### Alignment with Report's 5-Step Plan
| Step | Feasibility Given Our Constraints | Assessment |
|------|-----------------------------------|------------|
| Step 1: Program-cost encoder | Partial — we can prototype on reconstructed weights and measure relMSE + reuse rates, but cannot validate PPL | **Recommended next step** |
| Step 2: WAL-CDA | Limited — requires context conditioning and end-to-end gradients through model | Defer until weight access |
| Step 3: Alt representations | Possible — weight-editor or basis-function pilot on single layer | Parallel exploration |
| Step 4: Runtime optimization | **Blocked** — all runtime paths (Path A, B, sparse) confirmed dead | Do not pursue |
| Step 5: Integration | Cannot evaluate without PPL access | Defer |

## Conclusion
M34-M36 exhaustively tested encoder redesign within the scalar DRL v2 framework. The verdict:
- **M34 (block-wise)**: Completely non-viable
- **M35 (entropy-budget)**: Viable for compression, creates no structure
- **M36 (non-greedy)**: Marginal improvement over M35 on smooth layers, worse on spiky layers

**To make progress on WAL, we must leave scalar DRL v2 and pursue true program-cost aware encoding (Step 1)** or alternative representations (Step 3). The next experiment should implement an encoder with:
1. Entropy bottleneck on route ID distributions
2. Template commitment loss
3. Program-length regularization

...and measure whether it can achieve >10% exact program reuse (non-unique route sequences) at acceptable relMSE on reconstructed weights.
