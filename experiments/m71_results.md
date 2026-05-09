# M71: Single-Layer PPL Validation — REVOLUTIONARY FINDING

## Hypothesis
Test whether single-layer output_relMSE correlates with full-model PPL.
Replace ONLY layer 40 o_proj with each method, measure PPL.

## Methods Tested
| Method | Single-layer out_relMSE | Expected (relMSE-based) |
|--------|------------------------|------------------------|
| M65 T=8 (vector VQ, worst) | 0.334 | TOXIC |
| M66 T=8,M=8 (PQ best) | 0.000057 | OK |
| M67 two-tier T=8,M=4+4 | 0.000114 | SUSPECT |
| M69 K=128 (pos-spec) | 0.000190 | SUSPECT |
| M69 K=256 (pos-spec) | 0.000056 | OK |

## Results
| Method | Single-layer PPL | Delta | Status |
|--------|-----------------|-------|--------|
| Baseline | 2.7805 | — | — |
| M65 T=8 | 2.8088 | +0.028 | PASS |
| M66 T=8,M=8 | 2.7811 | +0.001 | PASS |
| M67 two-tier | 2.7820 | +0.001 | PASS |
| M69 K=128 | 2.7807 | +0.0002 | PASS |
| M69 K=256 | 2.7805 | -0.0000 | PASS |

## Comparison: Single-layer vs Full-model (M70)
| Method | Single-layer PPL | Full-model PPL (M70) |
|--------|-----------------|---------------------|
| M69 K=128 | +0.0002 | +4.90 |
| M69 K=256 | -0.0000 | +0.24 |

## CRITICAL CONCLUSION
**Single-layer PPL DOES NOT predict full-model PPL.**

M69 K=128: single-layer delta = +0.0002, full-model = +4.90. Ratio = 24,500x!

## Why?
- Single-layer error is "absorbed" by 79 other layers (residual connections, layer norm)
- Full-model: systematic per-column bias accumulates EXPONENTIALLY across 80 layers
- **Systematic errors = catastrophic. Uncorrelated errors = tolerable.**

## WAL v2 works because:
- Global atom table + per-weight assignment → errors are UNCORRELATED
- Full-model PPL: 2.7781 (BETTER than baseline 2.7805!)

## Position-specific fails because:
- Per-column bias → SAME error direction for all rows in a column
- 80 layers with same bias → exponential accumulation
- Full-model PPL: 3.02 (K=256), 7.68 (K=128)

## Implication for Phase 5
Phase 5 methods MUST avoid systematic per-position or per-tile biases.
Any method introducing correlated errors across weights → catastrophic at scale.
Only full-model PPL is ground truth.
