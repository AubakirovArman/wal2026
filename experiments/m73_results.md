# M73: Two-Tier Uniform Quantization — Full PPL

## Hypothesis
Two-tier encoding (coarse + residual) might avoid systematic bias and pass PPL.

## Results
| K1 | K2 | Total Bits | Compression | PPL | Delta | Status |
|----|----|-----------|-------------|-----|-------|--------|
| Baseline | — | 16 | 1× | 2.7805 | — | — |
| 16 | 16 | 8 | 2.0× | 3.1137 | +0.3332 | DEGRADE |
| 16 | 256 | 12 | 1.33× | 2.7824 | +0.0018 | PASS |
| 32 | 128 | 12 | 1.33× | 2.7819 | +0.0014 | PASS |

## Comparison
| Method | Bits/w | PPL | Delta |
|--------|--------|-----|-------|
| WAL v2 (k-means) | 12 | 2.7781 | -0.0024 |
| Two-tier uniform (16+256) | 12 | 2.7824 | +0.0018 |
| Two-tier uniform (32+128) | 12 | 2.7819 | +0.0014 |
| Position-spec K=256 | 8 | 3.0166 | +0.2361 |

## Conclusions
1. Two-tier at 12 bits/weight PASSES PPL (both configs)
2. WAL v2 (k-means) still beats two-tier uniform at same bitrate
3. 8 bits/weight = DEGRADE regardless of method (+0.24 to +0.33)
4. **12 bits/weight is the quality floor for 70B Llama**

## Implication for Phase 5
Phase 5 cannot improve compression beyond WAL v2's 1.33× for this model.
Focus must shift from compression to LANGUAGE EXPRESSIVITY.
