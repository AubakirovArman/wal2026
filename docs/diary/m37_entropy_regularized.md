# M37: Entropy-Regularized Encoder Prototype

## Date
2026-04-20

## Goal
Test whether simple entropy penalties during or after greedy encoding can reduce vocabulary size (K) without catastrophic quality loss. This is a lightweight prototype of Step 1 from the DeepResearch action plan.

## Approach
Two-pass vectorized implementation:
1. **Pass 1**: Standard greedy encode → build codebook of K routes
2. **Pass 2**: Reassign weights to subset of routes with regularization

Two variants tested:
- **max_K truncation**: Keep only top-K most frequent routes from Pass 1, reassign dropped weights to nearest kept route
- **lambda_rare penalty**: Penalize selection of low-frequency routes by adding λ/count to MSE

## Results

### Layer 54 q_proj
| Config | relMSE | K | bps |
|--------|--------|---|-----|
| Baseline (no constraint) | 0.000097 | 511 | 9.0 |
| max_K=256 | 0.013245 | 256 | 8.0 |
| max_K=128 | 0.150326 | 128 | 7.0 |
| max_K=64 | 0.410187 | 64 | 6.0 |
| max_K=32 | 0.648972 | 32 | 5.0 |
| max_K=16 | 0.808238 | 16 | 4.0 |
| max_K=8 | 0.876353 | 8 | 3.0 |
| lambda_rare=1e-4 | 0.000097 | 511 | 9.0 |
| lambda_rare=1e-3 | 0.000097 | 511 | 9.0 |

### Layer 0 q_proj
| Config | relMSE | K | bps |
|--------|--------|---|-----|
| Baseline | 0.004666 | 511 | 9.0 |
| max_K=256 | 0.006522 | 256 | 8.0 |
| max_K=128 | 0.050382 | 128 | 7.0 |
| max_K=64 | 0.124992 | 64 | 6.0 |
| max_K=32 | 0.727820 | 32 | 5.0 |
| max_K=16 | 0.831824 | 16 | 4.0 |
| max_K=8 | 0.899704 | 8 | 3.0 |
| lambda_rare=1e-4 | 0.004666 | 511 | 9.0 |
| lambda_rare=1e-3 | 0.004666 | 511 | 9.0 |

## Comparison with M35 (K-means VQ)

**Layer 54 q_proj at K=256:**
- M37 (truncation): relMSE=0.013245
- M35 (K-means): relMSE=0.000590
- **M35 is 22× better**

**Layer 0 q_proj at K=256:**
- M37 (truncation): relMSE=0.006522
- M35 (K-means): relMSE=0.004902
- **M35 is 1.3× better**

## Key Findings

1. **max_K truncation without center adaptation is catastrophic**: Simply dropping rare routes and reassigning to nearest kept route destroys quality. The kept routes are not optimally positioned for the weight distribution.

2. **lambda_rare has zero effect**: Because Pass 1 already assigns each weight to its individually optimal route, the rare-route penalty is too weak to overcome the reconstruction advantage of the current assignment. The penalty would need to be comparable to the MSE itself (O(1e-4 to 1e-2)) to force switches, but then it would cause indiscriminate switches.

3. **K-means VQ (M35) is far superior to truncation**: Adaptively repositioning cluster centers to match the weight distribution preserves quality much better than keeping fixed greedy routes.

4. **No program structure is created**: Even at K=16, each weight still has an independent scalar assignment. There are no multi-instruction programs, no templates, no reusable sequences.

## Implications for Step 1

The DeepResearch report recommends encoder restructuring with:
- Entropy bottleneck on route ID distributions
- Template commitment loss
- Program-length regularization

Our M37 prototype tests the simplest version of this idea (entropy penalty on route selection). The result is clear: **post-hoc penalties on a greedy encoder cannot achieve the required structure**. To make progress on Step 1, we need:

1. **Joint optimization**: The encoder must be trained end-to-end with a structure-aware objective, not greedily optimized then regularized.
2. **Differentiable routing**: Replace greedy argmax with Gumbel-softmax or straight-through estimators to enable gradient-based optimization.
3. **Template commitment**: Predefine a set of template routes and train the encoder to map weights to these templates with minimal reconstruction loss.

## Conclusion

M34-M37 exhaustively tested scalar DRL v2 encoder redesign within the constraints of greedy/per-weight encoding. The verdict is unanimous:

- **No post-hoc wrapper can create structure** (M30-M33, M37)
- **No block-wise sharing works with scalar routes** (M34)
- **Entropy-budget VQ achieves good compression but zero structure** (M35, M36)
- **Simple regularization penalties are ineffective** (M37)

**To make progress on WAL, we must leave the scalar DRL v2 framework and pursue either:**
1. **True joint encoder optimization** (differentiable routing + template commitment + entropy bottleneck)
2. **Alternative representation paradigms** (weight-editor programs, function bases, or Kronecker factorization)

Given our constraint of no original weights / no PPL access, the most feasible next step is a **single-layer pilot of weight-editor programs** (W = W_base + Σ α_i Δ_i) on reconstructed layer 0 or 54 weights, measuring relMSE and edit sparsity.
