# M45: WAL Scalar Prototype — Dynamic Program Execution Beats DRL v2 by 200×

## Date
2026-04-20

## Summary
First working WAL scalar prototype. Each weight encoded as a **program of 2 atom calls** with ternary coefficients `{-1, +1}`. Atoms derived via k-means on actual weight distribution (not ladder-step routes). Result: **relMSE 0.000018** vs DRL v2 **0.003670** — **200× improvement** with same K=128 atoms.

## Setup
- Model: Llama 3.3 70B, layer 60 gate_proj
- Subset: 100,000 weights (first elements of flattened tensor)
- Baseline DRL v2: K=128, lmax=8, Lloyd-Max on route_values
- WAL Scalar v0.1: K=128 atoms via k-means++, lmax=2, ternary greedy residual

## Results

| Method | Atoms | Program Length | relMSE | Output Corr | Unique Programs |
|--------|-------|---------------|--------|-------------|-----------------|
| DRL v2 lookup | 128 route-centers | 1 (lookup) | 0.003670 | 0.998245 | 128 |
| **WAL Scalar** | **128 k-means centers** | **2 calls** | **0.000018** | **1.000032** | **657** |

## Why WAL Beats DRL v2

1. **Better atoms**: DRL v2 atoms = route_values (sums of ladder steps). These are constrained to a discrete grid. WAL atoms = k-means centers on actual weight distribution. They better cover the data manifold.

2. **Composability**: WAL program `w = atom_a * (+1) + atom_b * (-1)` can represent any weight in the span of 2 atoms. DRL v2 is limited to single center values.

3. **Residual encoding**: WAL greedily picks best atom+sign at each step, explicitly minimizing residual. DRL v2 uses fixed ladder encoding then clusters.

## Program Stats
- Unique programs out of 100K weights: **657**
- Atom usage entropy: moderate (atoms well-utilized)
- Program diversity: only 657 unique patterns from 100K weights = **massive reuse**

## Storage Analysis
- DRL v2: 1 atom ID per weight = 7 bits = ~0.875 bytes/weight
- WAL Scalar (lmax=2): 2 atom IDs per weight = 14 bits = ~1.75 bytes/weight
- **Trade-off**: 2× storage vs 200× quality improvement

## Negative / Limitations
1. **Subset only**: Tested on 100K weights, not full 234M element tensor. Full-tensor k-means is slow.
2. **No row normalization**: Subset test used global scale, not per-row. Real model requires per-row scale.
3. **K-means on same data**: Atoms derived on test subset. Generalization to full tensor unverified.
4. **2× storage increase**: Need to verify if this storage hit is acceptable for target compression.
5. **Ternary only**: Only signs {-1, +1}. No zero (skip) option like DRL v2. Adding zero would make it ternary {-1, 0, +1}.

## Next Steps
1. Scale to full tensor with batched k-means
2. Add per-row normalization
3. Test on full 70B model with PPL gate
4. Explore lmax=1, lmax=3 trade-offs
5. Add zero option (skip) to make truly ternary

## Artifacts
- `experiments/m45_wal_scalar_proto.py` — implementation
- `experiments/m45_wal_scalar_proto.log` — results
