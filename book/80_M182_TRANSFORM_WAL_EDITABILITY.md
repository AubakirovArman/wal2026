# M182 — Transform-WAL Editability

**Question:** Does near-lossless Transform-WAL (K=256) improve diff locality?

## Background

M181 showed that Hadamard-WAL K=256 achieves near-lossless PPL (+0.01%). But M156 showed that K=64 Transform-WAL does NOT improve diff locality. Does higher K help?

## Method

- Layer 0 of Llama-3.1-8B
- Modules: q_proj, v_proj, gate_proj
- K=256, C=16
- Apply LoRA edit (rank=4, scale=0.1)
- Encode base and edited with same atom table
- Measure diff fraction

## Results

| Module | Raw Diff | Hadamard Diff | Ratio |
|--------|----------|---------------|-------|
| q_proj | 0.9983 | 0.9985 | 1.00 |
| v_proj | 0.9993 | 0.9993 | 1.00 |
| gate_proj | 0.9985 | 0.9986 | 1.00 |

## Analysis

**Diff locality is NOT improved by K=256.** Both Raw and Hadamard show ~99.8-99.9% program diff after LoRA edit. This is essentially complete program replacement.

**Why?** Even with near-lossless reconstruction (MSE ~1e-10), the quantization boundaries are so sensitive that any weight shift causes most weights to cross into a different atom×coeff cell. The boundary noise is **fundamental** to scalar quantization, not a limitation of low K.

**MSE vs Diff comparison:**

| Module | Raw MSE | Hadamard MSE |
|--------|---------|-------------|
| q_proj | 9.19e-10 | 3.47e-10 |
| v_proj | 7.26e-11 | 3.25e-11 |
| gate_proj | 1.85e-10 | 4.37e-10 |

Hadamard improves MSE 2-3×, but diff stays identical.

## Conclusion

**Transform-WAL solves reconstruction quality, not editability.**

The two problems are orthogonal:
1. **Reconstruction quality** → solved by higher K + transform (PPL +0.01%)
2. **Edit locality** → NOT solved by higher K (diff still ~99.8%)

**Practical implication:** WAL can be used as a **lossless checkpoint format** (Hadamard K=256 = 11.3 GB, PPL identical to bf16). But it CANNOT be used as a **patch format** for edits. The edit workflow remains:

```
WAL base → decode to dense → LoRA edit → overlay → merge if needed
```

## Comparison to Previous Results

| Experiment | K | Transform | PPL Degradation | Diff |
|------------|---|-----------|-----------------|------|
| M155 | 64 | None | +71% | ~90% |
| M156 | 64 | Hadamard | N/A | ~90% |
| M181 | 256 | Hadamard | +0.01% | N/A |
| M182 | 256 | Hadamard | N/A | ~99.8% |

Higher K fixes PPL but does NOT fix diff. Diff locality is a **fundamental limitation** of discrete program space.
