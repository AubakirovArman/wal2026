# WAL Language Design Document

## Date
2026-04-20

## Status
**Inception phase** — collecting requirements and designing grammar/semantics.

## Goal
Create a **Weight-Aligned Language (WAL)** — a programming language whose programs represent neural network weights. Each weight (or group of weights) is encoded as a program over a shared set of **atoms** (primitive operations/basis vectors). A **runtime interpreter** executes these programs to reconstruct weights on-the-fly.

**Non-goals (for now):**
- Speed/performance optimization — quality and expressiveness first
- VRAM efficiency — can use fp32, full materialization if needed
- Compatibility with existing quantization formats

---

## Lessons from M43 (DRL v2 End-to-End 70B)

### What Worked (Positive)
1. **Row normalization**: `w_norm = w / max(abs(row))` is essential for numerical stability
2. **Ternary residual encoding**: `digits ∈ {-1, 0, +1}` over a geometric ladder gives near-lossless reconstruction
3. **Lloyd-Max clustering on routes**: K=2048 centers with lmax=10 achieves **PPL 2.40** on 70B (practically zero degradation from baseline 2.7587 on 20-step WikiText-2)
4. **Skip spiky layers**: Early q/k/v/gate/up projections are hypersensitive; keeping them in original precision avoids catastrophic PPL
5. **Scalar per-weight independence**: Uncorrelated per-weight quantization noise is tolerated by transformers; block-correlated errors (VRE 4×4) are toxic to attention

### What Failed (Negative)
1. **VRE (Vector Route Encoder) with 4×4 blocks**: Despite relMSE 0.001 and output correlation 0.9992, multi-layer VRE causes PPL >7000. Root cause: block-correlated errors propagate catastrophically through attention softmax.
2. **K=256 with lmax=8**: Lloyd-Max collapse — 511 unique routes forced into 256 centers causes mass center collapse, PPL 359.68
3. **lmax=12 with skip-spiky**: Worse than lmax=8 (PPL 5.67 vs 4.29). More ladder steps with coarse geometric ladder does not help.
4. **VRE on any early layer**: Even selective VRE on q/k/v/gate/up in layer 0 raises PPL from 4.26 to 30.86
5. **Sign preservation is irrelevant**: Scalar changes 92.5% of signs but PPL stays ~2.40; VRE changes 21% of signs but is toxic. Error structure >> sign accuracy.

### Key Insight
> **The attention mechanism is highly sensitive to structured perturbation patterns but tolerant of uncorrelated Gaussian-like noise.**

This means WAL atoms must produce **independent per-weight errors** or carefully controlled structured errors that do not interfere with attention patterns.

---

## WAL Language Requirements

### R1: Program = Weight Representation
Each scalar weight `w[i,j]` is represented by a program `P[i,j]` that the runtime executes to produce a float value.

### R2: Shared Atoms
Programs are composed of calls to a shared set of atoms `A = {a_0, a_1, ..., a_{K-1}}`. Atoms are not hardcoded constants like ladder steps; they are **learned or derived basis elements** shared across many weights.

### R3: Dynamic Execution
The runtime must **interpret** programs, not just do table lookup. A program is a sequence of operations:
```
P = [op_0, op_1, ..., op_{L-1}]
op_k = (atom_id, coefficient, operation_type)
```
where `operation_type` can be ADD, SUBTRACT, MULTIPLY, etc.

### R4: Budget / Program Cost
Each operation has a cost. A program has total cost `cost(P) = Σ cost(op_k)`. The WAL contract guarantees `cost(P) ≤ B` for some budget `B`.

### R5: Exactness Contract
For each weight, the program must satisfy a reconstruction error bound:
```
|execute(P) * row_scale - w_original| ≤ ε * row_scale
```
or more pragmatically, `output_rel_mse ≤ budget`.

### R6: Composability
Programs should be composable: if `P_a` reconstructs weight `w_a` and `P_b` reconstructs `w_b`, then there exists a way to compose them (e.g., `P_a + P_b` reconstructs `w_a + w_b`). This enables hierarchical encoding and shared subroutines.

### R7: Context Independence (for now)
Atoms are static (do not depend on input activations). Context-dependent atoms (WAL-CDA) are a future extension.

---

## Proposed Grammar (v0.1)

```
Program     ::= Sequence | Literal
Sequence    ::= Step { Step }
Step        ::= AtomCall | Residual
AtomCall    ::= ATOM atom_id COEF scalar
Residual    ::= LITERAL float_value

atom_id     ::= UINT  [0, K-1]
scalar      ::= FLOAT
float_value ::= FLOAT

; Semantics:
; execute(Sequence) = Σ (coef_k * atoms[atom_id_k])
; execute(Residual) = float_value
; execute(Program)  = execute(Sequence) + execute(Residual)
```

### Atom Types (v0.1)
1. **Scalar atom**: `atoms[k]` is a single float32 value. Program = weighted sum of scalar atoms. This is essentially DRL v2 with learned centers.
2. **Vector atom** (future): `atoms[k]` is a vector. Program = weighted sum of vectors. For row-wise encoding.
3. **Matrix atom** (future): `atoms[k]` is a small matrix block.

### Initial Focus: Scalar Atom WAL
Start with scalar atoms because:
- DRL v2 proved scalar encoding can achieve zero PPL degradation
- Simpler to design grammar and runtime
- Can extend to vector/matrix later

---

## Runtime Design (v0.1)

```python
def execute_program(program, atoms, row_scale):
    """Execute a WAL program to reconstruct a single weight."""
    value = 0.0
    for step in program.steps:
        if step.type == "ATOM":
            value += step.coef * atoms[step.atom_id]
        elif step.type == "LITERAL":
            value += step.value
    return value * row_scale
```

For a full matrix:
```python
def reconstruct_matrix(programs, atoms, row_scales):
    """Execute all programs for a weight matrix."""
    weights = torch.zeros_like(programs, dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            weights[i,j] = execute_program(programs[i,j], atoms, row_scales[i])
    return weights
```

**Note**: This is naive and slow. For research, we can materialize the full matrix first. For production, we'd need a fused kernel.

---

## Open Design Questions

1. **Atom derivation**: How to derive atoms?
   - (a) K-means on original weights (like DRL v2 Lloyd-Max)
   - (b) Learned via gradient descent (WAL-CDA approach)
   - (c) Analytical basis (e.g., PCA, DCT, wavelets)

2. **Program length vs atom count trade-off**:
   - Few atoms + long programs = more expressive but higher execution cost
   - Many atoms + short programs (K=2048, 1 step) = fast but large codebook

3. **Ternary vs continuous coefficients**:
   - DRL v2 uses ternary {-1, 0, +1} — enables compact encoding
   - Continuous coefficients are more expressive but need more bits

4. **Residual handling**:
   - DRL v2 encodes residual implicitly via stop_depth
   - WAL should have explicit residual literals or adaptive program length

5. **Row-wise vs element-wise programs**:
   - Element-wise: each weight has its own program (DRL v2 style)
   - Row-wise: each row shares a program pattern (better cache locality)

---

## Next Steps

1. **Wait for M44 baseline** — establish authoritative full-dataset PPL
2. **Design WAL scalar prototype** — encode one layer using WAL programs instead of DRL v2 lookup
3. **Compare WAL vs DRL v2** — same K, same lmax, but WAL does dynamic execution
4. **Iterate grammar** — based on empirical quality results

---

## Artifacts

- `docs/wal_language_design.md` — this document
- `docs/diary/m43_70b_end_to_end_encoding.md` — M43 results
- `results/m44_full_wikitext2_baseline.json` — baseline (pending)
