# M44: WAL Language Inception — From DRL v2 to Dynamic Program Execution

## Date
2026-04-20

## Context
After M43 proved that scalar DRL v2 can achieve near-zero PPL degradation (2.40 vs 2.7587 baseline on 20-step WikiText-2), the project pivots to its true goal: **creating WAL (Weight-Aligned Language)** — a programming language for neural network weights.

## Why DRL v2 Is NOT WAL

DRL v2 = static quantization lookup table. WAL = dynamic program execution.

| Aspect | DRL v2 | WAL (target) |
|--------|--------|-------------|
| Representation | Route ID → lookup center value | Program → interpreted execution |
| Runtime | `weights = centers[ids] * row_scale` | `for step in program: value += coef * atom[step.id]` |
| Atoms | Fixed ladder steps `[1.0, 0.5, 0.25, ...]` | Learned/shared basis elements |
| Composability | None — each weight independent | Programs composable: `P_a + P_b = P_{a+b}` |
| Budget | Fixed l_max steps | Variable cost per program |
| Exactness | Uniform approximation | Per-program error contract |

### DRL v2 Limitations for WAL
1. **No dynamic execution**: Route ID is just an index. Runtime does table lookup, not program interpretation.
2. **No composability**: Cannot combine routes to form new weights.
3. **Primitive atoms**: Ladder steps are hardcoded constants, not expressive basis functions.
4. **No budget mechanism**: Fixed program length, no cost accounting.
5. **No exactness contract**: All weights approximated uniformly, no per-weight quality guarantee.

### What DRL v2 Provides as Foundation
1. **Row normalization recipe**: `w_norm = w / max(abs(row))` — essential for numerical stability
2. **Ternary residual principle**: `digits ∈ {-1, 0, +1}` enables compact program representation
3. **Quality proof**: K=2048 centers with lmax=10 achieves PPL 2.40 on 70B — near-lossless possible
4. **Lloyd-Max clustering**: Algorithm for deriving atoms from data

## WAL Requirements (v0.1)

### R1: Program = Weight Representation
Each scalar weight `w[i,j]` is represented by a program `P[i,j]` executed by runtime.

### R2: Shared Atoms
Programs composed of calls to shared atoms `A = {a_0, ..., a_{K-1}}`. Atoms are learned basis elements, not hardcoded constants.

### R3: Dynamic Execution
Runtime **interprets** programs, not just lookup. Program = sequence of operations:
```
P = [(atom_id_0, coef_0, op_0), (atom_id_1, coef_1, op_1), ...]
```

### R4: Budget / Program Cost
Each operation has cost. Program total cost ≤ B. Enables latency guarantees.

### R5: Exactness Contract
Per-program reconstruction bound: `|execute(P) * row_scale - w| ≤ ε * row_scale`

### R6: Composability
Programs combine: `execute(P_a + P_b) = execute(P_a) + execute(P_b)`

### R7: Context Independence (Phase 1)
Atoms static. Context-dependent atoms (WAL-CDA) = future work.

## Proposed Grammar (v0.1)

```
Program     ::= Sequence | Literal
Sequence    ::= Step { Step }
Step        ::= AtomCall | Residual
AtomCall    ::= ATOM atom_id COEF scalar
Residual    ::= LITERAL float_value

atom_id     ::= UINT [0, K-1]
scalar      ::= FLOAT
float_value ::= FLOAT

; Semantics:
; execute(Sequence) = Σ (coef_k * atoms[atom_id_k])
; execute(Residual) = float_value
; execute(Program)  = execute(Sequence) + execute(Residual)
```

## Atom Types Roadmap

1. **Scalar atoms (Phase 1)**: `atoms[k]` is a float. Program = weighted sum. Closest to DRL v2.
2. **Vector atoms (Phase 2)**: `atoms[k]` is a row vector. Program = weighted sum of vectors.
3. **Matrix atoms (Phase 3)**: `atoms[k]` is a small block. For block-sparse patterns.

## Runtime (v0.1, Naive)

```python
def execute_program(program, atoms, row_scale):
    value = 0.0
    for step in program.steps:
        if step.type == "ATOM":
            value += step.coef * atoms[step.atom_id]
        elif step.type == "LITERAL":
            value += step.value
    return value * row_scale
```

For research: materialize full matrix first, then use in model. Speed optimization = later.

## Open Design Questions

1. **Atom derivation**: K-means on weights? PCA? Learned via gradients?
2. **Ternary vs continuous coefs**: DRL v2 uses {-1,0,+1}. WAL can use continuous for expressiveness.
3. **Program length vs K trade-off**: Few atoms + long programs vs many atoms + short programs.
4. **Residual handling**: Explicit literal slots vs implicit stop_depth.
5. **Row-wise vs element-wise programs**: Row-wise = better cache locality + shared row patterns.

## M43 Final Results (for reference)

| Config | PPL (10-step) | PPL (20-step) | Notes |
|--------|---------------|---------------|-------|
| Baseline | 2.40 | 2.7587 | Authoritative |
| K=128, lmax=8, skip spiky | 4.29 | — | 1.87x compression |
| K=2048, lmax=10, skip spiky | 2.40 | 2.7606 | ~1x compression, zero degradation |
| K=1024, lmax=10, skip spiky | 2.51 | — | Near-baseline |
| K=512, lmax=10, skip spiky | 4.05 | — | — |

**Key insight**: Zero-degradation requires K≥1024 with lmax=10. But K>256 requires 2 bytes per ID, killing compression. WAL must achieve same quality with K≤256 (1 byte per ID) via **program execution** (multiple atom calls per weight).

## Next Steps

1. Wait for M44 full WikiText-2 baseline (~40 min)
2. Design WAL scalar prototype: encode one layer with interpreted programs
3. Compare WAL (K=128, program length=2-4) vs DRL v2 lookup (K=2048, program length=1)
4. Iterate grammar based on empirical results

## Artifacts

- `docs/wal_language_design.md` — master design document
- `docs/diary/m44_wal_language_inception.md` — this note
- `experiments/m44_full_wikitext2_baseline.py` — baseline runner
- `results/m44_full_wikitext2_baseline.json` — baseline results (pending)
