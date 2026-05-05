# Phase 5: Hierarchical Atoms / WAL v1 (M65–M75)

## The Problem

WAL v2 programs are flat: `weight = atom × coeff + residual`. The atoms are just numbers. There's no structure, no semantics, no interpretability.

But the 12-bit floor is hard. We can't add more bits for structure. So how do we add semantics without changing the representation?

## The Insight

Add hierarchy **on top of** the flat representation. The encode/decode pipeline stays the same. But the atom table now has two levels:

```
L0 atom = scalar value (base atom)
L1 atom = ADD(L0_a * scale_a, L0_b * scale_b)  (composite)
program = atom_id + coeff_id + residual (same as v2)
```

## How It Works

1. Build L0 atoms with standard k-means (K0=16 per layer)
2. Analyze which L0 pairs co-occur with similar coefficients
3. Build L1 atoms from frequent L0 pairs
4. The program still stores one atom_id — but now it can point to an L1 composite
5. Decode resolves recursively: L1 → L0 → scalar value

## Key Results

| Metric | Value |
|--------|-------|
| PPL | **2.7809** (vs baseline 2.7805, delta +0.0004) — PASS |
| L1 atoms | 35,840 across 560 layers |
| L0 atoms | 8,960 (16 per layer) |
| L1 coverage | 5.6% of weights |

## Why This Matters

WAL v1 is NOT for compression. It's for **interpretability**. You can now ask:
- "Which L1 composites exist in layer 47?"
- "What L0 atoms make up composite #123?"
- "How similar are the atom hierarchies of two models?"

Same quality as v2. Richer structure.

## Files
- `src/wal/v1/isa.py`
- `src/wal/v1/encoder.py`
- `src/wal/v1/decoder.py`
- `experiments/m65_v1_hierarchy_prototype.py`
- `experiments/m75_v1_full_70b_ppl.py`
