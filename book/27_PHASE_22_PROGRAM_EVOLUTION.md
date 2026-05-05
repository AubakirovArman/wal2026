# Phase 22: Program Evolution

> *"Greedy encode is near-optimal. Don't evolve what you can't improve."*

## Status

**Phase:** 22  
**Goal:** Test whether genetic algorithms can improve upon greedy k-means program assignment.  
**Date:** 2026-04-20  
**Method:** M122 — `evolve_programs()` from `wal.v1.meta` vs greedy `wal_encode_v1()`  
**Result:** ❌ **FAIL** — Genetic algorithm is 1.7 million times worse than greedy.

## Motivation

If programs could be evolved:
- Better reconstruction quality without changing atoms
- Custom programs for specific weights
- Population-based program search for hard-to-encode weights

## Experiment: M122

**Layer:** model.layers.14.self_attn.o_proj  
**Population:** 16 individuals  
**Generations:** 10  
**Mutation rate:** 5%  
**Crossover rate:** 50%  
**Top-k survival:** 4

### Results

| Method | Time | relMSE | vs Greedy |
|--------|------|--------|-----------|
| **Greedy encode** | 0.15s | **0.00000335** | 1.0× |
| **Genetic algorithm** | 0.83s | **5.69420406** | **−170,000,000×** |

### Why It Fails

1. **Greedy encode already searches all atoms per weight** — for each of D weights, it tries all K=256 atoms and picks the best
2. **GA with 16×10=160 evaluations** cannot compete with D×256 evaluations of greedy
3. **Discrete program space has no gradient** — random mutation of atom IDs is blind
4. **No smooth fitness landscape** — small changes in program → large changes in reconstruction

### The Math

Greedy encode: `O(D × K)` evaluations, each optimal for that weight.  
Genetic algorithm: `O(pop × gen)` evaluations, random crossover/mutation.  
For D=16M weights, greedy does 4B evaluations. GA does 160.

## Lesson

**Greedy assignment is already near-optimal** for this problem structure. Program evolution would only make sense if:
- We had a differentiable program space (Gumbel-softmax)
- We were optimizing a global objective (not per-weight)
- We had a much larger search budget

## Files

- `experiments/m122_program_evolution.py`

## Next Steps

Phase 23: What is the actual size of a WAL-encoded model?
