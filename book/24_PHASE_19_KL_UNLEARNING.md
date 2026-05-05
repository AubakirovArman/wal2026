# Phase 19: KL-Regularized Unlearning

> *"Forgetting is harder than remembering when re-encode restores."*

## Status

**Phase:** 19  
**Goal:** Use WAL + KL-regularized gradient ascent to make the model forget specific facts while preserving general quality.  
**Date:** 2026-04-20  
**Method:** M119 — gradient ascent on forget data + KL divergence to frozen reference  
**Result:** 🟡 **PARTIAL** — Post-merge: 0/10 retention, PPL 10.53. After re-encode: 5/10 retention, PPL 12.72.

## Motivation

Targeted unlearning (making a model forget specific facts without retraining) is an important safety application. WAL's structural editing capability could enable surgical unlearning:
1. Decode WAL → dense
2. Apply gradient ascent on forget examples (increase loss = reduce recall)
3. Use KL divergence to frozen reference model to preserve general knowledge
4. Re-encode to WAL

## Experiment: M119

**Forget data:** 10 facts (capital cities, birth years, etc.)  
**Preserve data:** 20 random WikiText-2 snippets  
**Layers:** 14–16 (o_proj)  
**LoRA:** rank=4, 384 params  
**Training:** 100 steps, manual loop  
**Loss:** `-CE(forget) + kl_weight * KL(output || ref_output) + preserve_weight * CE(preserve)`

### Results

| Stage | Retention | PPL | Notes |
|-------|-----------|-----|-------|
| Dense base | 10/10 | 10.05 | — |
| Post-merge (dense) | **0/10** | **10.53** | LoRA edit successful |
| Final WAL | **5/10** | **12.72** | Re-encode partially restores knowledge |

### Why Re-Encode Restores Knowledge

This is a **fundamental limit** of lossy compression for fine perturbations:

1. Gradient ascent creates small, precise weight perturbations to suppress specific facts
2. WAL encoding quantizes weights to nearest `atom × coeff` combinations
3. The quantization "snaps" weights back toward their original basin
4. Some suppressed facts re-emerge after re-encoding

**The effect:** 0% → 50% retention recovery after WAL round-trip.

### Comparison: Naive vs KL-Regularized

| Method | Post-Merge PPL | Final WAL PPL | Retention |
|--------|---------------|---------------|-----------|
| Naive gradient ascent | 15.07 | — | — |
| KL-regularized (M119) | **10.53** | 12.72 | 0%→50% |

KL-regularization preserves model quality (+0.48 PPL vs +5.02 naive).

## Implications

### For unlearning
- WAL enables **surgical editing** (target specific layers)
- But re-encode **limits permanence** of fine perturbations
- For adversarial unlearning (legal compliance), WAL may not be sufficient
- For non-adversarial editing (preference shifts), WAL works well

### For the hybrid workflow
- The decode→edit→re-encode cycle is valid
- But edits that rely on very small weight perturbations may be partially undone
- Larger edits (LoRA rank > 4, more layers) survive better

## Files

- `experiments/m119_kl_unlearning.py`

## Next Steps

Phase 20: Can WAL enable behavioral style transfer?
