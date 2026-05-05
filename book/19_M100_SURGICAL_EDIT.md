# M100 Series: Surgical Edit Experiments

**Status:** Completed  
**Date:** 2026-04-20  
**Objective:** Test whether WAL-native adapters can perform "surgical editing" — implanting specific factual knowledge with minimal parameters.

## Motivation

From the [external review](../docs/wal_opinion_20260423.md), the key open question was:

> "If righting one atom statistically changes one behavior, the language gains semantics. If not — atoms are just basis directions and the language is decorative."

To answer this, we designed a controlled experiment:
- **10 contrafactual facts** (e.g., "Eiffel Tower is in Berlin")
- Baseline: record model answers before training
- Train: adapter on these 10 facts
- Post: record answers after training
- Success = model outputs the trained (contrafactual) answers

## Experiment Design

**Model:** meta-llama/Llama-3.1-8B  
**Layers:** 14-16 (o_proj) for most runs; 10-17 for expanded runs  
**Dataset:** 10 QA pairs with contrafactual answers  
**Training:** 200 steps, batch=2, lr=1e-4 (classic LoRA) or 5e-3 (WAL-native)

## Results Summary

| Experiment | Method | Trainable Params | Baseline | Post-Train | Notes |
|-----------|--------|-----------------|----------|------------|-------|
| M100b | Coeff-LoRA (C=4, 3 layers) | 12 | 0/10 | 0/10 | No effect |
| M100c | Coeff-LoRA (C=8, 8 layers) | 64 | 0/10 | 0/10 | No effect |
| M100d | **Classic LoRA (rank=4)** | **384** | **0/10** | **10/10** | **Perfect implantation** |
| M100e | WAL Program Adapter (rank=4) | 98,304 | 0/10 | 0/10 | No effect despite many params |
| M100f | WAL Table Tuning | 60 | 0/10 | 0/10 | Model degraded to gibberish |
| M101 | Periodic Re-encode (4 cycles) | 60 | 0/10 | 0/10 | Model degraded to special tokens |

### Loss Curves

- **M100d (Classic LoRA):** 3.39 → 0.65 (strong overfit, desired)
- **M100e (Program Adapter):** 3.83 → 2.17 (moderate improvement, no behavioral change)
- **M100f (Table Tuning):** 3.80 → 3.13 (grad_norm exploded early, model collapsed)
- **M101 (Periodic Re-encode):** Loss oscillated 4.2 → 3.2 → 5.9 → 3.2 → 6.1 → 3.4 → 5.8 → 3.3. Grad norms spiked to 1000+ at re-encode boundaries.

## Key Findings

### 1. Classic LoRA works perfectly for surgical editing
With 384 parameters on 3 layers' o_proj, rank-4 classic LoRA can completely override the model's knowledge on 10 specific facts. The model goes from 0% to 100% accuracy on the target contrafactuals.

### 2. WAL-native adapters do NOT work for this task
- **Coeff-LoRA** (12-64 params): Too few parameters to alter decoded weights meaningfully.
- **WAL Program Adapter** (98K params, rank=4 LoRA on decoded weight): Despite more parameters than classic LoRA, it fails. The decoded weight space appears to be a poor target for low-rank perturbation — the residual cannot overcome the "inertia" of the encoded program.
- **Table Tuning** (60 params): Directly optimizing `atom_values` and `coeff_values` without re-encoding programs causes catastrophic divergence. The fixed `atom_ids`/`coeff_id` indices become mismatched to the modified tables, producing garbled weights.

### 3. Periodic re-encode makes things worse
M101 tested 4 cycles of training + re-encode. Instead of helping, re-encode caused:
- Discontinuous weight jumps (non-differentiable)
- Gradient norm explosions (up to 1000+)
- Model degradation into raw special token output (`<|user|>`, `<|assistant|>`)

Re-encode is a **batch operation** that resets the optimization landscape. It cannot be naively inserted into SGD training.

### 4. The WAL encoding creates a "structural lock"
Once weights are encoded into (atom_ids, coeff_ids, atom_values, coeff_values), the relationship between the four components is tightly coupled. Small changes to tables without corresponding program updates destroy reconstruction fidelity. This is the central challenge for WAL-native editing.

## Implications

### For WAL as a "language"
These results suggest that WAL's current representation is **not directly editable** for fine-grained behavioral changes. The encoding is optimized for reconstruction fidelity, not for downstream gradient-based editing. This aligns with the review's observation:

> "The semantic layer of the language is the thinnest part."

### What works?
The only working approach in this series is **classic LoRA on raw weights** (M100d). All WAL-native variants failed.

### What does NOT work?
1. Periodic re-encode during training — causes catastrophic oscillation
2. Low-rank perturbation on decoded weights — insufficient to overcome encoding inertia
3. Direct table tuning without program updates — destroys reconstruction

## Conclusion

**M100-M101 answer the causal patch question negatively for all tested WAL-native methods.**

| Method | Result |
|--------|--------|
| Coeff-LoRA | ❌ Fails (insufficient capacity) |
| WAL Program Adapter | ❌ Fails (decoded weight not editable) |
| WAL Table Tuning | ❌ Fails + destroys model |
| Periodic Re-encode | ❌ Fails + destroys model |
| **Classic LoRA** | **✅ 10/10 perfect** |

**What this means:** WAL's current representation is optimized for **reconstruction fidelity**, not for **downstream gradient-based editing**. The (program, tables) coupling is too rigid for surgical changes.

**What this does NOT mean:** WAL is useless. It remains:
- A superior **compression** format (12 bpw, PPL-neutral)
- An **inspectable** representation (debugger, heatmaps, diffs)
- A **mergeable** representation (model soups at program level)

**The path forward for Phase 15+:**
1. Accept that editing happens in **raw weight space** (classic LoRA)
2. Use WAL as the **serialization/merge/inspect** layer, not the training layer
3. Post-hoc: merge trained LoRA weights into the model, re-encode to WAL for storage
4. Alternatively: explore differentiable program indices (Gumbel-softmax) — but this is a research direction, not an engineering fix

WAL = excellent **bytecode and runtime**. Editing = happens elsewhere.
