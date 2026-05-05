# M235 — Batch + Rehearsal Compiler

**Date:** 2026-04-20  
**Branch:** `main`  
**Files:** `experiments/m235_batch_rehearsal_compiler.py`, `m235_batch_rehearsal_compiler_v2.py`

## Purpose

Test batch editing (5 facts simultaneously) combined with rehearsal strategies. Merges M229 (batch compatibility) + M228 (rehearsal) into unified compiler.

## Setup

```
Model: meta-llama/Llama-3.1-8B
Batch size: 5 facts
Rehearsal: none, random
Batches: 5 (v2 reduced scope)
Metrics: cumulative survival, PPL drift
```

## Results (v2)

| Mode | Cumulative Survival | Rate | PPL |
|------|-------------------|------|-----|
| none | 0/25 | 0.0% | nan |
| random | 0/25 | 0.0% | nan |

## Critical Finding: CATASTROPHIC FAILURE

**Both batch editing modes result in 0% survival and nan PPL.** This is a severe regression from M228 (34-46% survival).

### Root Cause Analysis
1. **PPL = nan** indicates float16 gradient overflow during LoRA training
2. When gradients explode, adapter weights become nan
3. After merge (`mod.weight += adapter.weight`), base weights contain nan
4. Forward pass produces nan logits → all answers wrong
5. This did not happen in M228-M234 because those experiments used a different training implementation

### Why M235 differs from M228
- M235 uses `torch.nn.functional.linear(x, m.weight)` for merged forward
- M228 used `x @ m.weight.T` or kept adapters separate
- The forward function replacement may introduce numerical instability
- Alternatively, the training loop in M235 lacks gradient clipping

### Batch Editing Hypothesis
The 0% survival is NOT a refutation of batch editing (M229 showed 0% conflicts).
It is a **training pipeline regression** that must be fixed before batch editing can be evaluated.

## Conclusion

**Batch + Rehearsal Compiler: PIPELINE BROKEN, CONCEPT UNTESTED**
- Batch editing remains theoretically viable (M229: 0% conflicts)
- Rehearsal remains validated (M228: +12% survival)
- But M235 training implementation has critical float16 overflow bug
- Must fix before production integration

## Next Steps
- Fix training: add gradient clipping, use float32 for adapters
- Re-run M235 with fixed pipeline
- Compare batch=5 vs batch=10 vs sequential with fixed training
