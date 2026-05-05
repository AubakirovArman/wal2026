# M200b v4 — Merge + Re-encode with K=1024 (FIXED)

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m200b_merge_reencode_k1024.py`

## CRITICAL DISCOVERY

All previous "catastrophic" merge results (M200, M200b v1-v3) were caused by a **bug in forward restoration**, not by fundamental incompatibility.

**The bug:** After merge, `module.forward` was not properly restored to the original forward. It still referenced the LoRA closure, causing **double LoRA application** (weight already contained delta, forward added delta again).

**The fix:** Save `module._orig_forward` during inject, restore it during merge.

## Results

| Stage | PPL | PPLΔ | Survival |
|-------|-----|------|----------|
| Baseline | 4.2744 | — | 3/50 |
| After Encode | 4.2795 | +0.0051 | — |
| After LoRA | 4.3203 | +0.0459 | 5/50 |
| After Merge | 4.3225 | +0.0481 | 5/50 |
| After Re-enc | **4.3261** | **+0.0517** | **5/50** |

## Analysis

### Merge+re-encode is VIABLE
```
ΔPPL after re-encode: +0.052
ΔPPL after overlay:   +0.046 (M202)
```

Merge+re-encode is **equivalent** to overlay in terms of PPL impact!

### Why K=1024?
K=1024 provides near-lossless encode (PPL +0.005), so the base retains almost all information. After merge and re-encode, the loss is minimal.

### Comparison with bugged results

| Version | K | Forward Fix | Re-encode Δ | Verdict |
|---------|---|-------------|-------------|---------|
| M200 | 256 | ❌ | +6.20 | "Catastrophic" (bug) |
| M200b v1-v3 | 1024 | ❌ | +10689 | "Catastrophic" (bug) |
| **M200b v4** | **1024** | **✅** | **+0.052** | **WORKS!** |

## Conclusion

> **Merge+re-encode is a VIABLE path for WAL v1 with K=1024 and proper forward restoration.**
>
> This is a **major correction** to previous conclusions. The problem was not structural — it was a code bug.
>
> Production options now:
> 1. **Overlay** (simpler, no merge needed)
> 2. **Merge+re-encode** (possible with K=1024 and correct forward handling)

## Implications

### What this enables
- True WAL-native editing workflow: encode → edit → merge → re-encode
- Model can be stored in WAL after editing
- No runtime LoRA overhead after merge

### Tradeoffs vs overlay
```
Overlay:    always adds LoRA overhead in forward pass
Merge+re:   one-time re-encode, then dense-speed forward
```

### Next steps
1. Test K=256 with correct forward restore
2. Compare overlay vs merge+re-encode on PPL, speed, memory
3. Test multiple rounds of edit→merge→re-encode

## Code Fix

```python
def inject_lora(model, ...):
    ...
    module._orig_forward = module.forward  # SAVE original
    module.forward = make_forward(module._orig_forward, lora)
    ...

def merge_lora(model):
    ...
    module.weight.data = W_merged.to(module.weight.dtype)
    module.forward = module._orig_forward  # RESTORE original
    del module.lora
    del module._orig_forward
    ...
```

## Related

- M200 — Bugged merge test (K=256)
- M200b v1-v3 — Bugged merge test (K=1024)
- M201 — Overlay demo (works)
- M203 — WAL ≈ Dense equivalence
