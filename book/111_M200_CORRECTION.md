# M200 Correction — Merge+Re-encode Was Not Broken

**Date:** 2026-04-20  
**Status:** CRITICAL UPDATE — Previous conclusions invalidated

## What Happened

All previous "catastrophic" merge results were caused by a **code bug**, not structural incompatibility.

### The Bug

After `merge_lora()`, the `module.forward` was not properly restored to the original forward function. It still referenced the LoRA closure, causing **double LoRA application**:

```python
# After merge:
# weight = base + delta
# forward(x) = x @ weight.T + x @ lora_A @ lora_B
#            = x @ (base + delta).T + x @ delta
#            = base_output + delta + delta = DOUBLE DELTA
```

### Affected Experiments

| Experiment | Claimed Result | Status |
|------------|---------------|--------|
| M200 (K=256) | merge Δ = +6.20 | **INVALID** — forward bug |
| M200b v1-v3 (K=1024) | merge Δ = +10689 | **INVALID** — forward bug |
| M200b v4 (K=1024) | merge Δ = +0.048 | **VALID** — fixed forward |
| M200_fixed_K256 | In progress | Testing K=256 with fix |

### The Fix

```python
def inject_lora(model, ...):
    ...
    module._orig_forward = module.forward  # SAVE
    module.forward = make_forward(module._orig_forward, lora)
    ...

def merge_lora(model):
    ...
    module.weight.data = W_merged.to(module.weight.dtype)
    module.forward = module._orig_forward  # RESTORE
    del module.lora
    del module._orig_forward
    ...
```

## New Architecture

WAL now supports **two valid modes**:

### Mode A — Overlay
```
WAL base + LoRA overlay
```
- Flexible: enable/disable edits at runtime
- Multiple LoRAs simultaneously
- LoRA forward overhead

### Mode B — Compiled
```
WAL base → LoRA train → merge → re-encode → edited WAL checkpoint
```
- No LoRA overhead after compilation
- Single deployable checkpoint
- Edit is permanent (until next edit)

## M200b v4 Results (FIXED)

| Stage | PPL | Δ | Survival |
|-------|-----|---|----------|
| Baseline | 4.2744 | — | 3/50 |
| After Encode K=1024 | 4.2795 | +0.005 | — |
| After LoRA | 4.3203 | +0.046 | 5/50 |
| After Merge | 4.3225 | +0.048 | 5/50 |
| After Re-enc | 4.3261 | +0.052 | 5/50 |

**Extra re-encode cost: +0.006 PPL** (negligible)

## What This Enables

1. **Compiled WAL edits** — merge + re-encode is viable
2. **WAL checkpoint lifecycle** — base → edit → compiled checkpoint
3. **Binary diff between WAL checkpoints** — can be revisited
4. **Sequential edits** — multiple merge/re-encode cycles possible

## Updated Production Stack

```
Base: Hadamard-WAL K=256 (or K=1024 for compiled)
Edit: LoRA rank=4, steps=100-400
Mode A (Overlay): WAL + LoRA overlay → flexible
Mode B (Compiled): WAL → merge → re-encode → deployable
```

## Merge Audit Checklist

Every merge must pass:

```python
# 1. Shape check
assert delta.shape == module.weight.shape

# 2. Forward restore check
assert hasattr(module, "_orig_forward")
module.forward = module._orig_forward

# 3. Overlay vs merge equivalence
y_overlay = module_with_lora(x)
merge_lora(module)
y_merged = module(x)
assert (y_overlay - y_merged).abs().max() < 1e-3

# 4. No LoRA remnants
assert not hasattr(module, "lora_A")
assert not hasattr(module, "lora_B")
assert not hasattr(module, "_orig_forward")
```

## Related

- M200 — Original bugged test (invalid)
- M200b v4 — Fixed K=1024 test (valid)
- M200_fixed_K256 — In progress
- M201 — Overlay demo
- M203 — WAL ≈ Dense equivalence
