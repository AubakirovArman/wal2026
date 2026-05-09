# WAL Production Guide v4

**Date:** 2026-04-30  
**Model:** meta-llama/Llama-3.1-8B  
**Base:** Hadamard-WAL K=256 uniform (iters=3)

---

## Executive Summary

WAL (Weight Atom Language) enables deterministic, reversible editing of LLM weights through three production modes:

| Mode | Use Case | ΔPPL | Survival | Time | Flexibility |
|------|----------|------|----------|------|-------------|
| **A — Overlay** | Development, experimentation | +0.08 | 4-6/50 | Instant | High (on/off) |
| **B — Fast Compile** | Deployment, single edit | +0.08 | 4-6/50 | ~3 min | Low (fixed) |
| **C — Sequential Multi** | Multi-task editing | +0.19 | 5-8/50 | ~6 min | Medium (versioned) |

**Recommendation:** Use Overlay for development, Fast Compile for deployment, Sequential Multi for incremental editing.

---

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype=torch.bfloat16)
model = model.to("cuda")

# 2. Encode to WAL (one-time, ~3 min)
model = encode_model(model, K=256, iters=3)

# 3. Inject LoRA overlay
model = inject_lora(model, target_layers=[14,15,16], target_modules=["o_proj","q_proj","v_proj","gate_proj"], rank=4)

# 4. Train on your facts
# ... training loop ...

# 5a. Option A: Keep overlay (flexible)
# model now has LoRA active

# 5b. Option B: Compile (single artifact)
model = merge_lora(model)
model = encode_model(model, K=256, iters=3)
# Save checkpoint
```

---

## Mode A — Overlay (Development)

```
WAL base + LoRA overlay (runtime)
```

**When to use:** Development, A/B testing, reversible edits

**Pros:**
- Instant on/off — remove LoRA to revert
- No re-encode cost
- Multiple overlays possible (with interference caveats)

**Cons:**
- LoRA forward overhead (~1-5%)
- Cannot compose multiple overlays (M206: interference catastrophic)

**Code:**
```python
# Load encoded base
model = load_checkpoint("wal_base.pt")

# Inject LoRA
model = inject_lora(model, layers=[14,15,16], modules=["o_proj","q_proj","v_proj","gate_proj"])

# Train
# ...

# Save only LoRA weights (small)
torch.save({name: module.lora.state_dict() for name, module in model.named_modules() if hasattr(module, 'lora')}, "edit_lora.pt")

# Later: load base + LoRA
model = load_checkpoint("wal_base.pt")
model = inject_lora(model, ...)
# load lora weights
```

**Metrics:** PPL 4.35 (+0.08), survival 4-6/50

---

## Mode B — Fast Compile (Deployment)

```
WAL base → LoRA → merge → re-encode K=256
```

**When to use:** Production deployment, single artifact needed

**Pros:**
- Single checkpoint (no LoRA overhead)
- Fast encode (~3 min)
- Deterministic result

**Cons:**
- Not reversible (need backup base)
- +0.08 PPL cost

**Code:**
```python
# Load base
model = load_checkpoint("wal_base.pt")

# Inject + train LoRA
model = inject_lora(model, ...)
# ... train ...

# Merge
model = merge_lora(model)

# Re-encode
model = encode_model(model, K=256, iters=3)

# Save single checkpoint
torch.save(model.state_dict(), "wal_compiled.pt")
```

**Metrics:** PPL 4.35 (+0.08), survival 5/50, encode time ~3 min

---

## Mode C — Sequential Multi-Edit (Incremental)

```
Base_v0 → Edit_1 → merge → re-encode → Base_v1
Base_v1 → Edit_2 → merge → re-encode → Base_v2
...
```

**When to use:** Incremental factual editing, versioning, rollback

**Pros:**
- Each version is a standalone checkpoint
- Can rollback to any previous version
- Multi-task capability (5-8/50 survival)

**Cons:**
- Higher PPL cost (+0.19 per 2 groups)
- Recency bias — earlier edits fade

**Code:**
```python
def edit_and_compile(model, facts, version_name):
    model = inject_lora(model, ...)
    # ... train on facts ...
    model = merge_lora(model)
    model = encode_model(model, K=256, iters=3)
    torch.save(model.state_dict(), f"wal_v{version_name}.pt")
    return model

# Version 1
model_v1 = edit_and_compile(model_v0, facts_group_1, "1")

# Version 2
model_v2 = edit_and_compile(model_v1, facts_group_2, "2")

# Rollback
model_v1 = load_checkpoint("wal_v1.pt")
```

**Metrics:** PPL 4.47 (+0.19), survival 5.3/50 (2 groups), 7.7/50 (3 groups)

---

## Safety Checks

### 1. Spectral Norm Check

```python
def check_spectral_norm(model, threshold=1.0):
    max_sn = 0
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            A = module.lora.lora_A.float()
            B = module.lora.lora_B.float()
            delta = (A @ B).T
            _, s, _ = torch.linalg.svd(delta, full_matrices=False)
            max_sn = max(max_sn, s.max().item())
    return max_sn < threshold

# Before merge:
if not check_spectral_norm(model, threshold=1.0):
    print("WARNING: High spectral norm, edit may be unsafe")
```

**Thresholds:**
- < 1.0: Safe
- 1.0-2.0: Caution
- > 2.0: Risky

### 2. Learned Risk Model (M205b)

```python
import pickle
with open("experiments/m205b_model.pkl", "rb") as f:
    rf = pickle.load(f)

# Predict survival before training
features = [rank, wave_lambda, n_modules, n_layers, steps, lr,
            mean_spectral_norm, max_spectral_norm,
            mean_top10_energy, max_top10_energy, final_loss]
predicted_survival = rf.predict([features])[0]
print(f"Predicted survival: {predicted_survival:.1f}/50")
```

**Top predictors:** mean_top10_energy (0.37), steps (0.29), max_top10_energy (0.26)

---

## Choosing Edit Strength

| Goal | Steps | Expected Survival | ΔPPL | Time |
|------|-------|-------------------|------|------|
| Minimal edit | 50 | 3-5/50 | +0.02 | ~2 min |
| Weak edit | 100 | 4-6/50 | +0.08 | ~3 min |
| Medium edit | 200 | 8-15/50 | +0.40 | ~5 min |
| Strong edit | 400 | 18-21/50 | +0.64 | ~10 min |

---

## Known Limitations

1. **Multi-LoRA overlay doesn't work** (M206): Simultaneous LoRA interference is catastrophic. Use sequential editing instead.

2. **Recency bias in sequential editing** (M206c): Earlier groups fade. Group 1 survival: 1-3/16, Group 3: 2-4/18.

3. **Batch size vs steps tradeoff** (M207): Fixed steps=100, larger batches → lower per-fact survival. For large batches, increase steps proportionally.

4. **Not all facts are equal**: Some facts are "harder" to implant than others. Survival varies 3-21/50 depending on fact difficulty.

---

## Troubleshooting

### PPL too high after edit
- Reduce steps
- Reduce rank (try rank=2)
- Reduce target layers/modules

### Survival too low
- Increase steps (200-400)
- Increase rank (try rank=8)
- Ensure mixed training (facts + wikitext, 50/50)

### Merge fails / PPL explodes
- Check forward restoration (M200 bug)
- Ensure `module.forward = module._orig_forward` (not `orig`)
- Verify no double-LoRA

### Re-encode takes too long
- Use K=256 (~3 min) instead of K=1024 (~40 min)
- Enable batched Hadamard transform (75× speedup)

---

## File Reference

| File | Description |
|------|-------------|
| `experiments/m200_fixed_k256.py` | Correct merge+re-encode (K=256) |
| `experiments/m200b_merge_reencode_k1024.py` | Correct merge+re-encode (K=1024) |
| `experiments/m204b_steps400_merge_reencode.py` | Strong edit compiled mode |
| `experiments/m206b_sequential_multi_edit.py` | Sequential multi-edit |
| `experiments/m207_batch_concurrent_edits.py` | Batch size comparison |
| `experiments/m205b_stratified_risk_dataset.py` | Risk model data collection |

---

## Citation

```
WAL v4 Production Stack
- Base: Hadamard-WAL K=256 uniform
- Edit: LoRA rank=4, mixed training
- Modes: Overlay / Fast Compile / Sequential Multi
- Safety: Spectral norm + learned RF model
```
