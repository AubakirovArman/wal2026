# M171 — Unified WAL Runtime Pipeline

**Goal:** Create a single production API for WAL + LoRA workflow.

## API Design

```python
from wal.v1 import WALModel

# Load and encode
model = WALModel.from_dense("meta-llama/Llama-3.1-8B", K=256, C=16)

# Attach LoRA edit
model.attach_lora("factual_edit.safetensors")
model.enable_overlay("factual_edit")

# Safety check
report = model.safety_check()
# Returns: {"overall": "SAFE", "spectral_norm": {...}, "ppl_gate": {...}}

# Merge permanently
model.merge_overlay("factual_edit")

# Save new WAL checkpoint
model.save("base_factual.wal")
```

## Implementation

### `WALModel` Class

Located in `src/wal/v1/runtime.py`:

| Component | Purpose |
|-----------|---------|
| `from_dense()` | Load HF model → encode all Linear to WALCachedLinear |
| `load()` | Load WAL checkpoint (metadata + serialized programs) |
| `attach_lora()` | Load LoRA weights from safetensors/bin |
| `enable_overlay()` | Decode base → add LoRA → re-encode to WAL |
| `disable_overlay()` | Revert to base programs |
| `safety_check()` | Spectral norm (power iteration) + PPL gate |
| `merge_overlay()` | Permanently merge LoRA into base WAL |
| `save()` | Serialize WAL state + metadata |

### Safety Stack Integration

```python
def safety_check(self, overlay_name=None, tokenizer=None, max_length=128):
    # 1. Spectral norm via power iteration
    for lora_B in lora_weights:
        sigma = power_iteration(lora_B, steps=10)
    
    # 2. PPL gate
    ppl = compute_ppl(self.model, tokenizer, max_length)
    
    # 3. Overall assessment
    if max_sigma > 4.0 or ppl > 25:
        return "DANGEROUS"
    elif max_sigma > 1.0 or ppl > 15:
        return "MODERATE"
    else:
        return "SAFE"
```

## Test Results

Tiny synthetic model test:
- All 7 API methods executed successfully
- Safety check: SAFE (spectral norms 0.05–0.08)
- Forward pass with overlay: functional

## Status

**Production API skeleton complete.** Next steps:
1. Test on real Llama-3.1-8B with real LoRA
2. Add fingerprint drift to safety_check
3. Implement proper WAL serialization (currently placeholder)
4. Add rollback support for disable_overlay
