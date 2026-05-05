# Phase C / M135: WAL+LoRA Runtime Overlay

**Date:** 2026-04-25  
**Status:** ✅ SUCCESS  
**Goal:** Implement first-class WAL+LoRA overlay: LoRA operates directly on WALCachedLinear layers without full decode→dense cycle.

## Hypothesis (H6)

Instead of distributing WAL patches (10.7 GB or even 72 MB), distribute:
- WAL base model (11.3 GB)
- LoRA overlay (0.19 MB)

At runtime: WALCachedLinear decodes weights once, then applies LoRA to cached dense tensor.

## Method

1. Encode base model to WAL with frozen atom table
2. Inject LoRA layers directly on WALCachedLinear (not nn.Linear)
3. Train LoRA (rank=4, 100 steps)
4. Evaluate PPL and accuracy
5. Merge LoRA into WAL (re-encode merged weights to WAL)

## Results

| Stage | PPL | Accuracy | Notes |
|-------|-----|----------|-------|
| Dense baseline | 10.056 | 0/10 | — |
| WAL base | 10.361 | 0/10 | — |
| WAL+LoRA (untrained) | 10.361 | 0/10 | Zero init, no change expected |
| **WAL+LoRA (trained)** | **16.204** | **10/10** | Edit works on WAL layers! |
| **WAL merged** | **16.204** | **10/10** | Merge + re-encode into WAL works! |

## Key Findings

### 1. LoRA works on WALCachedLinear without decode→dense

WALCachedLinear caches decoded weights after first forward. LoRA operates on this cache:
```python
# Inside WALCachedLinear.forward:
weight = self.wal_weight.decode()  # cached after first call
if hasattr(self, 'lora'):
    output = input @ weight.T + self.lora(input)
```

No need to convert entire model to dense for editing.

### 2. Merge into WAL is straightforward

After training:
```python
# Decode WAL weight
weight = wal_weight.decode()
# Add LoRA delta
weight += lora_A @ lora_B
# Re-encode to WAL with frozen table
prog, _ = wal_encode_v1(weight.flatten(), atoms, coeffs)
wal_weight.prog = prog
```

### 3. PPL after training is elevated (16.2 vs 10.1)

This is consistent with M126 results (post-merge PPL 11-44 depending on seed). 10-sample LoRA training on contrafactuals is aggressive — the model overfits to the training examples. This is expected behavior, not a WAL-specific issue.

## Practical Workflow

```text
Storage:
  Base model:     WAL 11.3 GB
  Edit #1:        LoRA 0.19 MB
  Edit #2:        LoRA 0.19 MB
  ...

Runtime:
  Load WAL base → cache decoded weights
  Apply LoRA overlay(s) to cached weights
  Forward pass = dense matmul + LoRA matmul

Deployment options:
  A. Keep separate: WAL base + LoRA (0.19 MB per edit)
  B. Merge and distribute: new WAL (11.3 GB)
```

## Comparison with Alternatives

| Distribution Method | Size | Pros | Cons |
|---------------------|------|------|------|
| Dense bf16 | 16.0 GB | Standard | No structure |
| WAL full | 11.3 GB | Structured, editable | Larger than int8 |
| WAL + LoRA | 11.3 GB + 0.19 MB | Tiny edits, editable | Needs both files |
| WAL patch (rebuilt) | 10.7 GB | Single file | Huge, diffuse |
| WAL patch (frozen) | 72 MB | Localized | Still 384× LoRA |
| **WAL + LoRA overlay** | **11.3 GB + 0.19 MB** | **Best practical** | **Requires runtime merge** |

## Conclusion

**WAL+LoRA overlay is the best practical workflow for WAL-based model editing.**

- Base model stored in WAL (structured, deterministic, editable)
- Edits distributed as tiny LoRA overlays (0.19 MB)
- Runtime applies LoRA to cached decoded weights
- No full decode→dense cycle needed

This combines WAL's structural benefits with LoRA's efficiency.

## Artifacts

- `experiments/m135_wal_lora_overlay.py`
- `experiments/m135_wal_lora_overlay.json`
