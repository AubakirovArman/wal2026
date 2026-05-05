# Phase 29 / M131: Edit Compilation

**Date:** 2025-04-24  
**Status:** ✅ Completed (negative result)  
**Goal:** Compile a LoRA edit into a WAL patch and compare patch size vs LoRA weights.

## Hypothesis

If WAL-diff shows localized changes after an edit, we could distribute edits as sparse patches: only the changed (atom_id, coeff_id) pairs. This would be much smaller than LoRA weights.

## Method

1. Encode base model → WAL_base
2. Decode to dense
3. Apply LoRA edit (rank=4, 100 steps) on layers 14-16 o_proj
4. Encode edited model → WAL_edited (same seed)
5. Compute patch = all positions where program changed
6. Compare patch size vs LoRA size

Patch format per entry: (position: 4 bytes, atom_id: 1 byte, coeff_id: 1 byte) = 6 bytes.

## Results

| Metric | Value |
|--------|-------|
| Total weights in model | 7,504,658,432 |
| Changed program entries | 1,876,164,608 |
| Change percentage | **25.00%** |
| LoRA size (fp16) | **0.19 MB** |
| WAL patch size (packed) | **10.7 GB** |
| Patch vs LoRA | **0.000018×** |

**WAL patch is 57,000× larger than LoRA weights.**

## Why It Failed

1. **Re-encode quantization error (M130):** Even with canonicalization and same seed, ~25% of weights get different (atom, coeff) pairs after re-encode.

2. **Diffuse changes:** The 25% diff is uniform across ALL layers, not just the 3 target layers. Every layer shows exactly the same change percentage.

3. **Patch size explodes:** 1.87 billion changed entries × 6 bytes = 10.7 GB. Compare to LoRA: 3 layers × (4096×4 + 4×4096) params × 2 bytes = 0.19 MB.

## Comparison

| Distribution Format | Size | Relative |
|---------------------|------|----------|
| Dense model (bf16) | 16.0 GB | 1.0× |
| WAL full model | 11.3 GB | 0.7× |
| LoRA weights | 0.19 MB | 0.0012× |
| WAL patch (diff) | 10.7 GB | 0.67× |

## Conclusion

**WAL patch compilation via program diff is completely infeasible.**

The quantization noise from re-encode dominates any real edit signal. A "patch" would be nearly as large as the full model.

**What works instead:**
- **LoRA:** 0.19 MB — the gold standard for edit distribution
- **Full WAL model:** 11.3 GB — for storage/distribution of complete models
- **WAL + LoRA:** Store base in WAL, apply LoRA at runtime (0.19 MB overlay)

WAL's value is not in patch diffing — it's in structured weight representation, interpretability, and the edit→merge→re-encode pipeline.

## Artifacts

- `experiments/m131_edit_compilation.py`
- `experiments/m131_edit_compilation.json`
