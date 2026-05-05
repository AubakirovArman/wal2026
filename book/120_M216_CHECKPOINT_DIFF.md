# M216: WAL Checkpoint Diff

**Status:** ✅ Complete
**Date:** 2026-04-30
**Model:** Llama-3.1-8B, 3 sequential edits

## Question

How much of a WAL checkpoint changes after compiled editing? Can we make efficient binary diffs?

## Results

| Edit | Changed Modules | Mean Rel Change | Changed Params | Diff Size |
|------|-----------------|-----------------|----------------|-----------|
| 1 | 224/224 (100%) | 7.50% | 69.1% | 27.6 GB |
| 2 | 224/224 (100%) | 5.44% | 61.8% | 24.7 GB |
| 3 | 224/224 (100%) | 4.82% | 56.5% | 22.6 GB |
| **Cumulative v0→v3** | **100%** | **10.82%** | **77.1%** | **30.8 GB** |

## Key Finding

**WAL re-encode makes diff NON-LOCAL!**

LoRA edit touches only 4 modules in 3 layers. But after merge + re-encode (Hadamard + K-means), changes spread to **ALL 224 modules** and **77% of parameters**.

This means:
- WAL binary patch between versions is NOT efficient (diff ~31GB ≈ 2× full checkpoint)
- Cannot do "small diff" between WAL versions
- Each version is essentially a full checkpoint

## Practical Implication

WAL versioning requires storing full checkpoints, not diffs. For Git-like workflow, need different approach — possibly store LoRA weights separately and apply at load time.
