# M150 — Real LoRA Patch Compression

**Date:** 2026-04-20
**Status:** ✅ Complete (v2 synthetic deltas, CPU)
**Goal:** Measure WAL patch size for structured low-rank edits vs LoRA size.

## Method

- Model on CPU, 2 layers, 2 modules
- K=64, C=8, iters=1, frozen atom table
- Synthetic LoRA deltas: `delta = A @ B * scale`
- Configs: rank ∈ {1,4,8}, scale ∈ {0.01, 0.05}

## Results

| Rank | Scale | Patch MB | LoRA MB | **Ratio** | Avg Change |
|------|-------|----------|---------|-----------|------------|
| 1 | 0.01 | 5.56 | 0.053 | **104×** | 0.38% |
| 4 | 0.01 | 6.05 | 0.213 | **28×** | 0.96% |
| 8 | 0.01 | 6.42 | 0.426 | **15×** | 1.41% |
| 4 | 0.05 | 9.41 | 0.213 | **44×** | 4.97% |
| 8 | 0.05 | 11.33 | 0.426 | **27×** | 7.26% |

## Key Findings

1. **WAL patch is ALWAYS larger than LoRA** — minimum ratio 15× (rank=8, scale=0.01)
2. **Smaller rank = larger ratio** — rank=1 gives 104×, rank=8 gives 15×
3. **Larger scale = larger patch** — more aggressive edits cross more quantization boundaries
4. **Avg change is small** — even at scale=0.05, only 7.26% of weights change program

## Comparison to M139 (Random Perturbation)

| Edit Type | Patch Size |
|-----------|-----------|
| Random noise (M139) | 32.92 MB |
| LoRA rank=4, scale=0.01 | 6.05 MB |
| LoRA rank=8, scale=0.05 | 11.33 MB |
| LoRA itself | 0.21–0.43 MB |

**LoRA patch is 3–5× smaller than random patch**, but still **15–100× larger than LoRA itself**.

## Implications

- **LoRA remains the gold standard** for edit distribution (0.2 MB vs 6–11 MB WAL patch)
- **WAL patch viable only when:** LoRA cannot express the edit (non-low-rank structural change)
- **For typical fine-tuning:** Use WAL base + LoRA overlay, not WAL patch

## Artifacts

- `experiments/m150_real_lora_patch_compression_v2.py`
- `experiments/m150_real_lora_patch_compression.json`
