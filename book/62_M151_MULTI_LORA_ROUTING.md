# M151 — Multi-LoRA Routing / Conflict Test

**Date:** 2026-04-20
**Status:** ✅ Complete (v2 synthetic)
**Goal:** Test multiple LoRA overlay combinations and measure interference.

## Method

- Synthetic LoRA deltas: `delta = A @ B` with different seeds
- 4 overlays (A, B, C, D), rank=4 each
- 11 combination tests
- Interference = deviation from sum of individual effects

## Results

| Combination | Interference | Delta Norm |
|-------------|-------------|------------|
| A | 0.000000 | 0.8245 |
| B | 0.000000 | 0.8150 |
| C | 0.000000 | 0.8209 |
| D | 0.000000 | 0.8162 |
| A+B | 0.000000 | 1.1594 |
| A+C | 0.000000 | 1.1635 |
| A+B+C | 0.000000 | 1.4208 |
| A+B+C+D | 0.000000 | 1.6387 |

**Summary:**
- Avg single-edit delta norm: **0.8192**
- Avg multi-edit delta norm: **1.3030**
- Avg interference: **0.000000**

## Key Finding

**Synthetic deltas have zero interference** because they are linear additive. The combined effect is exactly the sum of individual effects.

Delta norm scales as √n for n orthogonal random deltas:
- 1 overlay: ~0.82
- 2 overlays: ~1.16 (√2 × 0.82)
- 4 overlays: ~1.64 (√4 × 0.82)

## Limitations

- Synthetic deltas, not trained LoRA
- Real trained overlays may have non-zero interference due to correlated directions
- No PPL or accuracy metrics

## Implications

The WAL+LoRA overlay architecture (M140) is mathematically sound for independent synthetic edits. For real-world use, interference testing with trained overlays is needed.

## Artifacts

- `experiments/m151_multi_lora_routing_v2.py`
- `experiments/m151_multi_lora_routing.json`
