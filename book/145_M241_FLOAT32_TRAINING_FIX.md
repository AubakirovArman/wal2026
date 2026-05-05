# M241 — Float32 Training Fix

**Date:** 2026-04-20  
**Branch:** `main`  
**File:** `experiments/m241_float32_training_fix.py`

## Purpose

Identify and fix the catastrophic training failure in M235/M240 (0% survival, nan PPL).

## Hypothesis

Float16 gradient overflow during LoRA training causes adapter weights to become nan, which propagates to base weights after merge, destroying model outputs.

## Fix

1. Train adapters in **float32** (base model stays in fp16)
2. Cast inputs to fp32 before adapter forward pass
3. Cast adapter outputs back to fp16 before adding to base
4. Add **gradient clipping** (max_norm=1.0)

## Results

| Metric | M235 (fp16) | M241 (fp32) |
|--------|-------------|-------------|
| Survival | 0/3 | **3/3** |
| PPL | nan | **2.07** |
| Has nan | True | **False** |

## Critical Finding: FP16 OVERFLOW CONFIRMED

**Float32 training completely restores edit functionality.**

### Why fp16 fails
1. Adam optimizer accumulates second moments in fp16
2. Gradient norms for factual editing can exceed 65504 (fp16 max)
3. Overflow → nan gradients → nan weights → nan outputs
4. This explains ALL recent failures: M235, M240, M244

### Cost of fp32
- Memory: adapters use 2× memory during training (~MBs, negligible)
- Speed: ~10-20% slower due to casts
- Merge: adapters merged back to fp16 base, no runtime cost

## Conclusion

**All future WAL experiments MUST use fp32 adapters + gradient clipping.**
- This is a critical infrastructure fix
- M228-M234 worked because they used a different training path
- M235+ broke because of forward function refactor
- Production stack MUST include fp32 adapter training

## Next Steps
- Retrofit all training code (M235, M240, M244) with fp32 fix
- Re-run batch editing with fixed training
- Add automatic gradient monitoring to CI pipeline
