# WAL Production Stack v2

**Date:** 2026-04-20  
**Based on:** M181–M200 experiments

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Application Layer                      │
│  - Factual editing API                  │
│  - Safety monitoring                    │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│  Edit Layer: Wave-Regularized LoRA      │
│  - λ = 0.025 (mixed targets)            │
│  - Rank = 4 (survival 10/50)            │
│  - Target: layers 14-16, 4 modules      │
│  - Training: mixed wikitext + facts     │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│  Base Layer: Hadamard-WAL Adaptive K    │
│  - K = 128/256/512 by spectral norm    │
│  - iters = 5 k-means (chunked GPU)     │
│  - PPL overhead: +0.038 (0.66%)        │
│  - Encode time: ~20 min for 8B         │
└─────────────────────────────────────────┘
                   │
┌─────────────────────────────────────────┐
│  Runtime: WALCachedLinear + LoRA        │
│  - Base weights: WAL-encoded            │
│  - Overlay: LoRA deltas                 │
│  - NO merge + re-encode!                │
└─────────────────────────────────────────┘
```

---

## Key Findings by Experiment

### M181 — Near-Lossless Checkpoint
- Hadamard K=256: PPL 4.3173 vs baseline 4.3169 (+0.01%)
- **Checkpoint format is viable**

### M182 — WAL Patch Non-Locality
- ~99.8% program diff after LoRA edit
- **WAL patch fundamentally non-local**

### M186 — Wave Depth Map
- Period 16/32 waves by depth
- v_proj norms grow 2.09× late/early

### M187 — Program-Wave Test
- WAL programs have NO wave structure
- **Waves are weight-level, not program-level**

### M188 — LoRA Delta Wave Risk
- Scale dominates rank
- Module sensitivity: gate_proj > q_proj > v_proj

### M189 — Phase Coherence
- Amplitude features invariant to phase shuffle
- Spectral norm captures structural risk

### M190 — Raw-WAL Adaptive Budget
- Failed: PPL 6.02 vs uniform 4.71
- Risk formula too aggressive

### M191/192/193 — Wave-Regularized Training
- Post-hoc regularization too weak
- λ=0.1 sweet spot on tiny transformer
- λ=1.0 destabilizes

### M193 v2 — Real LoRA Wave Risk
- Mixed training prevents catastrophic forgetting
- Spectral norm > WaveRiskScore

### M195/195b/195b+ — Hadamard Adaptive K
- M195: adaptive +0.038 vs uniform +0.062
- M195b: k-means adaptive +0.067 vs uniform +0.138 (2× better)
- M195b+: iters=5: adaptive +0.038, gap vs uniform only +0.006
- **k-means iterations matter more than K choice**

### M196 — Wave-LoRA o_proj only
- λ=0.1 improves survival 0→2/10, no PPL loss

### M196b — Wave-LoRA mixed targets
- λ=0.1 HURTS survival on mixed (6→3, 10→5)
- Module-count sensitivity discovered

### M196c — Penalty Schedules
- Constant λ=0.1 most reliable
- Cosine decay alone: 0/10 survival

### M196d — λ Scaling
- λ=0.025, 0.05, 0.1 tested
- Variance dominates effect in single runs

### M196e — Variance Test (n=5)
- **λ=0.025 best: mean survival 4.80 vs 3.40 baseline (+41%)**
- λ=0.1 worst: 3.20
- Non-monotonic relationship
- Need n≥15–20 for p<0.05

### M198 — Depth-Wave Budget
- Uniform K=256: 12.4069 (Δ=-0.08)
- Risk Adaptive: 12.4927 (Δ=+0.004)
- Depth Adaptive: *(pending)*

### M200 — End-to-End Demo
- Encode: near-lossless (-0.0076 PPL)
- LoRA on encoded: works fine (-0.0973)
- **Merge + re-encode: CATASTROPHE (+6.20 PPL, +60%)**
- **Production path: base + overlay ONLY**

---

## Production Configuration

### Base Checkpoint
```python
method = "hadamard_wal_adaptive_k"
k_policy = "percentile"  # 128 (p25), 256 (p25-p75), 512 (p75)
k_means_iters = 5
chunk_size = 1_000_000  # for GPU k-means
exclude = ["embed_tokens", "lm_head"]  # too large for Hadamard
```

### LoRA Edit
```python
rank = 4
target_layers = [14, 15, 16]
target_modules = ["o_proj", "q_proj", "v_proj", "gate_proj"]
wave_lambda = 0.025  # per-module normalized
lr = 5e-5
steps = 100
training = "mixed"  # 50% wikitext-2 + 50% task data
```

### Runtime
```python
# NEVER merge + re-encode
# Use overlay at inference time
class WALCachedLinear(nn.Module):
    def __init__(self, base_weight, atoms, indices, lora_A, lora_B):
        self.base = decode_wal(atoms, indices)  # on-demand
        self.lora = LoRALayer(lora_A, lora_B)
    
    def forward(self, x):
        return x @ self.base.T + self.lora(x)
```

### Safety
```python
# Monitor spectral norm of LoRA deltas
# Alert if > threshold per module
# Auto-rollback if PPL degradation > 1%
```

---

## Open Questions

1. **Statistical confidence**: n=5 runs show λ=0.025 best, but need n=20 for p<0.05
2. **Size reduction**: K=128 modules save bits, but atom table overhead negates savings
3. **Learned risk model**: Need 100+ runs for XGBoost/RF training
4. **Higher ranks**: rank=8 may give better survival with acceptable PPL cost
5. **More layers**: target layers 10-20 may improve survival further
6. **Cross-model atoms**: Can we share atom tables across models?

---

## Files

| Experiment | File | Status |
|-----------|------|--------|
| M195b+ | `experiments/m195b_plus_hadamard_adaptive_kmeans.py` | ✅ |
| M196e | `experiments/m196e_wave_lora_variance_test.py` | ✅ |
| M198 | `experiments/m198_depth_wave_budget.py` | ⏳ |
| M200 | `experiments/m200_end_to_end_wal_v2.py` | ✅ |

## Book Chapters

- 82_M188 through 99_M200 in `book/`
