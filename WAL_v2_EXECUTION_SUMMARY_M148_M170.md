# WAL v2 Execution Summary — M148 to M170

**Date:** 2026-04-20  
**Model:** meta-llama/Llama-3.1-8B (8.03B params)  
**Status:** 20/22 experiments complete (M164, M165 blocked)

---

## Completion Matrix

| Milestone | Status | Key Finding |
|-----------|--------|-------------|
| M148 — Spec Freeze | ✅ | WAL v1 spec frozen, 6 compatibility tests pass |
| M149 — Frozen Vocab PPL | ✅ | Frozen table MSE 1.5× worse, diff 0.819 vs 0.855 |
| M150 — LoRA Patch Compression | ✅ | WAL patch 15–104× larger than LoRA |
| M151 — Multi-LoRA Routing | ✅ | Zero interference, linear additivity confirmed |
| M152 — Safety Score | ✅ | Perfect monotonicity: 0.5→SAFE, 12→DANGEROUS |
| M153 — Transform-WAL Encoder | ✅ | RandOrth 2–8× better MSE than Raw |
| M154 — Fix Hadamard | ✅ | Orthonormal, exact inverse, energy preservation |
| M155 — Partial PPL Gate | ✅ | K=64 too coarse: N=0 Δ+0.7%, N=31 Δ+71% |
| M156 — Transform-WAL Diff | ✅ | Transform does NOT improve patch locality |
| M157 — Transform Vocab | ✅ | Per-transform atoms 10–50× better than reuse |
| M158 — Transform Selection | ✅ | Single transform avg 1.12×, module-specific unstable |
| M159 — Transform Metadata | ✅ | Hadamard=0MB, RandOrth seed=0MB, full Q=98GB |
| M160 — Spectral Energy Map | ✅ | Uniform energy (~0.25/quadrant) across all layers |
| M161 — Spectral Delta LoRA | ✅ | rank=1 is 38.7% sparse vs rank=8 27.9% |
| M162 — Fingerprint Benchmark | ✅ | Noise detectable in attention (0.10), weak in MLP (0.01) |
| M163 — Fingerprint Drift | ✅ | Non-linear drift: scale<0.01 invisible, scale>0.1 clear |
| M164 — Cross-Model Vocab | ⏸️ | Blocked: need 70B/Qwen checkpoints |
| M165 — Cross-Architecture | ⏸️ | Blocked: need GPT-2/Mistral checkpoints |
| M166 — Soft-WALLinear | ✅ | WAL weights trainable, loss comparable to dense |
| M167 — STE/Gumbel | ✅ | **Breakthrough**: Gumbel-Softmax + STE enables program learning |
| M168 — Benchmark Suite | ✅ | Unified JSON schema for all experiments |
| M169 — Ablation Dashboard | ✅ | 23 experiment files aggregated |

---

## Top-Level Conclusions

### 1. Transform-WAL is Research-Only
- Improves MSE 2–8× over Raw-WAL
- Does NOT solve diff locality (still ~90% uniform)
- K=64 is too coarse for production (PPL +71% for full model)
- **Verdict:** Secondary research track, not production path

### 2. WAL + LoRA Overlay is Production Path
- Base: WAL-encoded model (11.3 GB)
- Edit: LoRA overlay (0.19 MB)
- Runtime: decode WAL → cache + LoRA forward
- Safety: spectral norm score validated on real deltas
- **Verdict:** Only viable production workflow

### 3. Gumbel-Softmax Enables Native WAL Training
- M167 proves differentiable program space works
- STE through Gumbel-Softmax allows gradient flow to atom_ids
- Opens path to training models directly in WAL space
- **Verdict:** WAL v2 training architecture viable

### 4. Fingerprints are Secondary Safety Signals
- Detect large perturbations in attention layers
- Weak for MLP layers and small perturbations
- Best used as ensemble with spectral norm + PPL gate
- **Verdict:** Useful augmentation, not primary guardrail

### 5. Spectral Analysis Explains LoRA Behavior
- LLM weights have uniform spectral energy (no bias)
- LoRA deltas are spectrally distributed (not sparse)
- Higher rank = more distributed = safer (explains M138)
- **Verdict:** Spectral norm is theoretically grounded safety metric

---

## Blocked Experiments

| Milestone | Blocker | Resolution Path |
|-----------|---------|-----------------|
| M164 — Cross-Model Vocab | Need 70B, Qwen, Gemma checkpoints | Download or request access |
| M165 — Cross-Architecture | Need GPT-2, Mistral, non-Llama | Use HuggingFace public models |

Both can be unblocked with public HF models (gpt2, EleutherAI/pythia, etc.).

---

## Next Steps

1. **M170 — WAL v2 Spec Draft:** Synthesize all findings into v2 specification
2. **Unblock M164/M165:** Run with gpt2 / pythia-70m as cross-model negative controls
3. **M167 Extension:** Scale Gumbel-WAL to real model size (memory engineering)
4. **Production Hardening:** Integrate safety score + fingerprint + PPL gate into single pipeline
