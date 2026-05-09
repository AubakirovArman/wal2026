# WAL v2 Execution Plan: M148–M170

**Date:** 2026-04-20
**Status:** Strategic planning document

---

## Core Hypothesis (H-WALv2)

> Raw weights are not the natural coordinate system for weight tokenization.
> A stable transform space can reduce quantization error, improve re-encode stability,
> and make program diffs smaller and more meaningful.

```
WAL v1:  tokenize raw weights
WAL v2:  transform weights → tokenize transformed coefficients
```

---

## What M142–M147 Taught Us

| Experiment | Result | Interpretation |
|-----------|--------|----------------|
| M142 Transform-WAL | **RandOrth 28× better MSE** | Transform before scalar WAL works |
| M143 Wave-Atom ISA | **65–584× worse than scalar** | Waves as replacement ISA fail |
| M144 Graph-WAL | **7–110× worse than raw** | Data-dependent graph basis amplifies errors |
| M145 Fingerprints | **14/28 separable, 0% k-NN** | Need real fine-tuned models |
| M146 Cross-Model Vocab | **Distributions identical (std ~0.0133)** | Shared vocabulary viable |
| M147 WAL-Friendly Training | **WAL reg +1.0% vs L2 +2.4%** | Simple regularizer underperforms |

**Key insight:**
```
❌ Bad path: replace scalar atoms with waves/graphs
✅ Good path: apply transform BEFORE scalar WAL
```

---

## Three Development Lines

### Line A — WAL v1 Production Core (Stable)
```
canonicalization
frozen vocabulary
12-bit packing
WAL+LoRA overlay
patch format v2
safety score draft
```
Goal: stable usable framework.

### Line B — WAL v2 Transform Core (Main Research)
```
W → Transform(W) → WAL encode → WAL decode → inverse Transform → W_recon
```
Goal: prove transform-space improves PPL, re-encode stability, patch size, diff locality.

### Line C — WAL Intelligence / Forensics / Training (Long-term)
```
fingerprints
cross-model vocab
WAL-friendly training
model diagnostics
```
Goal: analyze model state, fine-tune type, degradation, overfit, safety/domain shift.

---

## Success Criteria for WAL v2

Transform-WAL must improve at least **2 of 5**:
1. PPL better than or equal to Raw-WAL
2. Patch size smaller than Raw-WAL
3. Re-encode ΔPPL lower
4. Program diff more local
5. Fingerprints better separate models

**Big success:** 3 of 5.
**Breakthrough:** PPL ≤ Raw-WAL AND patch size -30%+ AND re-encode loss lower.

---

## M148–M170 Detailed Plan

### Track A — Production Core

#### M148 — WAL v1 Consolidation / Spec Freeze
- Gather all stable components into official WAL v1 spec
- Output: `WAL_v1_spec.md`, binary format spec, compatibility tests

#### M149 — Frozen Vocabulary PPL Matrix
- Compare: Raw-WAL rebuilt vs frozen, Transform-WAL rebuilt vs frozen
- Metrics: PPL, MSE, program entropy, patch locality, encode time

#### M150 — Real LoRA Patch Compression
- Real LoRA edits (rank=4/8, steps=50/100)
- Test: raw, RLE, bitmask, block patch, transition table, coeff-only, atom-only
- Target: real LoRA WAL patch < 20 MB

#### M151 — Multi-LoRA Routing / Conflict Test
- Trained overlays: factual, style, refusal/safety, domain
- Test combinations: A, B, A+B, A+C, A+B+C, conflicting A vs anti-A
- Metrics: target accuracy, cross-interference, PPL, latency, memory

#### M152 — Safety Score on Real LoRA
- Validate M141 on real LoRA deltas from M138
- Metrics: spectral norm, Frobenius, max/mean abs, boundary crossing, ΔPPL, survival, KL
- Goal: SafetyScore predicts ΔPPL before full evaluation

### Track B — Transform-WAL Core

#### M153 — Real Transform-WAL Encoder
- Full pipeline: W → Transform → WAL encode → WAL decode → inverse Transform
- Compare: Raw-WAL, RandOrth-WAL, FFT2-WAL, DCT2-WAL, Hadamard-WAL
- Metrics: MSE, PPL on replaced layer, program entropy, encode/decode time, memory

#### M154 — Fix Hadamard Properly
- Orthonormal variant: H_norm = H / sqrt(n)
- Test power-of-2 padding, inverse exactness, energy preservation
- Hadamard may be best production transform (real-valued, fast)

#### M155 — Partial Model Transform-WAL PPL Gate
- Gradual rollout: 1 layer → all q_proj → all k_proj → all v_proj → all o_proj → all attention → all MLP → full model
- Criterion: Transform-WAL PPL ≤ Raw-WAL PPL

#### M156 — Transform-WAL Diff Locality
- Compare: Raw-WAL frozen + LoRA vs Transform-WAL frozen + same LoRA
- Metrics: target diff %, non-target diff %, patch size, RLE/bitmask compression, transition entropy, boundary crossing
- Goal: Transform-WAL patch < Raw-WAL patch

#### M157 — Transform Vocabulary Study
- Options: A. raw global atoms in transform space, B. transform-specific global atoms, C. per-module transform atoms, D. per-transform per-module atoms, E. frozen family-level transform vocab
- Compare: PPL, patch locality, entropy, size, encode time
- Bet: transform-specific global atoms best compromise.

#### M158 — Transform Selection per Module
- M142 showed best transform varies by layer: q_proj→RandOrth, k_proj→FFT2, v_proj→DCT2, o_proj→FFT2, gate_proj→RandOrth
- Test: single transform for all vs module-specific vs layer-specific
- Criterion: module-specific improves PPL/patch without too much metadata

#### M159 — Transform Metadata Cost
- Options: store full Q matrix, generate Q from seed, shared Q across layers, shared Q per module type, structured random orthogonal, Hadamard/DCT no storage
- Goal: metadata overhead negligible

### Track C — Spectral / Wave Theory

#### M160 — Spectral Energy Map
- DCT/FFT energy distribution per layer/module
- Low/mid/high frequency ratio, spectral entropy, decay slope
- Compare: base, LoRA-edited, instruct, code/math/domain models
- Goal: find "frequency fingerprint" of layers and fine-tunes

#### M161 — Spectral Delta of LoRA
- Decompose ΔW = B @ A via FFT/DCT/RandOrth
- Check: is ΔW spectrally sparse? Which bands change? Does rank=1 collapse have distinct spectral signature? Does rank=4 stable edit have smoother spectrum?
- May explain M138: why rank=1/200 collapses while rank=4 survives

### Track D — Fingerprints / Forensics

#### M162 — Real Model Fingerprints Benchmark
- Models: Llama base, Llama instruct, CodeLlama, math-tuned, medical/domain-tuned, safety-tuned, merged-LoRA, overfit-LoRA, quantized, noisy
- Features: WAL entropy, coeff entropy, program transition entropy, residual density, spectral fingerprints, singular value stats, patch stats
- Classifiers: kNN, logistic regression, random forest, small MLP
- Target: model type classification >80%

#### M163 — Fingerprint Drift During Training
- Log fingerprints during LoRA training: step 0, 10, 25, 50, 100, 200
- Correlate with PPL, target accuracy, SafetyScore, ΔW norm
- Goal: find early signal of overfit/collapse before PPL rises

### Track E — Cross-Model / Cross-Family Vocabulary

#### M164 — Real Cross-Model Vocabulary
- Build vocab on Llama-3.1-8B base, encode Llama-3.1-8B-Instruct, other fine-tunes, 70B sample, Qwen/Gemma as negative controls
- Metrics: PPL, MSE, atoms used, entropy, patch locality
- Goal: one vocabulary per model family

#### M165 — Cross-Architecture Negative Control
- Test boundaries: Llama vocab → Llama fine-tunes (works?), Llama → Qwen (fails?), Llama → Gemma (fails?), Qwen → Qwen, Gemma → Gemma
- If intra-family works and cross-family fails → each family has its own "weight tokenizer"

### Track F — WAL-Friendly Training

#### M166 — Soft-WALLinear Small Model
- Soft-WALLinear: atom logits, coeff logits, soft assignment, anneal temperature, harden to discrete ids
- Start small: 10M → 50M → 100M
- Goal: can model train while staying WAL-friendly?

#### M167 — STE / Gumbel Program IDs
- Gumbel-softmax atom_id, Gumbel-softmax coeff_id, straight-through hard ids, temperature schedule
- Metrics: training loss, validation PPL, program entropy, final WAL encode loss

### Track G — Evaluation Discipline

#### M168 — Standard WAL Benchmark Suite
- Unified benchmark: WikiText PPL, C4 small PPL, target survival, preserve QA, KL to base, latency, memory, patch size, program diff locality, fingerprints
- Unified JSON output format

#### M169 — WAL Ablation Dashboard
- Table of all modes (Raw-WAL, Raw-WAL frozen, Raw-WAL rebuilt, RandOrth-WAL, DCT-WAL, FFT-WAL, Hadamard-WAL, WAL+LoRA, WAL patch) vs metrics (PPL, size, latency, editability, patch size, stability, forensics)

---

## Top 5 Next Experiments (Priority Order)

1. **M153 — Real Transform-WAL Encoder** (main research check)
2. **M154 — Fix Hadamard** (production transform candidate)
3. **M156 — Transform-WAL Diff Locality** (does patch shrink?)
4. **M152 — Safety Score on Real LoRA** (production guardrail)
5. **M150 — Real LoRA WAL Patch Compression** (compiled structural edit)

---

## Strategic Positioning

```
WAL v1:  proved weight tokenization
WAL v1.5: frozen vocabulary + LoRA overlay + patch v2 + 12-bit packing
WAL v2:   transform-domain weight tokenization
WAL v3:   WAL-friendly training / differentiable program space
```

> **The most important thing: do not chase too-fantastic ideas before verifying Transform-WAL on PPL.**

Wave-Atom and Graph-WAL gave useful negative results. But M142 gave a real positive signal. Therefore the next main thrust:

```
Transform-WAL: real encode → full/partial PPL → diff locality → patch size
```

If this confirms, WAL gets a new foundation: not just "weights as programs", but:

> **Weights as a signal that must first be transformed into the right space, then tokenized as programs.**
