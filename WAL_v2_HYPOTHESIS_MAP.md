# WAL v2 Research Hypothesis Map

**Date:** 2026-04-25  
**Status:** Living document — updated as experiments progress  
**Principle:** Negative results define boundaries, not tombstones.

---

## 1. Analogs and Related Work

### Closest Relatives

| Work | Mechanic | Philosophy | What WAL Can Borrow |
|------|----------|------------|---------------------|
| GPTVQ / VQ weights | codebook + assignment per weight/vector | compression-first | codebook construction, assignment algorithms |
| AQLM | multi-codebook quantization, additive representation | extreme compression (2-3 bits) | **additive/multi-codebook ISA** for WAL v2 |
| QuIP / QuIP# | incoherence processing + Hadamard + lattice codebooks | transform-before-quantize | **orthogonal transform before encode** |
| BitNet b1.58 | ternary {-1,0,1} weights via BitLinear | architecture constraint | WAL-aware pretraining / QAT idea |
| LoftQ / QLoRA | quantized base + LoRA init | practical fine-tuning | **WAL+LoRA overlay as first-class format** |
| Weight Patching / Diff Interpretation | weight diffs → behavior analysis | mechanistic interpretability | **model forensics via WAL statistics** |
| TurboQuant | PolarQuant + QJL + residual correction | KV-cache compression | transform + residual for WAL-KV |

### Honest Assessment

> **No full direct analog of WAL exists in public literature.**

Individual pieces exist (codebooks, VQ, multi-codebook, weight diffs), but the **combination** is unique:
```
weight = program
+ text assembly
+ binary format
+ VM/runtime
+ debugger
+ global atoms
+ canonicalization
+ decode → edit → re-encode workflow
+ program heatmaps
+ structured checkpoint framing
```

---

## 2. What WAL v1 Has Proven

| Claim | Evidence | Strength |
|-------|----------|----------|
| Weights can be stably represented as programs | M110-M116, M126 v4 | ✅ Strong |
| Decode → edit → re-encode pipeline works | M126 v4: 97.5% survival, +0.39 PPL | ✅ Strong |
| Global atoms work across all layers | M116: +0.03 PPL, 225× storage savings | ✅ Strong |
| Canonicalization makes encode deterministic | M129: 0% diff same seed | ✅ Strong |
| Cached inference = dense speed | M132: 0.97-1.02x | ✅ Strong |
| Atoms are basis directions, not semantic units | M121: entropy 0.966, no specialization | ✅ Strong |
| Program soup / averaging impossible | M117: PPL 6.4×10¹³ | ✅ Strong |
| Genetic evolution on programs fails | M122: GA 170M× worse than greedy | ✅ Strong |

---

## 3. What v1 Has NOT Proven (Conditional Negatives)

| Claim | Status | Condition |
|-------|--------|-----------|
| WAL patch via program diff | ❌ Negative | Scalar ISA, K=256, C=16, greedy encode, rebuild atoms, rank=4/8 |
| WAL-diff localizes edits | ❌ Negative | Same conditions + canonicalization |
| Semantic atoms | ❌ Negative | Post-hoc k-means, scalar ISA, no training pressure |
| WAL-native direct editing | ❌ Negative | Discrete IDs, table tuning breaks consistency |
| Vector atoms | ⚠️ Failed | Ternary coeffs insufficient; coefficient language was wrong |

**Key principle:** These are negative at the **tested configuration**, not universally.

---

## 4. Active Hypotheses (H1-H8)

### H1. WAL-diff noise comes from atom/coeff table rebuild ✅ CONFIRMED
**Test:** M133 (Phase A) — freeze atom table, rebuild only programs.  
**Result:** Non-target diff = **0.000%**, target diff = **25.000%**. Ratio = **25,000×**.  
**Status:** **ALIVE with caveat** — table shift was the sole cause of diffuse diff.  
**Patch size recalculation:** 72 MB (target only) vs 0.19 MB LoRA = 384×. Still larger than LoRA.  
**Next:** Optimize patch encoding (delta, RLE) or accept WAL+LoRA overlay as practical path.

### H2. Diff should be measured at distribution level, not weight level
**Idea:** Raw program diff is too noisy. Use statistics:  
- atom entropy shift per layer  
- residual density shift  
- atom transition matrix  
- KL(program_distribution_base ‖ edited)  
**Test:** E2 — compute fingerprint vectors for base vs edited models.  
**Success:** Fingerprint classifier distinguishes edit types.

### H3. WAL needs transform ISA (not raw scalar)
**Inspiration:** QuIP/QuIP# — space before quantization matters.  
**Idea:**
```
W' = Hadamard(W)  # or learned orthogonal transform
program = encode(W')
W = inverse_hadamard(decode(program))
```
**Test:** E3 — encode after Hadamard, measure PPL and diff locality.  
**Success:** Lower boundary noise, more localized diff.

### H4. Vector atoms are alive — coefficient mechanism was wrong
**Idea:** Previous failure was `vector atom + ternary coeff`. New variants:
```
vector atom + learned scalar coeff
vector atom + additive residual codebook
vector atom + low-rank local basis
vector atom + block-wise orthogonal transform
vector atom + AQLM-style multi-codebook
```
**Test:** E4 — prototype one variant, measure PPL vs scalar WAL.  
**Success:** PPL competitive or better with richer expressiveness.

### H5. WAL-native editing via soft program relaxation
**Idea:** Gumbel-softmax / straight-through on program IDs:
```
atom_logits = f(weight_context)
atom_id = softmax(atom_logits / temperature)
coeff_id = softmax(coeff_logits / temperature)
forward = Σ_i atom[i] × coeff[i] × prob[i]
anneal temperature → hard IDs
```
**Test:** E5 — microbenchmark on single layer.  
**Success:** Differentiable editing in program space converges.

### H6. WAL+LoRA overlay (not patch) ✅ CONFIRMED
**Idea:** Base in WAL, edits as LoRA overlays on cached decoded weights.  
**Test:** M135 (Phase C) — LoRA on WALCachedLinear, train, merge, evaluate.  
**Result:** Trained LoRA on WAL layers → PPL 16.20, Acc 10/10. Merge + re-encode works.  
**Status:** **ALIVE and practical**. This is the recommended production workflow.  
**Next:** Optimize merge path, benchmark multi-LoRA overlays.

### H7. Semantic atoms via training with WAL constraint
**Idea:** Post-hoc k-means → basis directions. Training pressure → semantics:
```
Train small transformer with WALLinear / soft-WALLinear
Regularizers:
  - encourage stable atom usage
  - encourage sparse program transitions
  - encourage module-specific program grammar
```
**Test:** E7 — train GPT-2 small with WAL constraint.  
**Success:** Emergent atom specialization by layer/function.

### H8. WAL as decompiler / forensics tool
**Idea:** Even if patching fails, WAL enables model understanding:
```
dense model → WAL
fine-tuned model → WAL
statistics → behavioral classifier
```
**Test:** E8 — fingerprint 10+ models (base, instruct, code, math, medical).  
**Success:** Classifier accuracy >80% on model type from WAL stats alone.

---

## 5. Kill Criteria vs Alive Criteria

### For Each Hypothesis

| Hypothesis | Kill Criteria (give up) | Alive Criteria (pursue) |
|------------|------------------------|------------------------|
| H1 | Non-target diff >5% with frozen table | Non-target diff <1% |
| H2 | Fingerprints don't cluster by edit type | Fingerprints distinguish edits with >80% accuracy |
| H3 | Transform adds >0.5 PPL or no diff improvement | PPL neutral + diff localization improves |
| H4 | All vector variants worse than scalar by >0.5 PPL | Any variant matches or beats scalar |
| H5 | Soft relaxation doesn't converge or >10× slower | Converges, <2× slower than hard IDs |
| H6 | Overlay overhead >10% or breaks cache | Overhead <5%, preserves cached speed |
| H7 | No emergent specialization after 10B tokens | Specialization visible in heatmaps |
| H8 | Classifier <60% on held-out models | Classifier >80% on held-out models |

---

## 6. Experiment Queue

| Priority | ID | Name | Blocks On | Est. Time |
|----------|-----|------|-----------|-----------|
| P0 | E1 | Fixed atom table diff | — | 30 min |
| P1 | E2 | Distribution fingerprints | E1 | 2 hours |
| P2 | E6 | WAL+LoRA overlay | — | 4 hours |
| P3 | E3 | Transform before encode | E1 | 4 hours |
| P4 | E4 | Vector atom variants | — | 8 hours |
| P5 | E5 | Soft program relaxation | E4 | 8 hours |
| P6 | E7 | WAL-constrained training | E4-E5 | 2 days |
| P7 | E8 | Model forensics | E2 | 1 day |

---

## 7. What to Never Say Again

```text
❌ "WAL patch is impossible"
✅ "WAL patch via scalar program diff after full re-encode with rebuild global atoms
    does not work under current conditions"

❌ "Semantic atoms are impossible"
✅ "Post-hoc k-means scalar atoms do not show semantic specialization"

❌ "WAL-native editing is impossible"
✅ "Direct discrete program tuning breaks consistency; differentiable or constrained
    approaches remain untested"

❌ "This is a dead end"
✅ "This path is bounded by the current configuration; changing conditions may
    revive the hypothesis"
```

---

## 8. Core Insight

> **WAL is not an engineering task with a known answer. It is the invention of a new layer between dense weights and quantized checkpoints.**
>
> We are not looking for confirmation of an old idea. We are looking for the right space in which "weight = program" becomes more useful than dense weights.

---

## Phase D — 12-bit Production Packing ✅ DONE
**Result:** 13.98 GB → 10.48 GB (25% reduction). Throughput 9.6B weights/sec. Perfect correctness.  
**Status:** Production-ready. Default WAL storage format.

## Phase E — Frozen Atom Library per Model Family 📋 PENDING
**Goal:** Pre-computed atoms for Llama-3.x, Qwen, Gemma.  
**Test:** PPL delta vs from-scratch, encode speed, cross-model consistency.

## Phase F — Semantic Fingerprints ✅ POSITIVE
**Result:** Fingerprints distinguish base/seed/noise/K128 variants. K128 shows massive Δ (entropy ↓1.02, atoms_used ↓50%).
**Status:** Concept validated. Next: test on real fine-tuned models (instruct, code, medical).

## Phase G — Re-Encode Loss Characterization 📋 PENDING
**Goal:** Systematic study of what survives re-encode.  
**Test:** Rank/layers/steps/K/C sweep.

## Phase H — Behavioral Editing at Scale 📋 PENDING
**Goal:** Proper style/behavior editing with 1000+ examples + KL-reg.  
**Test:** Rank sweep, early stopping, WAL re-encode survival.

---

*Document status: v1.1 — updated after M133-M136*  
*Next update: after next phase completion*
