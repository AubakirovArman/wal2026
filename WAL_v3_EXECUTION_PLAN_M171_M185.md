# WAL v3 Execution Plan — M171 to M185

**Date:** 2026-04-20  
**Based on:** Strategic analysis of M148–M170 results  
**Status:** Planning phase

---

## Strategic Context

After M148–M170, three viable tracks emerged:

```text
Track A — WAL Production Hardening    (M171–M174)
Track B — Gumbel-WAL Scale-Up         (M175–M179)
Track C — High-K Transform-WAL        (M180–M183)
Track D — Intelligence / Forensics    (M184–M185)
```

Priority: Track A first (production stability), then Track B (breakthrough validation), then Track C (transform quality).

---

## Track A — WAL Production Hardening

### M171 — Unified WAL Runtime Pipeline
**Goal:** Single API for WAL + LoRA workflow.

```python
model = WALModel.load("base.wal")
model.attach_lora("edit_1.safetensors")
model.enable_overlay("edit_1")
model.safety_check()  # spectral norm + fingerprint + PPL
model.merge_overlay("edit_1")
model.save("base_edit_1.wal")
```

**Success criteria:**
- 1 command loads WAL base + LoRA
- 1 command merges into new WAL
- PPL / safety / patch stats auto-logged

### M172 — Real Trained Multi-LoRA Interference
**Goal:** Test real trained overlays, not synthetic deltas.

Edits: factual, style, domain, safety, anti-factual.

Metrics: target accuracy, cross-task degradation, PPL, KL, interference score.

### M173 — Production Safety Stack
**Goal:** Integrate spectral norm + fingerprint drift + PPL gate into unified guardrail.

Validate on trained LoRA (not synthetic).

### M174 — WAL Patch Niche Test
**Goal:** Find where WAL patch is useful (not vs LoRA size, but vs compiled/audit properties).

Compare: LoRA overlay, WAL patch, full re-encoded WAL.

---

## Track B — Gumbel-WAL Scale-Up

### M175 — Gumbel-WAL 10M → 100M Scale Test
**Goal:** Does Gumbel-WAL scale beyond tiny models?

Models: 10M, 30M, 70M, 100M.
Variants: dense baseline, post-hoc WAL, Gumbel fixed atoms, Gumbel learned coeffs, Gumbel learned atoms+coeffs.

### M176 — Factorized Logits Memory Problem
**Goal:** Solve `[N, K*C]` memory explosion.

Options: top-k logits, factorized (atom_logits + coeff_logits), low-rank, blockwise, hash-based.

### M177 — Temperature / Curriculum Schedule
**Goal:** Find schedule where model doesn't collapse into few programs.

Sweep: temp_start, temp_end, schedule type.

### M178 — Learn Atoms + Programs Jointly
**Goal:** End-to-end learning of atoms + coeffs + programs.

Regularizers: atom diversity, entropy, usage balance, sparsity.

### M179 — Gumbel-WAL vs LoRA Adaptation
**Goal:** Can Gumbel-WAL replace LoRA for some edits?

Base frozen, learn program logits for target layers only.

---

## Track C — High-K Transform-WAL

### M180 — GPU High-K Transform-WAL
**Goal:** K=256/512/1024 on GPU.

Transforms: Raw, Hadamard, DCT, RandOrth(seed).

### M181 — Transform-WAL Full PPL Matrix
**Goal:** Per-layer-type PPL gate.

Replace: layer 0 only → all q_proj → all v_proj → all attention → all MLP → full model.

### M182 — Transform-WAL Editability Matrix
**Goal:** Does high-K transform preserve editability?

Real LoRA edits on Raw/Hadamard/DCT/RandOrth frozen vocab.

### M183 — Transform Selection Simplification
**Goal:** Confirm single transform rule on K=256/512.

---

## Track D — Intelligence / Forensics

### M184 — Real Fine-Tuned Fingerprint Benchmark
**Goal:** >80% classifier accuracy on real model variants.

Models: base, instruct, code, math, medical, safety, overfit-LoRA, merged-LoRA, quantized.

### M185 — Fingerprint During Real Training
**Goal:** Early overfit detection before PPL collapse.

Fingerprint snapshots every N steps during LoRA training.

---

## Execution Order

```text
Phase 1 (immediate):
  M171 — Unified WAL Runtime Pipeline
  M175 — Gumbel-WAL 10M/30M scale-up
  M176 — Factorized logits memory test

Phase 2 (next):
  M177 — Temperature schedule
  M180 — GPU High-K Transform-WAL
  M181 — High-K PPL gate

Phase 3 (later):
  M172 — Real Multi-LoRA interference
  M173 — Production safety stack
  M178 — Joint atom learning
  M182 — Transform editability
  M174 — WAL patch niche
  M183 — Transform simplification
  M184 — Real fingerprint benchmark
  M185 — Fingerprint during training
```

---

## Key Risks

1. **Gumbel-WAL doesn't scale** — M175 is make-or-break for Track B
2. **High-K still too coarse** — M180/M181 determine Track C viability
3. **Memory explosion** — M176 must solve before Track B is practical
4. **Scope creep** — stick to 3 tracks, defer everything else
