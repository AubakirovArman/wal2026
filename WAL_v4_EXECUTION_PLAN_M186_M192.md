# WAL v4 Execution Plan — M186 to M192

**Date:** 2026-04-20  
**Phase:** Wave-Guided WAL  
**Based on:** M181/M182 conclusions

---

## Strategic Context

M181: Hadamard-WAL K=256 = near-lossless checkpoint format (PPL +0.01%).
M182: Diff locality = fundamental limit (99.8% diff even with K=256).

**New framework:** Waves guide encoding, risk, and training — but do NOT replace atoms.

---

## Track A — Wave Diagnostics

### M186 — Wave Depth Map
**Goal:** Find wave patterns across model depth.

For each layer/module, compute:
- norm(W), spectral_norm(W)
- atom_entropy, coeff_entropy, program_entropy
- DCT energy entropy
- top singular value, mean abs weight

Then FFT/DCT over layer index for each feature.

**Question:** Is there periodicity across layers? Does attention differ from MLP?

### M187 — Program-Wave
**Goal:** How does WAL language change across layers?

Build matrices:
- A[layer, atom_id] = frequency(atom_id in layer)
- C[layer, coeff_id] = frequency(coeff_id in layer)

Then FFT/DCT over layer dimension.

**Question:** Which atoms strengthen in early/middle/late layers? Is there coeff-wave?

### M188 — LoRA Delta Wave Risk
**Goal:** Understand why some edits are stable, others collapse.

For each ΔW = LoRA_B @ LoRA_A, compute:
- DCT/FFT/Hadamard spectrum
- top-1% energy, top-10% energy
- spectral entropy
- phase coherence
- spectral norm
- fingerprint drift

**Hypothesis:** Dangerous edits have concentrated waves — few frequencies carry too much energy.

**Output:** WaveRiskScore metric.

---

## Track B — Wave-Informed Encoding

### M189 — Phase Coherence Test
**Goal:** Is phase important, not just amplitude?

Destructive tests:
1. Keep amplitude, shuffle phase → inverse FFT → PPL
2. Keep phase, distort amplitude → inverse FFT → PPL

**Question:** Does phase structure matter for model quality?

### M190 — Wave-Guided WAL Budget
**Goal:** Allocate K/C per-layer based on wave risk.

Policy:
- low risk layer → K=128
- medium risk → K=256
- high risk → K=512

Features for risk scoring:
- spectral entropy
- top-frequency concentration
- DCT/Hadamard energy spread
- singular value profile
- program entropy

**Question:** Can we get M181 quality with smaller total size?

---

## Track C — Wave Training

### M191 — Wave-Regularized LoRA
**Goal:** Train LoRA with wave penalty.

Loss:
```
target_loss + λ * spectral_concentration_penalty
```

Penalty: prevent top-10% DCT coefficients from holding too much energy.

**Question:** Can wave regularization reduce PPL collapse for low-rank edits?

### M192 — Gumbel-WAL + Wave Regularization
**Goal:** Connect Gumbel-WAL with wave ideas.

Add to Gumbel-WAL training:
- program entropy regularizer
- spectral/wave regularizer
- layer-depth signal regularizer

**Question:** Less program collapse, better validation loss, more uniform atom usage?

---

## Execution Order

```
Phase 1 (immediate):
  M188 — LoRA Delta Wave Risk
  M190 — Wave-Guided WAL Budget
  M191 — Wave-Regularized LoRA

Phase 2 (next):
  M186 — Wave Depth Map
  M187 — Program-Wave
  M189 — Phase Coherence

Phase 3 (later):
  M192 — Gumbel-WAL + Wave Regularization
```

---

## What NOT to Do

1. Do NOT try WAL patch as LoRA replacement — M182 proved diff 99.8% = dead
2. Do NOT return to Wave-Atom ISA in old form — top-K DCT lost to scalar WAL
3. Do NOT do Graph-WAL — already gave bad signal
