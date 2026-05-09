# Execution Plan: M193–M198

## Приоритет 1: M193 — Real LoRA Wave Risk Calibration

**Goal:** Проверить, отличает ли WaveRiskScore реальные хорошие edits от плохих.

**Method:**
- Llama-3.1-8B, target layers [14,15,16] o_proj
- Contrafactual dataset (10 facts)
- Train 4 LoRA configs:
  - rank=1 steps=200 (collapse candidate)
  - rank=4 steps=50 (safe)
  - rank=4 steps=100 (balanced)
  - rank=8 steps=200 (marginal)
- Metrics per config:
  - spectral norm
  - WaveRiskScore
  - top10 energy, spectral entropy
  - fingerprint drift
  - PPL delta
  - target survival

**Success criteria:**
- SAFE configs имеют low risk scores
- COLLAPSE configs имеют high risk scores
- WaveRiskScore коррелирует с PPL/survival

---

## Приоритет 2: M196 — Wave-Regularized Real LoRA Training

**Goal:** Проверить, спасает ли wave-regularization rank=1 от collapse.

**Method:**
- Обычная LoRA
- LoRA + λ=0.05 wave penalty
- LoRA + λ=0.1 wave penalty
- LoRA + λ=0.2 wave penalty
- Rank=1 steps=200, rank=2 steps=200, rank=4 steps=100
- Metrics: survival, PPL, spectral norm, WaveRiskScore, re-encode loss

**Success criteria:**
- Wave-reg улучшает survival на rank=1
- Spectral norm уменьшается с λ
- PPL не деградирует

---

## Приоритет 3: M195 — Hadamard Wave-Guided Budget v2

**Goal:** Уменьшить размер WAL checkpoint без потери PPL.

**Method:**
- Hadamard-WAL (не raw-WAL)
- Percentile policy:
  - bottom 30% risk → K=128
  - middle 50% risk → K=256
  - top 20% risk → K=512
- Metrics: PPL, checkpoint size, K distribution

**Success criteria:**
- PPL ≤ uniform K=256 + 0.1
- Размер < uniform K=256

---

## M194 — Module-Specific Safety Thresholds

**Goal:** Per-module-type thresholds вместо общих.

```python
thresholds = {
    "gate_proj": {"safe": 100, "danger": 500},
    "q_proj":    {"safe": 60,  "danger": 300},
    "v_proj":    {"safe": 30,  "danger": 150},
}
```

Validate on M193 data.

---

## M197 — Phase Coherence PPL Test

**Goal:** Понять, что важнее для модели: amplitude или phase.

**Method:**
- FFT(W) → shuffle phase → IFFT → replace layer → PPL
- FFT(W) → distort amplitude → IFFT → replace layer → PPL

---

## M198 — Depth-Wave Budget by Module Type

**Goal:** Использовать реальные depth-wave закономерности M186 для budget allocation.

**Policy:**
```
early q/k → higher K
late v/o → higher K
late gate/up/down → higher K
stable modules → lower K
```

Compare with uniform K=256.
