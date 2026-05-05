# M193 v1 — Real LoRA Wave Risk Calibration (Initial Attempt)

**Goal:** First attempt to validate WaveRiskScore on real trained LoRA.

## Method

- Llama-3.1-8B, target layers [14,15,16] o_proj
- Contrafactual dataset (10 facts)
- 4 LoRA configs trained:
  - rank=1 steps=200
  - rank=4 steps=50
  - rank=4 steps=100
  - rank=8 steps=200
- Metrics: survival, PPL, spectral norm, WaveRiskScore, top10 energy

## Results

| Config | Survival | PPL Δ | SpecNorm | WaveRisk |
|--------|----------|-------|----------|----------|
| rank1_steps200 | 1/10 | +1309 | 16.52 | 0.35 |
| rank4_steps50 | 1/10 | +1230 | 0.88 | -2.43 |
| rank4_steps100 | 0/10 | +3460 | 5.00 | -1.75 |
| rank8_steps200 | 0/10 | +1247 | 5.13 | -1.87 |

## Problems

### Catastrophic Forgetting

All configs show PPL increase of 1000×–3000×. Training on 10 contrafactual facts without general corpus regularization destroys model knowledge.

### WaveRiskScore Fails

The M188 formula (`top1*2 + top10*1 + sv_top1*2 + spec_norm*0.1 - entropy*0.2`) incorrectly ranks configs:
- rank1 (worst PPL, highest spec_norm=16.52): risk=0.35
- rank4_steps100 (catastrophic PPL +3460, spec_norm=5.00): risk=-1.75

The anomaly: rank4_steps100 has moderate spec_norm (5.00) but the worst PPL (+3460).

### Baseline PPL Artifact

Baseline PPL=1.17 due to overly short test text. Not a realistic metric.

## Lessons Learned

1. **Contrafactual-only training breaks models** — mixed training is essential
2. **Synthetic risk formulas don't transfer** — need real calibration
3. **PPL must use real corpus** (wikitext-2), not dummy text
4. **Spectral norm partially correlates** but is not sufficient alone

## Next Step

M193 v2 fixes all these issues with mixed training, wikitext-2 PPL, and AdamW regularization.
