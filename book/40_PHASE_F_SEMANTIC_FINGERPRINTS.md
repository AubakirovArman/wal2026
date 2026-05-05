# Phase F / M137: Semantic Fingerprints via WAL Statistics

**Date:** 2026-04-25  
**Status:** ✅ Positive result  
**Goal:** Test if WAL statistical fingerprints can distinguish different model states.

## Hypothesis (H8)

Instead of looking for semantic atoms (which failed in M121), look at **statistical fingerprints** of WAL programs:
- atom entropy per layer
- coeff entropy per layer
- top-3 atom dominance
- residual density
- atoms used (% of 256)

Different model states (base, edited, different configs) should have distinguishable fingerprints.

## Method

Compute fingerprints for 4 variants:
1. **Base** (seed=42, K=256)
2. **Different seed** (seed=123, K=256)
3. **Dense + small noise + re-encode** (K=256)
4. **Different K** (K=128, seed=42)

## Results

| Variant | Atom Entropy | Coeff Entropy | Top-3 | Atoms Used |
|---------|-------------|---------------|-------|------------|
| base | 7.7528 | 3.9578 | 0.0300 | 0.9952 |
| seed123 | 7.7003 (Δ0.05) | 3.9481 (Δ0.01) | 0.0291 | 0.9989 |
| dense+noise | 7.8318 (Δ0.08) | 3.9779 (Δ0.02) | 0.0232 | 1.0000 |
| K128 | 6.7317 (Δ1.02) | 3.9275 (Δ0.03) | 0.0480 | 0.4999 |

## Analysis

### Different seed (Δ small)
- Atom entropy: 7.75 → 7.70 (Δ0.05)
- Atoms used: 99.5% → 99.9% (Δ0.4%)
- Different k-means samples converge to slightly different atom distributions

### Noise + re-encode (Δ moderate)
- Atom entropy: 7.75 → 7.83 (Δ0.08) — **increases**
- Top-3 dominance: 3.0% → 2.3% — **decreases**
- Noise spreads weights across more atoms, making distribution more uniform

### K=128 (Δ large)
- Atom entropy: 7.75 → 6.73 (Δ1.02) — **massive drop**
- Atoms used: 99.5% → 50.0% — **half unused**
- Top-3 dominance: 3.0% → 4.8% — **more concentrated**
- Fewer atoms → lower entropy ceiling, more concentration on used atoms

## Conclusion

**WAL fingerprints are sensitive to model state changes.**

The fingerprint differences are:
- Small for same-K, different-seed encodes
- Moderate for noise+re-encode
- Large for different K values

This validates the concept of WAL-based model forensics. Next steps:
- Test on real fine-tuned models (instruct, code, medical)
- Build classifier: fingerprint → model type
- Target: >80% accuracy on held-out models

## Artifacts

- `experiments/m137_semantic_fingerprints.py`
- `experiments/m137_semantic_fingerprints.json`
