# Phase 10: Meta-Learning (M81–M82)

## The Problem

WAL compresses weights. But what if you want to fine-tune the model? Re-encoding is slow. Fine-tuning dense weights defeats the purpose.

## The Insight

Fine-tune **programs**, not weights.

## What Was Built

### WALProgramAdapter
LoRA-style residual adapter:
```python
adapted_weight = base_weight + (lora_A @ lora_B) * scaling
```
- Only adapter parameters are trainable
- Base WAL weight stays frozen
- Merge method for inference fusion

### WALCoeffAdapter
Learned coefficient offsets:
```python
adapted_coeff = base_coeff + learned_delta
```
- Operates in coefficient space (more WAL-native)
- Only 0.4% of full weight parameters for C=8

### WALAtomAdapter
Selective atom perturbations:
- Adapts subset of atoms (default: first 8)
- 2.0% of full weight parameters

### program_soup()
Merge programs from N models:
- Mean: average atom_ids and coeff_ids
- Majority: majority vote per position
- Weighted: weighted average

### evolve_programs()
Genetic algorithm on atom combinations:
- Population of program buffers
- Fitness = negative MSE
- Operators: mutation, crossover, elitism

## Test Results

| Test | Result |
|------|--------|
| WALProgramAdapter | ✅ PASS |
| WALCoeffAdapter | ✅ PASS |
| WALAtomAdapter | ✅ PASS |
| Program soup | ✅ PASS |
| Genetic evolution | ✅ PASS |
| Adapter + WALCachedLinear | ✅ PASS |
| Gradient flow | ✅ PASS |

## Why This Matters

Meta-learning on WAL programs is fundamentally different from weight fine-tuning:
- Weight fine-tuning: changes all N×M parameters → expressive but large
- Program fine-tuning: changes program selection or small adapters → compact
- Coefficient adaptation: changes multipliers, not basis vectors
- Genetic evolution: discrete optimization when gradients don't apply

## Files
- `src/wal/v1/meta.py`
- `experiments/m81_meta_learning.py`
- `experiments/m82_adapter_integration.py`
