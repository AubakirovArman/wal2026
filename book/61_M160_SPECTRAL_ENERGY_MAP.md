# M160 — Spectral Energy Map

**Date:** 2026-04-20
**Status:** ✅ Complete (v3 real weights)
**Goal:** Measure DCT energy distribution across layers and modules.

## Method

- Real Llama-3.1-8B weights loaded on CPU
- 5 layers (0, 8, 16, 24, 31), 5 modules each
- 2D DCT type-2 with ortho normalization
- Energy divided into 4 quadrants: LL, LH, HL, HH

## Results

All energy ratios are approximately **0.25** (±0.002) across all layers and modules.

| Layer | Module | LL | LH | HL | HH |
|-------|--------|-----|-----|-----|-----|
| 0 | q_proj | 0.248 | 0.248 | 0.252 | 0.252 |
| 0 | k_proj | 0.249 | 0.249 | 0.251 | 0.251 |
| 0 | v_proj | 0.250 | 0.250 | 0.250 | 0.250 |
| 0 | o_proj | 0.250 | 0.250 | 0.250 | 0.250 |
| 0 | gate_proj | 0.250 | 0.250 | 0.250 | 0.250 |
| 31 | q_proj | 0.250 | 0.249 | 0.251 | 0.250 |
| 31 | k_proj | 0.250 | 0.249 | 0.251 | 0.250 |
| 31 | v_proj | 0.249 | 0.250 | 0.250 | 0.250 |

## Key Finding

**Trained LLM weights have nearly uniform DCT energy distribution.**

This is consistent with the observation that trained neural network weights resemble random matrices with small perturbations. The training process (SGD) does not introduce strong spectral bias in the DCT basis.

## Implications

- DCT quadrants alone are not discriminative for layer/module type
- Fine-grained spectral analysis (e.g., per-frequency-bin entropy, decay slope) may be needed
- The spectral uniformity supports the use of global transforms (Hadamard, DCT) — no frequency band is dramatically more important than others

## Artifacts

- `experiments/m160_spectral_energy_map_v3.py`
- `experiments/m160_spectral_energy_map.json`
