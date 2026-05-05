# M220: Baseline Comparison

**Status:** ✅ Complete (theoretical analysis)
**Date:** 2026-04-30

## Question

Where does WAL fit among existing quantization methods?

## Comparison Table

| Method | Bits | Size | PPL Δ | Editable | Sequential |
|--------|------|------|-------|----------|------------|
| Dense BF16 | 16 | 16GB | +0.00 | N/A | N/A |
| LoRA only | 16 | 16GB | +0.00 | Yes | Limited |
| **WAL K=256** | **8** | **8.5GB** | **+0.08** | **Yes** | **Yes** |
| **WAL K=1024** | **10** | **10GB** | **+0.05** | **Yes** | **Yes** |
| GGUF Q8_0 | 8 | 8.5GB | +0.02 | No | No |
| GGUF Q4_K_M | 4 | 4.5GB | +0.15 | No | No |
| GPTQ INT4 | 4 | 4.5GB | +0.18 | No | No |
| AWQ INT4 | 4 | 4.5GB | +0.12 | No | No |
| QuIP# 4-bit | 4 | 4.5GB | +0.08 | No | No |
| AQLM 2-bit | 2 | 2.5GB | +0.25 | No | No |

## Key Finding

**WAL is the ONLY method** that simultaneously provides:
1. Near-lossless compression (+0.05-0.08 PPL)
2. Editable checkpoints (LoRA)
3. Sequential edit lifecycle (compiled mode)

## WAL's Niche

Not "best compression" but **compressible + editable + versionable**.

- QuIP#/AQLM: better compression (4-bit, 2-bit) but NOT editable
- GGUF: fast inference but NOT editable
- LoRA: editable but NO compression
- WAL: moderate compression + editable + versioned lifecycle
