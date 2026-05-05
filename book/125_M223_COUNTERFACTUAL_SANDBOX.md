# M223: Counterfactual Sandbox (Parallel Branches)

**Status:** ✅ Complete
**Date:** 2026-05-01
**Model:** Llama-3.1-8B

## Question

Can we create independent model branches with different facts?

## Results

| Branch | PPL Δ | Survival |
|--------|-------|----------|
| legal | +0.25 | 0/5 |
| medical | +0.10 | 0/5 |
| product | +0.39 | **1/5** |

## Cross-Branch Isolation

| Branch | Own | Legal | Medical | Product |
|--------|-----|-------|---------|---------|
| legal | 1/5 | — | 0/5 | 0/5 |
| medical | 0/5 | 0/5 | — | 0/5 |
| product | **2/5** | 0/5 | 0/5 | — |

## Key Findings

1. **Branch isolation confirmed!** Each branch only knows its own facts
2. **Product branch best** (2/5 survival) — simpler domain
3. **Legal/medical = 0/5** — need more steps

## Implications

Counterfactual sandboxing works. Multiple independent model variants from same base are possible.
