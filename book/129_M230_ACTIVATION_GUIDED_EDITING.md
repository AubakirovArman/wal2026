# M230: Activation-Guided Editing (Auto-Select Modules)

**Status:** ✅ Complete
**Date:** 2026-05-01

## Question

Can we auto-select optimal layers and modules for factual editing by tracing activation magnitudes on target facts?

## Hypothesis

Modules with highest activation response to target facts should be the best targets for LoRA editing.

## Method

1. Forward-pass target facts through base model
2. Record per-module activation norms (L2 of hidden states)
3. Select top-k layers and modules by activation response
4. Train LoRA on selected modules
5. Compare vs hardcoded selection (layers 14-16, modules o_proj/q_proj/v_proj/gate_proj)

## Results

```
Method                    PPL Δ    Survival  Time
--------------------------------------------------
Activation-Guided        +0.0236   0/5      12.0s
Hardcoded (layers 14-16) +0.3096   3/5       9.6s
```

### Activation Trace (Top 10)

```
model.layers.31.mlp.gate_proj:    146.49
model.layers.31.mlp.act_fn:       120.83
model.layers.31.mlp.up_proj:      109.96
model.layers.30.mlp.gate_proj:    108.01
model.layers.31.mlp.down_proj:    105.20
model.layers.29.mlp.gate_proj:     90.91
model.layers.31.self_attn.q_proj:  85.25
model.layers.28.mlp.gate_proj:     84.03
model.layers.0.self_attn.q_proj:   79.25
```

### Selected by Activation-Guided
- **Layers:** [31, 30, 29, 28] (late layers!)
- **Modules:** [q_proj, gate_proj, k_proj, up_proj]

### Hardcoded Baseline
- **Layers:** [14, 15, 16] (middle layers)
- **Modules:** [o_proj, q_proj, v_proj, gate_proj]

## Key Finding

> **High activation ≠ good editability.**

Activation-Guided selected late layers (28-31) with massive activation norms but produced **0/5 survival**. Hardcoded middle layers (14-16) with lower activation produced **3/5 survival**.

## Why Late Layers Fail for Factual Editing

- Late layers (28-31) are **output formation layers** — they shape token distributions but don't store factual associations
- Middle layers (14-16) contain **factual knowledge** in MLP blocks (as shown by ROME/MEMIT literature)
- High activation in late layers reflects **output confidence**, not **editable knowledge**

## Implications

- Activation tracing is useful for **diagnostics** (detecting anomalous behavior, measuring model engagement)
- Activation tracing is **NOT suitable for auto-selecting editing targets**
- Factual editing requires **domain knowledge** about where facts are stored (middle MLP layers)
- Future: combine activation tracing with **causal mediation analysis** (like ROME) for proper target selection

## Conclusion

Activation-Guided Editing as a standalone module selector **fails**. The best editing targets are not the most active modules. This confirms that:

1. Factual knowledge lives in specific layers (middle, not late)
2. Activation magnitude measures engagement, not editability
3. Auto-selection needs causal reasoning, not just correlation

## Next Steps

- Use activation tracing for **forensics/diagnostics** (M224 extension)
- For editing target selection: use **causal tracing** (ROME-style) or **hardcoded domain knowledge**
- Explore if activation patterns can predict **PPL impact** of edits (not survival)
