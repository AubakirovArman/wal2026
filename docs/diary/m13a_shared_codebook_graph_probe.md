# M13A SHARED CODEBOOK GRAPH PROBE

## Date
2026 (exact date from git log or experiment run)

## Goal
Probe a graph-style shared PQ codebook across multiple weight tensors.

## Configuration
See source code for full configuration.

## Method / What was tested
Hypothesis (route-graph idea):
- Currently each tensor has its own per-stage PQ codebooks.
- Many block "motifs" recur across tensors (q vs k vs v of the same layer,
  same projection in adjacent layers, etc.).
- A graph-DB style representation would store one shared dictionary of motifs
  and let every tensor just point into it (= edges into shared nodes).

This probe measures, per product split and stage, how much storage and
quality changes when the codebook is shared across a chosen group of
tensors versus encoded per-tensor.

Run:
  python wal/experiments/m13a_shared_codebook_graph_probe.py       --tensor-names model.layers.54.self_attn.q_proj.weight,model.layers.54.self_attn.k_proj.weight       --block-size 32 --codebook-size 256 --num-stages 3 --product-splits 4       --out wal/results/m13a_l54_qk_shared.json

## Result
Encode test.

## Artifacts
- `experiments/m13a_shared_codebook_graph_probe.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.