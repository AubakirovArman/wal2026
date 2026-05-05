# M23 ID INFLUENCE GRAMMAR

## Date
2026 (exact date from git log or experiment run)

## Goal
M23: activation-weighted ID influence grammar for packed Block-RVQ layers.

## Configuration
See source code for full configuration.

## Method / What was tested
Builds a per-layer "grammar" over discrete weight tokens. For every packed
Block-RVQ layer we accumulate how much each `(stage, id)` contributes under
real activations from calibration windows.

Influence model:
    influence(row, block, stage) ~= ||x_block||_2 * row_scale[row] * ||codebook[id]||_2

This is not a downstream quality metric; it is a routing prior that answers:
which discrete IDs are the hot words of a layer's language?

Output:
  - per-layer concentration metrics: top8/top16/top32/top64 influence share
  - effective weighted vocab size (exp(entropy))
  - top tokens `(stage,id)` by influence share
  - per-stage histograms for later hot/cold split experiments

## Result
Encode test.

## Artifacts
- `experiments/m23_id_influence_grammar.py`

## Notes from dev_diary_ru.md
```
## Шаг 15. M23 и M25: грамматика `(stage, id)` и same-encoding runtime меняют картину

- Для ответа на вопрос про важность токенов был добавлен `experiments/m23_id_influence_grammar.py`.
- Он строит activation-weighted grammar поверх packed Block-RVQ слоёв: для каждого `(stage, id)` токена аккумулируется влияние, зависящее от нормы входного блока, `row_scale` и нормы соответствующего codebook-вектора.

```

```
- Реализация:
  - `experiments/m27_rrf_collect.py` — на двух калибровочных окнах wikitext собирает per-(stage,id) influence (формула M23: `||x_block|| · row_scale · ||codebook[id]||`), per-id occurrence counts и **структурную** interference matrix per stage (tile_size=256 = `block_n` B2-kernel'а; entry (i,j) = доля тайлов, где встречаются оба id).
  - Артефакты: `results/m23_influence/l54_gate_up.pt` (6.4 MB), `..._summary.json`.
  - `experiments/m27_rrf_step1a_offline.py` — гоняет три политики (`topk_count`, `topk_influence`, RRF linear-scan) на capacities `{32, 64, 128, 256}`, считает count-mass hit rate.
- Результаты (mean across 12 stages per layer):
```

```
- Реализация:
  - `experiments/m27_ptdp_collect.py` — кодирует только `model.layers.54.mlp.gate_proj` и `model.layers.54.mlp.up_proj`, прогоняет 2 calibration windows и для каждого tile/stage сохраняет `stage_tile_counts` и `stage_tile_influence`.
  - Артефакты: `results/m23_influence/l54_gate_up_ptdp.pt`, `results/m23_influence/l54_gate_up_ptdp_summary.json`.
- Результаты (mean count-hit across 12 stages per layer):

```

## Known Results (from project context)

**Result:** Grammar (stage, id) influence analysis. ID vocabulary is real and structured.

**Notes:** Led to experimental full_weight_hot path in runtime.
