# M27 WAL LO PROTO

## Date
2026 (exact date from git log or experiment run)

## Goal
M27 WAL-LO Step 12b: layer-output reconstruction with strong program cost.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m27_wal_lo_proto.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m27_wal_lo_proto.py`

## Notes from dev_diary_ru.md
```
- Идея от пользователя: посмотреть на `(stage, id)` как на "регистры" с allocation-policy в духе linear-scan, где `Priority = Influence × (1 − avg_interference)`. Цель Step 1a — **offline** проверить, поднимает ли interference-aware allocator hit rate над голым topk на `l54.mlp.gate_proj`/`l54.mlp.up_proj` — без kernel-touch'а.
- Реализация:
  - `experiments/m27_rrf_collect.py` — на двух калибровочных окнах wikitext собирает per-(stage,id) influence (формула M23: `||x_block|| · row_scale · ||codebook[id]||`), per-id occurrence counts и **структурную** interference matrix per stage (tile_size=256 = `block_n` B2-kernel'а; entry (i,j) = доля тайлов, где встречаются оба id).
  - Артефакты: `results/m23_influence/l54_gate_up.pt` (6.4 MB), `..._summary.json`.
  - `experiments/m27_rrf_step1a_offline.py` — гоняет три политики (`topk_count`, `topk_influence`, RRF linear-scan) на capacities `{32, 64, 128, 256}`, считает count-mass hit rate.
```

```
  - `experiments/m27_rrf_collect.py` — на двух калибровочных окнах wikitext собирает per-(stage,id) influence (формула M23: `||x_block|| · row_scale · ||codebook[id]||`), per-id occurrence counts и **структурную** interference matrix per stage (tile_size=256 = `block_n` B2-kernel'а; entry (i,j) = доля тайлов, где встречаются оба id).
  - Артефакты: `results/m23_influence/l54_gate_up.pt` (6.4 MB), `..._summary.json`.
  - `experiments/m27_rrf_step1a_offline.py` — гоняет три политики (`topk_count`, `topk_influence`, RRF linear-scan) на capacities `{32, 64, 128, 256}`, считает count-mass hit rate.
- Результаты (mean across 12 stages per layer):
  | cap | topk_count | topk_influence | RRF |
```

```
- Следующая явная гипотеза от пользователя была уже в духе tile-level compilation: не один большой register file на всю стадию, а **Per-Tile Dynamic Palette (PTDP)**. Для каждого тайла веса размером `64 x 256` (64 output rows, 256 input columns) собирается собственная palette на `topk = 32/48/64` id по **tile-local activation-weighted influence**, и дальше мы смотрим, может ли эта palette покрывать сам тайл хотя бы на уровне `0.75-0.85` без какого-либо kernel work.
- Реализация:
  - `experiments/m27_ptdp_collect.py` — кодирует только `model.layers.54.mlp.gate_proj` и `model.layers.54.mlp.up_proj`, прогоняет 2 calibration windows и для каждого tile/stage сохраняет `stage_tile_counts` и `stage_tile_influence`.
  - Артефакты: `results/m23_influence/l54_gate_up_ptdp.pt`, `results/m23_influence/l54_gate_up_ptdp_summary.json`.
- Результаты (mean count-hit across 12 stages per layer):
```
