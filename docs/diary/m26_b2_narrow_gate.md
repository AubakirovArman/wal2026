# M26 B2 NARROW GATE

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m26_b2_narrow_gate.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m26_b2_narrow_gate.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m26_b2_narrow_gate.py`

## Notes from dev_diary_ru.md
```
- В `src/runtime.py` добавлен exact path `full_weight_hot_v2`. Он использует один кэшированный recon-буфер на (group, dtype, device), инициализирует его первой стадией через `index_copy_`, а каждую следующую активную sub-stage прибавляет в него через `index_add_` без аллокаций.
- Numerically он остаётся exact относительно `full_weight_hot` и относительно `full_weight_fast` на тех же encodings.
- Запуск same-encoding compare на persisted cache `l54_q_gu`, `16` окон raw WikiText-2 (`results/m26_l54_q_gu_hot_v2_16w.json`):
  - `full_weight_fast`   PPL `2.7996`, `943.29` tok/s, peak VRAM `46385.9` MB, avg layer rel-MSE `4.193e-02`;
  - `full_weight_hot`    PPL `2.7996`, `962.06` tok/s, peak VRAM `46385.9` MB, avg layer rel-MSE `4.193e-02`;
```

```
  - `full_weight_hot`    PPL `2.7996`, `962.06` tok/s, peak VRAM `46385.9` MB, avg layer rel-MSE `4.193e-02`;
  - `full_weight_hot_v2` PPL `2.7996`, `971.28` tok/s, peak VRAM `46385.9` MB, avg layer rel-MSE `4.193e-02`.
- Дополнительный замер на persisted cache `l0_qkv_gu`, `16` окон (`results/m26_l0_qkv_gu_hot_v2_16w.json`), там патчатся `q,k,v,gate,up` слоя 0:
  - `full_weight_fast`   PPL `32.2524`, `947.92` tok/s, peak VRAM `47214.8` MB;
  - `full_weight_hot`    PPL `32.2524`, `948.17` tok/s, peak VRAM `88903.8` MB;
```

```
  - `fast_norm` pre-row-scale helper, чтобы plan строился из того же cached tensor, что и `full_weight_fast`.
- Узкие synthetic forward checks проходили: отдельные `q_proj / gate_proj / up_proj` слои считались без `NaN`, а после перехода на `fast_norm` средняя абсолютная разница на `q_proj` против `full_weight_fast` сжалась до `1.31e-03` (но max abs diff оставался `3.125e-02`, то есть это уже не exact parity).
- Честный end-to-end gate на `l54_q_gu`, `16` окон same-encoding (`results/m26_l54_q_gu_triton_hot_cold_persistent_16w.json`) дал отрицательный результат:
  - `full_weight_fast`   `PPL = 2.7996`, `936.81 tok/s`, peak `46385.9 MB`;
  - `full_weight_hot_v2` `PPL = 2.7996`, `959.52 tok/s`, peak `46385.9 MB`;
```
