# M55A WAL VARIABLE LENGTH

## Date
2026 (exact date from git log or experiment run)

## Goal
M55a: WAL-0 variable-length programs with early stopping.

## Configuration
K_ATOMS=128, KMEANS_ITERS=5, batch=524288

## Method / What was tested
For each weight, greedily encode until residual < threshold OR lmax reached.
Measure stop_depth distribution and compression gain.

## Result
PPL evaluation.

## Artifacts
- `experiments/m55a_wal_variable_length.py`
- `experiments/m55a_wal_variable_length.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- Следующий шаг: M55 — variable length (early stopping) для ещё большей компрессии.


## Шаг 30. M55a: Variable Length — trade-off compression vs quality

- GPU-native early stopping encode на layer 40 o_proj.
- Тестированы thresholds: 0.000, 0.001, 0.002, 0.005, 0.010, 0.020, 0.050.
- **Key insight**: при малых threshold (< 0.005) overhead stop_depth encoding превышает выигрыш от ранней остановки.
- **threshold=0.005**: d=0: 1.7%, d=1: 96.0%, d=2: 2.3%. Экономия: 0.24 bits/weight (2.3%). Но relMAE=0.032 — возможно, слишком грубо.
- **threshold=0.010**: d=0: 3.4%, d=1: 96.1%, d=2: 0.5%. Экономия: 0.38 bits/weight (3.8%). relMAE=0.047.
- **threshold=0.050**: d=0: 17.0%, d=1: 83.0%, d=2: 0.0%. Экономия: 1.57 bits/weight (15.6%). Но relMAE=0.177 — катастрофа.
- **Вывод**: variable length работает, но threshold нужно подбирать очень аккуратно. Для качества threshold должен быть < 0.002, но тогда экономия минимальна. Возможно, adaptive threshold (per-layer или per-row) решит проблему.
- Следующий шаг: M56 — grammar analysis программного потока. Нужно понять, есть ли структура помимо частот.


## Шаг 31. M56a: Grammar Analysis — WAL-0 программы = i.i.d. поток

- GPU-native grammar analysis на layer 40 o_proj.
```



## Extracted Metrics (from source)

- Time: .2
