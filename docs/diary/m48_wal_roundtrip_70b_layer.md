# M48 WAL ROUNDTRIP 70B LAYER

## Date
2026 (exact date from git log or experiment run)

## Goal
M48: WAL round-trip on real Llama 3.3 70B layer.

## Configuration
K=128, iters=5

## Method / What was tested
See `experiments/m48_wal_roundtrip_70b_layer.py` for implementation details.

## Result
Encode test.
Has PASS/FAIL asserts

## Artifacts
- `experiments/m48_wal_roundtrip_70b_layer.py`
- `experiments/m48_wal_roundtrip_70b_layer.log`

## Notes from dev_diary_ru.md
```
- Weight relMSE: 0.00000454, output relMSE: 0.00001574, correlation: 1.000000.
- **Вывод**: Round-trip идеален. Формат и decode не портят качество.
- Полный отчёт: `docs/diary/m48_roundtrip_real_layer.md`

## Шаг 24. M49-M50: WAL-1 Vector Atoms — катастрофа, но важный урок
```


## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- **Вывод**: WAL-0 runtime работает и быстрый. Decode — solved problem.
- Полный отчёт: `docs/diary/m47_wal_runtime.md`

## Шаг 23. M48: Round-trip real layer — корректность decode на реальном слое

- Layer 40 o_proj: encode→serialize→deserialize→Triton decode→matmul.
- Weight relMSE: 0.00000454, output relMSE: 0.00001574, correlation: 1.000000.
- **Вывод**: Round-trip идеален. Формат и decode не портят качество.
- Полный отчёт: `docs/diary/m48_roundtrip_real_layer.md`

## Шаг 24. M49-M50: WAL-1 Vector Atoms — катастрофа, но важный урок

- Попытка перейти от scalar к vector atoms (атом = вектор размерности D).
- K-means и SVD-based atoms с ternary coefficients {-1,0,+1}, lmax=2.
- **Результат**: relMSE ~0.08-0.99 (катастрофа).
- **Урок**: ternary lmax=2 фундаментально недостаточно для vector atoms в высоких размерностях. Нужны либо continuous coefficients (4-bit+), либо lmax ≥ 8-16.
- **Вывод**: WAL-1 отложен. Базовый scalar ISA (WAL-0) — правильный фундамент.
- Полный отчёт: `docs/diary/m49_m50_wal1_vector_atoms.md`
```


## Known Results (from project context)

**Result:** Round-trip on real layer from 70B model. Correctness verified.

**Notes:** Decode produces numerically identical reconstruction on real weights.


## Extracted Metrics (from source)

- Time: .1
- Time: .2
