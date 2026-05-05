# M49 WAL1 VECTOR ATOMS

## Date
2026 (exact date from git log or experiment run)

## Goal
M49: WAL-1 Vector Atoms prototype — row-wise encoding vs WAL-0 scalar.

## Configuration
K=128, batch=1024, iters=5

## Method / What was tested
See `experiments/m49_wal1_vector_atoms.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m49_wal1_vector_atoms.py`
- `experiments/m49_wal1_vector_atoms.log`

## Notes from dev_diary_ru.md
```
- **Урок**: ternary lmax=2 фундаментально недостаточно для vector atoms в высоких размерностях. Нужны либо continuous coefficients (4-bit+), либо lmax ≥ 8-16.
- **Вывод**: WAL-1 отложен. Базовый scalar ISA (WAL-0) — правильный фундамент.
- Полный отчёт: `docs/diary/m49_m50_wal1_vector_atoms.md`

## Шаг 25. M51: WAL Compiler — компиляция не даёт выигрыша для K=128
```


## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- **Вывод**: Round-trip идеален. Формат и decode не портят качество.
- Полный отчёт: `docs/diary/m48_roundtrip_real_layer.md`

## Шаг 24. M49-M50: WAL-1 Vector Atoms — катастрофа, но важный урок

- Попытка перейти от scalar к vector atoms (атом = вектор размерности D).
- K-means и SVD-based atoms с ternary coefficients {-1,0,+1}, lmax=2.
- **Результат**: relMSE ~0.08-0.99 (катастрофа).
- **Урок**: ternary lmax=2 фундаментально недостаточно для vector atoms в высоких размерностях. Нужны либо continuous coefficients (4-bit+), либо lmax ≥ 8-16.
- **Вывод**: WAL-1 отложен. Базовый scalar ISA (WAL-0) — правильный фундамент.
- Полный отчёт: `docs/diary/m49_m50_wal1_vector_atoms.md`

## Шаг 25. M51: WAL Compiler — компиляция не даёт выигрыша для K=128

- Попытка compile-time specialization atom table в Triton kernel.
- Generic kernel уже 417 Mw/s (near memory bandwidth).
- **Вывод**: для K=128 compile-time specialization не нужен. Для K=8-32 мог бы дать выигрыш, но не приоритет.
- Полный отчёт: `docs/diary/m51_wal_compiler.md`
```


## Known Results (from project context)

**Result:** WAL-1 vector atoms — CATASTROPHIC PPL. Important negative result.

**Notes:** Vector quantization of atoms themselves fails. Scalar atoms are necessary for quality.
