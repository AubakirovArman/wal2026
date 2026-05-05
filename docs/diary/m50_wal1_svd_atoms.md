# M50 WAL1 SVD ATOMS

## Date
2026 (exact date from git log or experiment run)

## Goal
M50: WAL-1 with SVD-based atoms on real Llama 70B weights.

## Configuration
K=128, iters=5

## Method / What was tested
See `experiments/m50_wal1_svd_atoms.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m50_wal1_svd_atoms.py`
- `experiments/m50_wal1_svd_atoms.log`

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

**Result:** SVD-based atoms — also catastrophic.

**Notes:** Confirmed that atom structure must preserve per-weight independence.
