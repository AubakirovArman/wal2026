# M51 WAL COMPILER

## Date
2026 (exact date from git log or experiment run)

## Goal
M51: WAL Compiler — JIT specialized kernels with inline atom tables.

## Configuration
K=128

## Method / What was tested
See `experiments/m51_wal_compiler.py` for implementation details.

## Result
Benchmark.

## Artifacts
- `experiments/m51_wal_compiler.py`
- `experiments/m51_wal_compiler.log`

## Notes from dev_diary_ru.md
```
- Generic kernel уже 417 Mw/s (near memory bandwidth).
- **Вывод**: для K=128 compile-time specialization не нужен. Для K=8-32 мог бы дать выигрыш, но не приоритет.
- Полный отчёт: `docs/diary/m51_wal_compiler.md`

## Шаг 26. M52: Cross-layer atom sharing — shared atoms бьют per-layer
```


## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- **Вывод**: WAL-1 отложен. Базовый scalar ISA (WAL-0) — правильный фундамент.
- Полный отчёт: `docs/diary/m49_m50_wal1_vector_atoms.md`

## Шаг 25. M51: WAL Compiler — компиляция не даёт выигрыша для K=128

- Попытка compile-time specialization atom table в Triton kernel.
- Generic kernel уже 417 Mw/s (near memory bandwidth).
- **Вывод**: для K=128 compile-time specialization не нужен. Для K=8-32 мог бы дать выигрыш, но не приоритет.
- Полный отчёт: `docs/diary/m51_wal_compiler.md`

## Шаг 26. M52: Cross-layer atom sharing — shared atoms бьют per-layer

- Shared atoms (пул из 8 слоёв) vs per-layer atoms.
- **Результат**: shared atoms **лучше** per-layer до 7.7× по relMSE.
- **Почему**: больше данных → лучше k-means → лучше atoms.
- **Вывод**: глобальный atom table жизнеспособен. Это путь к единому языку для всей модели.
- Полный отчёт: `docs/diary/m52_cross_layer_sharing.md`

```


## Known Results (from project context)

**Result:** WAL Compiler — compilation gives no win for K=128.

**Notes:** Compiler optimization not beneficial at small K. Focus shifted to cross-layer sharing.
