# M47 WAL RUNTIME TEST

## Date
2026 (exact date from git log or experiment run)

## Goal
M47: WAL-0 Runtime test — encode, decode, serialize, benchmark.

## Configuration
K=128, iters=5

## Method / What was tested
See `experiments/m47_wal_runtime_test.py` for implementation details.

## Result
Benchmark.
Has PASS/FAIL asserts

## Artifacts
- `experiments/m47_wal_runtime_test.py`
- `experiments/m47_wal_runtime_test.log`

## Notes from dev_diary_ru.md
```
- Round-trip serialize→deserialize: max error 0.0.
- **Вывод**: WAL-0 runtime работает и быстрый. Decode — solved problem.
- Полный отчёт: `docs/diary/m47_wal_runtime.md`

## Шаг 23. M48: Round-trip real layer — корректность decode на реальном слое
```


## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- **Вывод**: WAL-0 scalar на уровне качества dense модели. Это валидирует базовый ISA.
- Полный отчёт: `docs/diary/m46_wal_scalar_70b_ppl.md`

## Шаг 22. M47: WAL Runtime — decode, round-trip, serialization

- Реализован полный execution stack: `isa.py`, `encoder.py`, `decoder.py`, `triton_kernels.py`, `format.py`.
- Triton decode: 406.7 Mw/s на 100M weights (near memory bandwidth).
- Round-trip serialize→deserialize: max error 0.0.
- **Вывод**: WAL-0 runtime работает и быстрый. Decode — solved problem.
- Полный отчёт: `docs/diary/m47_wal_runtime.md`

## Шаг 23. M48: Round-trip real layer — корректность decode на реальном слое

- Layer 40 o_proj: encode→serialize→deserialize→Triton decode→matmul.
- Weight relMSE: 0.00000454, output relMSE: 0.00001574, correlation: 1.000000.
- **Вывод**: Round-trip идеален. Формат и decode не портят качество.
- Полный отчёт: `docs/diary/m48_roundtrip_real_layer.md`

```


## Known Results (from project context)

**Result:** WAL Runtime — decode, round-trip, serialization test.

**Notes:** Verified WAL program execution correctness.


## Extracted Metrics (from source)

- Time: .2
