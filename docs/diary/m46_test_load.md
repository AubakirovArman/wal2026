# M46 TEST LOAD

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m46_test_load.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m46_test_load.py` for implementation details.

## Result
Unknown.
Has PASS/FAIL asserts

## Artifacts
- `experiments/m46_test_load.py`
- `experiments/m46_test_load.log`

## Notes from dev_diary_ru.md
```
- Время: 2715 секунд (45.2 минуты).
- **Вывод**: WAL-0 scalar на уровне качества dense модели. Это валидирует базовый ISA.
- Полный отчёт: `docs/diary/m46_wal_scalar_70b_ppl.md`

## Шаг 22. M47: WAL Runtime — decode, round-trip, serialization
```


## Detailed Notes from dev_diary_ru.md

### Mention 1

```text



## Шаг 21. M46: Full 70B WAL Scalar PPL — доказательство качества на полной модели

- Полный end-to-end encode всех 560 block linear тензоров Llama 3.3 70B.
- K=128, lmax=2, per-row normalization, skip-spiky layers (std < 0.08).
- **Результат PPL**: 2.7821 vs dense baseline 2.7805 — **gap всего +0.06%** (+0.0016 nats).
- Закодировано 540 params, пропущено 183 spiky params.
- Время: 2715 секунд (45.2 минуты).
- **Вывод**: WAL-0 scalar на уровне качества dense модели. Это валидирует базовый ISA.
- Полный отчёт: `docs/diary/m46_wal_scalar_70b_ppl.md`

## Шаг 22. M47: WAL Runtime — decode, round-trip, serialization

- Реализован полный execution stack: `isa.py`, `encoder.py`, `decoder.py`, `triton_kernels.py`, `format.py`.
- Triton decode: 406.7 Mw/s на 100M weights (near memory bandwidth).
- Round-trip serialize→deserialize: max error 0.0.
```

### Mention 2

```text

## Текущий фокус: WAL-0 → Язык (M54+)

M46-M53 доказали: WAL-0 scalar — отличный **кодек**. Качество уровня dense, decode быстрый, encode масштабируется.

Но это ещё не **язык**. Нужно три вещи:

1. **Codebook layer** (M54): unique programs → IDs. Как в DRL v2: route → ID.
2. **Variable length** (M55): early stopping, stop_depth. Не фиксированный lmax=2.
3. **Grammar / Structure** (M56+): анализ частот, n-grams, reusable subroutines.

Приоритет: GPU-native всё. Никаких CPU копий.



## Шаг 28. M54a: WAL-0 Codebook Mining — язык существует!

- GPU-native codebook mining на layer 40 self_attn.o_proj (67M weights).
```

### Mention 3

```text

- Full end-to-end encode всех 540 params с codebook layer.
- **PPL: 2.7828** — gap vs baseline 2.7805: **+0.08%** (+0.0023 nats).
- Сравнение: M46=2.7821, M53c=2.7858. Codebook не портит качество.
- **Время encode: 437 секунд (7.3 минуты)** — в 6× быстрее M53c (2225s) и M46 (2715s)!
- **Total unique programs: 609,643** на всю модель, avg 1129 per layer.
- **Баг found и fixed**: `codebook_recon` был проиндексирован по `unique_prog` порядку, а `program_ids` — по `sort_idx` (frequency sort). Это давало catastrophic PPL 299,002 на первом запуске. Fix: использовать `inverse` от `torch.unique` напрямую.
- **Вывод**: WAL-0 с codebook layer масштабируется на полную модель, сохраняет качество, и encode в 6× быстрее.


```

