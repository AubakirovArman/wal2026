# M54B WAL CODEBOOK DECODE

## Date
2026 (exact date from git log or experiment run)

## Goal
M54b: WAL-0 codebook packing + direct decode from codebook.

## Configuration
K_ATOMS=128, KMEANS_ITERS=5, block_size=1024

## Method / What was tested
Two decode strategies:
1. Atom-lookup decode: program_id → (atom_ids, signs) → gather atoms → sum
2. Precomputed-recon decode: program_id → recon_value (precomputed float32)

Both GPU-native. Compare speed vs raw WAL-0 decode.

## Result
Encode test.

## Artifacts
- `experiments/m54b_wal_codebook_decode.py`
- `experiments/m54b_wal_codebook_decode.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- Mining time: **0.088s** на GPU.
- Сравнение с DRL v2: DRL давал ~1500 unique routes на тензор при L_max=12. WAL-0 даёт ~1080 при lmax=2 — и качество в 200× лучше.
- **Вывод**: WAL-0 программы обладают огромной повторяемостью. Это доказывает, что "язык" реально существует — мы можем построить vocabulary из ~1000 слов, которыми выражаются 67M весов.
- Следующий шаг: M54b — pack programs в compact IDs + decode directly from codebook.


## Шаг 29. M54b: Codebook Decode — 1.1 TW/s!

- Сравнены 3 стратегии decode на layer 40 o_proj (67M weights):
  1. **Atom-lookup via codebook**: 23,945 Mw/s — gather atoms по codebook, затем sum.
  2. **Precomputed recon lookup**: **1,144,207 Mw/s** (~1.1 TW/s!) — просто `table[program_id]`.
  3. **Triton raw programs**: 93.5 Mw/s — unexpectedly slow, возможно из-за lmax=2.
- **Все три дают exact reconstruction** (max error 0.00).
- **Precomputed recon** — революция: decode = один `index_select` по таблице из ~1000 float32 значений. Это near-memory-bandwidth limit.
- **Компрессия**: 10 bits/weight для program_id (uint10) + 4236 bytes codebook table.
- Сравнение: raw WAL-0 = 16 bits/weight. Codebook экономит **6 bits/weight (37.5%)**.
- Следующий шаг: M55 — variable length (early stopping) для ещё большей компрессии.

```

### Mention 2

```text
- Следующий шаг: M54b — pack programs в compact IDs + decode directly from codebook.


## Шаг 29. M54b: Codebook Decode — 1.1 TW/s!

- Сравнены 3 стратегии decode на layer 40 o_proj (67M weights):
  1. **Atom-lookup via codebook**: 23,945 Mw/s — gather atoms по codebook, затем sum.
  2. **Precomputed recon lookup**: **1,144,207 Mw/s** (~1.1 TW/s!) — просто `table[program_id]`.
  3. **Triton raw programs**: 93.5 Mw/s — unexpectedly slow, возможно из-за lmax=2.
- **Все три дают exact reconstruction** (max error 0.00).
- **Precomputed recon** — революция: decode = один `index_select` по таблице из ~1000 float32 значений. Это near-memory-bandwidth limit.
- **Компрессия**: 10 bits/weight для program_id (uint10) + 4236 bytes codebook table.
- Сравнение: raw WAL-0 = 16 bits/weight. Codebook экономит **6 bits/weight (37.5%)**.
- Следующий шаг: M55 — variable length (early stopping) для ещё большей компрессии.


## Шаг 30. M55a: Variable Length — trade-off compression vs quality

```

