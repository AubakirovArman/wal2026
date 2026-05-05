# M20 FAST RECON MICROBENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
M20 microbench: cached fast reconstruct vs reference per_group / full_weight.

## Configuration
iters=50

## Method / What was tested
Loads one real Llama-3.3-70B linear weight (l54.self_attn.q_proj), encodes it
with the production Block-RVQ config used by the q_gu frontier, then benches
forward() across matmul strategies. Also reports rel_mse vs the bf16 baseline.

## Result
Benchmark.

## Artifacts
- `experiments/m20_fast_recon_microbench.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text

Это и есть та основная operational summary, которую теперь нужно держать перед глазами в каждом следующем runtime шаге.

## Шаг 13. M20: fast reconstruct стал новой packed baseline

- После длинной серии local-palette и grouped-runtime экспериментов стало ясно, что для новой Block-RVQ ветки нужен более честный packed baseline, иначе мы сравниваем новые идеи со слишком медленным reference path.
- Для этого в `src/runtime.py` был добавлен M20 fast reconstruct: кэшируем `bf16` codebook-представления и переиспользуем буферы восстановления веса вместо повторной полной пересборки на каждом вызове.

M20 microbench на матрице `8192 x 8192` при `seq=2048` показал:

- `per_group`: `35.03 ms`
- `per_group_fast`: `26.56 ms`
- `full_weight`: `33.74 ms`
- `full_weight_fast`: `24.95 ms`

При этом relMSE остался тем же порядком (`~0.0404`), то есть выигрыш оказался именно runtime-выигрышем, а не изменением кодирования.

Что это дало практически:
```

### Mention 2

```text
## Шаг 13. M20: fast reconstruct стал новой packed baseline

- После длинной серии local-palette и grouped-runtime экспериментов стало ясно, что для новой Block-RVQ ветки нужен более честный packed baseline, иначе мы сравниваем новые идеи со слишком медленным reference path.
- Для этого в `src/runtime.py` был добавлен M20 fast reconstruct: кэшируем `bf16` codebook-представления и переиспользуем буферы восстановления веса вместо повторной полной пересборки на каждом вызове.

M20 microbench на матрице `8192 x 8192` при `seq=2048` показал:

- `per_group`: `35.03 ms`
- `per_group_fast`: `26.56 ms`
- `full_weight`: `33.74 ms`
- `full_weight_fast`: `24.95 ms`

При этом relMSE остался тем же порядком (`~0.0404`), то есть выигрыш оказался именно runtime-выигрышем, а не изменением кодирования.

Что это дало практически:

- все последующие packed сравнения теперь имеют более честную baseline-точку;
- стало видно, что даже без нового kernel можно снять заметную часть издержек просто правильным cache/layout уровнем;
```

### Mention 3

```text
- После длинной серии local-palette и grouped-runtime экспериментов стало ясно, что для новой Block-RVQ ветки нужен более честный packed baseline, иначе мы сравниваем новые идеи со слишком медленным reference path.
- Для этого в `src/runtime.py` был добавлен M20 fast reconstruct: кэшируем `bf16` codebook-представления и переиспользуем буферы восстановления веса вместо повторной полной пересборки на каждом вызове.

M20 microbench на матрице `8192 x 8192` при `seq=2048` показал:

- `per_group`: `35.03 ms`
- `per_group_fast`: `26.56 ms`
- `full_weight`: `33.74 ms`
- `full_weight_fast`: `24.95 ms`

При этом relMSE остался тем же порядком (`~0.0404`), то есть выигрыш оказался именно runtime-выигрышем, а не изменением кодирования.

Что это дало практически:

- все последующие packed сравнения теперь имеют более честную baseline-точку;
- стало видно, что даже без нового kernel можно снять заметную часть издержек просто правильным cache/layout уровнем;
- именно `full_weight_fast` после этого стал рабочим packed default для следующих M21--M25 шагов.

```


## Known Results (from project context)

**Result:** Fast reconstruct: caches bf16 codebook representations.

**Notes:** M20 became new packed baseline. full_weight_fast became working packed default for M21-M25.
