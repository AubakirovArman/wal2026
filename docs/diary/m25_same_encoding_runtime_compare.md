# M25 SAME ENCODING RUNTIME COMPARE

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m25_same_encoding_runtime_compare.

## Configuration
See source code for full configuration.

## Method / What was tested
See `experiments/m25_same_encoding_runtime_compare.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m25_same_encoding_runtime_compare.py`

## Notes from dev_diary_ru.md
```
- добавлена сериализация и повторная загрузка encodings через `src/encoding_io.py`;
- добавлен `replace_with_preencoded_packed_block_rvq()`;
- создан `experiments/m25_same_encoding_runtime_compare.py`, который сравнивает runtime-стратегии на буквально одних и тех же encodings.

Это оказалось критически важным. Отдельные encode pass сильно шумели runtime-выводы, и до same-encoding harness можно было принять encode-noise за runtime gain или runtime loss.
```


## Detailed Notes from dev_diary_ru.md

### Mention 1

```text

- все последующие packed сравнения теперь имеют более честную baseline-точку;
- стало видно, что даже без нового kernel можно снять заметную часть издержек просто правильным cache/layout уровнем;
- именно `full_weight_fast` после этого стал рабочим packed default для следующих M21--M25 шагов.

Главный вывод шага 13:

- без fast reconstruct все дальнейшие выводы про stage-drop, hot/cold и Triton были бы сильно зашумлены слишком медленным baseline;
- M20 не решил runtime целиком, но зафиксировал правильную отправную точку для честной frontier-оценки.

## Шаг 14. M21, M22 и M24: число residual stage --- полезный, но слишком грубый рычаг

- После M20 следующим естественным вопросом было: если уже есть calibrated residual stages, нельзя ли просто исполнять меньше стадий и получить speedup почти бесплатно?
- Здесь сразу всплыл важный correctness-нюанс: для product-split RVQ stage indices живут в split-major порядке, и наивный global stage-drop был просто неверен. В `src/runtime.py` пришлось добавить split-aware active stage indexing.

На локальном M21 microbench это действительно выглядело многообещающе. Для `l54.self_attn.q_proj`:

- `3` stage: `24.95 ms`, relMSE `0.0404`
```

### Mention 2

```text
- вопрос стоит не только как many stages, а какие именно `(stage, id)` токены реально важны для данного слоя и данного активационного режима;
- это и подвело нас к следующему шагу: грамматике влияния и stage-local hot/cold runtime.

## Шаг 15. M23 и M25: грамматика `(stage, id)` и same-encoding runtime меняют картину

- Для ответа на вопрос про важность токенов был добавлен `experiments/m23_id_influence_grammar.py`.
- Он строит activation-weighted grammar поверх packed Block-RVQ слоёв: для каждого `(stage, id)` токена аккумулируется влияние, зависящее от нормы входного блока, `row_scale` и нормы соответствующего codebook-вектора.

Первый важный вывод M23:

- merged layer-wide vocabulary действительно слишком диффузен;
- на `first8_qk_gu` средняя merged `top32_share` составляет только `0.058` для attention и `0.066` для MLP;
- но отдельные sub-stage заметно концентрированы: средняя `top8_share` по стадиям `0.107`, медиана `0.101`, максимум `0.451`.

Отсюда возникло главное инженерное решение:

- hot/cold split надо делать не по whole-layer vocabulary, а stage-local: `topN IDs` внутри каждой sub-stage.

```

### Mention 3

```text

- hot/cold split надо делать не по whole-layer vocabulary, а stage-local: `topN IDs` внутри каждой sub-stage.

После этого в runtime появился experimental path `full_weight_hot`, а затем M25 довёл методологию до честного состояния:

- добавлена сериализация и повторная загрузка encodings через `src/encoding_io.py`;
- добавлен `replace_with_preencoded_packed_block_rvq()`;
- создан `experiments/m25_same_encoding_runtime_compare.py`, который сравнивает runtime-стратегии на буквально одних и тех же encodings.

Это оказалось критически важным. Отдельные encode pass сильно шумели runtime-выводы, и до same-encoding harness можно было принять encode-noise за runtime gain или runtime loss.

После честного same-encoding сравнения картина стала такой:

- `l0_qkv_gu`, `4` окна: `full_weight_fast` `1228.23 tok/s` -> `full_weight_hot` `1335.39 tok/s` при идентичной `PPL = 56.9796`
- `l54_q_gu`, `4` окна: `1308.03 tok/s` -> `1396.46 tok/s` при идентичной `PPL = 2.3850`
- `l54_q_gu`, `16` окон: `933.41 tok/s` -> `962.71 tok/s` при идентичной `PPL = 2.7996` и том же `eval_peak = 46385.9 MB`

Это первый чистый факт, что grammar-aware stage-local hot/cold действительно может быть exact runtime improvement, если мерить его честно.
```


## Known Results (from project context)

**Result:** Same-encoding runtime comparison methodology.

**Notes:** Drove method to honest packed positive frontier assessment.


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
