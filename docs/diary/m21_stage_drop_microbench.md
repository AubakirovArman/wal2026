# M21 STAGE DROP MICROBENCH

## Date
2026 (exact date from git log or experiment run)

## Goal
M21: variable-stage decoding microbench on l54.self_attn.q_proj.

## Configuration
iters=30, block_size=32

## Method / What was tested
Encodes once with num_stages=3, then sweeps `effective_stages` ∈ {3, 2, 1}
and reports per-call time + rel_mse vs bf16 baseline. This scopes whether
globally dropping stages is even survivable on quality.

## Result
Benchmark.

## Artifacts
- `experiments/m21_stage_drop_microbench.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

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
- без fast reconstruct все дальнейшие выводы про stage-drop, hot/cold и Triton были бы сильно зашумлены слишком медленным baseline;
- M20 не решил runtime целиком, но зафиксировал правильную отправную точку для честной frontier-оценки.

## Шаг 14. M21, M22 и M24: число residual stage --- полезный, но слишком грубый рычаг

- После M20 следующим естественным вопросом было: если уже есть calibrated residual stages, нельзя ли просто исполнять меньше стадий и получить speedup почти бесплатно?
- Здесь сразу всплыл важный correctness-нюанс: для product-split RVQ stage indices живут в split-major порядке, и наивный global stage-drop был просто неверен. В `src/runtime.py` пришлось добавить split-aware active stage indexing.

На локальном M21 microbench это действительно выглядело многообещающе. Для `l54.self_attn.q_proj`:

- `3` stage: `24.95 ms`, relMSE `0.0404`
- `2` stage: `16.87 ms`, relMSE `0.1171`
- `1` stage: `8.81 ms`, relMSE `0.3407`

То есть как локальный speed lever stage-drop работает, но качество падает очень быстро.

На честном `16`-window raw WikiText-2 gate для `first8_qk_gu` картина оказалась ещё важнее:

```

### Mention 3

```text
- После M20 следующим естественным вопросом было: если уже есть calibrated residual stages, нельзя ли просто исполнять меньше стадий и получить speedup почти бесплатно?
- Здесь сразу всплыл важный correctness-нюанс: для product-split RVQ stage indices живут в split-major порядке, и наивный global stage-drop был просто неверен. В `src/runtime.py` пришлось добавить split-aware active stage indexing.

На локальном M21 microbench это действительно выглядело многообещающе. Для `l54.self_attn.q_proj`:

- `3` stage: `24.95 ms`, relMSE `0.0404`
- `2` stage: `16.87 ms`, relMSE `0.1171`
- `1` stage: `8.81 ms`, relMSE `0.3407`

То есть как локальный speed lever stage-drop работает, но качество падает очень быстро.

На честном `16`-window raw WikiText-2 gate для `first8_qk_gu` картина оказалась ещё важнее:

- baseline `3` stage: `PPL = 2.9487`, `385.47 tok/s`
- uniform `2` stage: `PPL = 3.4761`, `497.58 tok/s` --- dead
- role-aware `attn=3, mlp=2`: `PPL = 3.1191`, `478.75 tok/s` --- только если допустим заметный quality cost
- per-layer calibration `cos >= 0.999`: `PPL = 2.9614`, `391.47 tok/s` --- безопасный frontier для режима "почти без quality loss"
- per-layer calibration `cos >= 0.99`: `PPL = 3.5730`, `408.85 tok/s` --- dead
```

