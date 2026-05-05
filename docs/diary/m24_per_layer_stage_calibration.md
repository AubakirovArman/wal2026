# M24 PER LAYER STAGE CALIBRATION

## Date
2026 (exact date from git log or experiment run)

## Goal
M24: per-layer calibrated stage cap.

## Configuration
block_size=32

## Method / What was tested
Idea: at calibration time, push N batches of real activations through each
PackedGroupedBlockRVQLinear at full stages (k=stages_per_split max) and at
candidate reduced stages. For each layer pick the smallest k whose output
cosine similarity stays above a threshold. Saves to JSON for use at eval.

This is the "lingvistic importance" knob — every layer gets its own
'how many phonemes are needed to still be understood' setting.

## Result
Encode test.

## Artifacts
- `experiments/m24_per_layer_stage_calibration.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

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


## Known Results (from project context)

**Result:** Per-layer stage calibration.

**Notes:** Different layers have different optimal calibration parameters.
