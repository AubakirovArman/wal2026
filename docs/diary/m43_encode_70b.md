# M43 ENCODE 70B

## Date
2026 (exact date from git log or experiment run)

## Goal
M43: Apply hybrid encoder to Llama 3.3 70B and measure PPL.

## Configuration
K=128, num_steps=0

## Method / What was tested
See `experiments/m43_encode_70b.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m43_encode_70b.py`

## Notes from dev_diary_ru.md
```
- Возможные направления: per-layer adaptive K, увеличение l_max с избежанием collapse, пропуск/снижение bitrate только на ранних чувствительных слоях.

- Полный отчёт: `docs/diary/m43_70b_end_to_end_encoding.md`


```
