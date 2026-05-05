# M6O PALETTE HOTNESS PROFILE

## Date
2026 (exact date from git log or experiment run)

## Goal
Run and evaluate m6o_palette_hotness_profile.

## Configuration
l_max=12, iters=20, threshold=0.0

## Method / What was tested
See `experiments/m6o_palette_hotness_profile.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m6o_palette_hotness_profile.py`

## Notes from dev_diary_ru.md
```

- После шага 12j нужен был уже не новый blind tweak, а более содержательная проверка: есть ли вообще у больших local palettes достаточно сильный hot-prefix, чтобы его имело смысл stage-ить отдельно?
- Для этого появился `experiments/m6o_palette_hotness_profile.py`.

Профиль оказался интересным.
```
