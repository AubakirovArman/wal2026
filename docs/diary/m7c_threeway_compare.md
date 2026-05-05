# M7C THREEWAY COMPARE

## Date
2026 (exact date from git log or experiment run)

## Goal
M7: Three-way compare — baseline vs eager-bf16 vs cached-packed.

## Configuration
See source code for full configuration.

## Method / What was tested
Runs WikiText-2 raw PPL + throughput + VRAM for:
  1. baseline bf16 (vanilla nn.Linear)
  2. eager-bf16 (route-decode once, then nn.Linear-equivalent)
  3. cached-packed deployment (current default)

Shows the speed/VRAM ceiling proof and which runtime is Pareto-optimal.

## Result
PPL evaluation.

## Artifacts
- `experiments/m7c_threeway_compare.py`

## Notes from dev_diary_ru.md
```

- После packed-побед и отрицательных Triton-результатов стало нужно заново зафиксировать общий operational picture: что у нас сегодня является лучшим deployable runtime, а что остаётся исследовательской packed веткой.
- Здесь вскрылся конкретный дефект в `experiments/m7c_threeway_compare.py`: eager-bf16 случайно получал `shape_policy_json`, из-за чего в supposedly eager compare снова появлялись медленные routed runtime layers.

После исправления compare harness и явного opt-in для shape policy картина на `16` окнах raw WikiText-2 стала такой:
```


## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
