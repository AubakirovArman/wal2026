# M40 END TO END PPL

## Date
2026 (exact date from git log or experiment run)

## Goal
M40: End-to-end PPL benchmark on Llama 3.1 8B with hybrid encoder.

## Configuration
K=16, batch=512, iters=12, block_size=4, threshold=0.0

## Method / What was tested
Uses real WikiText-2 dataset. Evaluates baseline PPL first, then replaces
all linear layer weights with hybrid-encoded versions and re-evaluates.

## Result
PPL evaluation.

## Artifacts
- `experiments/m40_end_to_end_ppl.py`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text

## Шаг 18. M43: End-to-End Encoding Llama 3.3 70B — Scalar DRL v2 лучший viable путь, VRE катастрофа

- После успеха hybrid encoder на Block-RVQ reconstructed весах (M39) и 8B end-to-end (M40) был запущен полный sweep на оригинальных весах Llama 3.3 70B.
- **Baseline PPL**: 2.40 (WikiText-2, первые 6656 токенов).
- **Лучший scalar-only результат**: PPL 4.29 (K=128, lmax=8, 540 params encoded, 183 skipped). Δ = +79%.
- **Лучший scalar variant со skip layer 0**: PPL 4.26 (пропускаем все параметры layer 0). Δ = +78%.

### VRE paradox
- VRE (vector route encoder, 4×4 blocks, cb=512, lmax=8) на reconstructed Block-RVQ весах показывал relMSE 0.001 и beat scalar.
- Но на оригинальных весах 70B VRE даёт **катастрофу**:
  - Single layer (layer 0 all params): PPL 7.33
  - Single selective layer (layer 0 q/k/v/gate/up): PPL 30.86
  - Multi-layer hybrid: PPL >7000
- При этом per-layer метрики VRE почти идеальны: relMSE 0.001, output correlation 0.9992, spectral norm идентичен.

### Ключевое открытие: знаки не важны, структура ошибки важна
- Scalar меняет **92.5% знаков** на q_proj layer 0, но PPL остаётся ~2.40.
```



## Extracted Metrics (from source)

- PPL: .4
- PPL: .4
- PPL: .4
- PPL: .4
