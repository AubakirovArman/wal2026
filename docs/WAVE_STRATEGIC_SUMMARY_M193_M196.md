# Стратегический вывод: Wave-анализ M193–M196

**Дата:** 2026-04-20

## Что доказано

1. **Mixed training обязателен** — предотвращает catastrophic forgetting
2. **Wave regularization реально помогает** — λ=0.1 survival 0→2/10 без PPL loss
3. **Hadamard adaptive K лучше uniform** — PPL +0.038 vs +0.062
4. **Synthetic WaveRisk нельзя напрямую переносить** — нужна real calibration

## Что ещё не доказано

1. WaveRiskScore готов для production safety
2. Survival можно поднять выше 2/10 без вреда PPL
3. Adaptive K даст размерный выигрыш после честного bit accounting
4. WAL v2 лучше обычного quantization baseline

## Новая архитектура WAL pipeline

```
WAL v1 = стабильный base checkpoint
WAL v2 = Hadamard adaptive checkpoint
LoRA = основной edit format
Wave-LoRA = основной безопасный training mode
WaveRisk = семейство признаков (не одна формула)
Gumbel-WAL = research track
```

## Следующие шаги (M195b–M200)

| Exp | Goal | Priority |
|-----|------|----------|
| M195b | Hadamard adaptive K + k-means | HIGH |
| M196b | Wave-LoRA на 50–100 facts | HIGH |
| M196c | Penalty schedule (warmup/cosine) | MEDIUM |
| M193b | Learned risk model (100+ runs) | MEDIUM |
| M200 | WAL v2 end-to-end demo | HIGH |

## Главный стратегический вывод

> **WAL v2 должен идти через Hadamard adaptive checkpoint.
> WAL+LoRA должен идти через Wave-Regularized LoRA.
> Safety должен идти не через одну формулу WaveRisk, а через learned risk model на реальных LoRA runs.**
