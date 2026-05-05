# Стратегический вывод: Wave-анализ M186–M192

**Дата:** 2026-04-20

## Главная формула

> **Волны живут в continuous weight space и в LoRA delta, но почти не живут в уже дискретизированных WAL-programs.**

## Что подтвердилось

### ✅ Веса по глубине имеют wave-структуру (M186)
- Period-16/32 паттерны
- Q/K нормы уменьшаются к поздним слоям
- V/O и MLP нормы растут к поздним слоям

### ✅ Wave-risk полезен как safety signal (M188)
- Scale edit'а доминирует над rank
- Module sensitivity: `gate_proj > q_proj > v_proj`
- Spectral norm — primary risk signal

### ✅ Gumbel-WAL + wave regularization даёт сигнал (M192)
- λ=0.1: task loss -2%, spectral norm -20%
- λ=1.0: destabilizes optimization
- Sweet spot существует

### ✅ Phase coherence подтверждает устойчивость amplitude features (M189)
- top1/top10 energy, entropy — phase-invariant (<0.1% change)
- Spectral norm — phase-sensitive (60% change)
- Amplitude features robust, norm captures structural risk

## Что не подтвердилось

### ❌ Program-wave отсутствует (M187)
- WAL program frequencies почти плоские
- K-means дискретизация сглаживает depth-wave

### ❌ Wave-guided K-budget провалился (M190)
- Adaptive K дал PPL 6.02 vs uniform K=256 PPL 4.71
- Risk formula не дискриминирует
- Raw-WAL без Hadamard неправильная база

### ❌ Post-hoc wave regularization слабая (M191)
- Нужна интеграция в training loop

## Правильная архитектура

```
raw weights / LoRA delta
→ wave analysis
→ safety / budget / regularization decision
→ WAL encode or LoRA train
```

## Wave-анализ — это слой управления WAL, не новый encoder

Нужен для:
1. Диагностики опасных слоёв
2. Оценки риска LoRA edit'ов
3. Регуляризации обучения
4. Ловли collapse
5. Возможно, распределения K/C budget

Не нужен для:
- Замены atom_id на wave_id
- Поиска смысла в atom_id

## Production stack (обновлённый)

```
Base:     Hadamard-WAL K=256 checkpoint
Edit:     LoRA overlay (не WAL patch)
Runtime:  WALCachedLinear + LoRA
Safety:   spectral norm + module-specific WaveRisk + PPL gate
Training: Gumbel-WAL + мягкая wave regularization (λ~0.1)
```
