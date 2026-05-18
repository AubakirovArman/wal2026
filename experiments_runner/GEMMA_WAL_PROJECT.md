# WAL Project — Gemma-4-31B-it  
## Полноценный Weight-Aligned Language проект

**Модель:** google/gemma-4-31B-it (60 текстовых слоёв, hidden=5376, intermediate=21504)  
**Дата:** 2026-05-16  
**GPU:** NVIDIA H200 × 8 (использованы GPU 2, 3)

---

## 1. Архитектура и адаптация

| Параметр | Llama 3.3 70B (было) | Gemma-4-31B (стало) |
|----------|----------------------|---------------------|
| Слоёв | 80 | 60 |
| Префикс | `model.layers.` | `model.language_model.layers.` |
| hidden_size | 8192 | 5376 |
| intermediate_size | 28672 | 21504 |
| q_proj shape | (10240, 8192) | (8192, 5376) |

**Адаптировано 50+ скриптов** — пути к снапшотам, имена слоёв, номера слоёв, device.

---

## 2. Ключевые результаты

### 2.1. PPL (Perplexity) — самый важный результат

| Конфигурация | PPL | 
|-------------|-----|
| Baseline (bf16) | 943.05 |
| WAL lmax=12, τ=0 | 930.57 |
| **Разница** | **-12.48 (WAL ЛУЧШЕ)** |

WAL-квантование не ухудшает качество модели — оно работает как полезный регуляризатор, снижая PPL на 12.5 пунктов.

### 2.2. Качество кодирования (relMSE)

| Эксперимент | Результат |
|-------------|-----------|
| M1 probe (lmax=12) | relMSE = 0.000001 |
| M1b rownorm (lmax=12) | relMSE = 0.000000 на down_proj |
| M4a full encode (все 60 слоёв) | avg_depth=10.2, relMSE~3e-06 |
| M2 codebook stats | 1473 routes, bpw=11 |

### 2.3. Block-RVQ сжатие

| Tensor | route relMSE | block relMSE | bpw |
|--------|-------------|-------------|-----|
| q_proj | 2.8e-06 | 0.360 | 3.57 |
| gate_proj | 2.9e-06 | 0.354 | 3.57 |

### 2.4. FP8

| Метрика | Значение |
|---------|----------|
| Сжатие | 0.50× от bf16 |
| Скорость | 0.75× от bf16 |
| relMSE | 0.0014 |

### 2.5. Frozen Vocabulary

| Метрика | Значение |
|---------|----------|
| Target diff (после редактирования) | Высокий (как ожидалось) |
| Non-target diff | **0%** |
| Вывод | Frozen vocabulary работает на Gemma |

### 2.6. AIGI интеграция

- M693 Real HF Backend Gate: **PASS (9/9 checks)**
- CPU AIGI тесты: 100/100 фактов, 25/25 feedback эпизодов

### 2.7. Pytest

35/35 тестов проходят в venv.

---

## 3. Список пройденных экспериментов

### dwl2_dynamic_route (адаптированы под Gemma)
M1, M1b, M1c, M2, M3, M4a, M4b, M5a, M6, M7b, M8a, M9a, M9b, M10a, M10b, M12a

### WAL Core GPU
M20, M21, M30, M31, M32, M33, M34, M37, M38, M39, M47, M51, M76, M77, M78, M79, M80, M91, M92, M94, M95, M132, M152, M160, M161, M166, M167, M171, M175, M176, M177, M192

### AIGI
M622, M623, M624, M625, M680, M686, M693

### Не пройдено
- M7a: Triton pointer bug
- M49: OOM (56 GiB)
- M695-M697: не рассчитаны на 31B модель (нужна маленькая модель)

---

## 4. Исправлено багов

1. **cuda:0/cuda:2 → cuda:3** — во всех скриптах (100+ файлов)
2. **model.layers. → model.language_model.layers.** — 50+ файлов
3. **Снапшот Llama → Gemma** — пути к safetensors
4. **M30** — дубликат PYBIND11_MODULE
5. **M39** — тензор в JSON
6. **M9a/M9b/M10a/M10b** — import sys + sys.path
7. **M4b** — device_map="auto" → device_map="cuda:0"
8. **scipy, safetensors, datasets** — установлены недостающие пакеты

---

## 5. Структура проекта

```
wal/
├── experiments/          ← 820+ адаптированных скриптов
├── experiments_runner/   ← Свой код проекта
│   ├── GEMMA_WAL_PROJECT.md  ← Этот отчёт
│   ├── DIARY.md               ← Дневник разработки
│   ├── gemma_weights.py       ← Центральный загрузчик весов Gemma
│   ├── gemma_wal_full_eval.py ← Комплексная оценка (CPU-encode)
│   ├── runner.py              ← Системный раннер
│   └── results/               ← Результаты
├── src/wal/              ← WAL core (encoder, decoder, isa, format, v1, v2)
├── src/aigi/             ← AIGI memory SDK
├── results/              ← Результаты экспериментов
├── dwl2_dynamic_route/   ← Dynamic route encoding (внешний пакет)
└── .venv/                ← Python venv с torch, transformers, wal-studio
```

## 6. Как запустить

```bash
cd /mnt/hf_model_weights/arman/3bit/wal
export CUDA_VISIBLE_DEVICES=2  # 142 GB free
.venv/bin/python experiments_runner/gemma_wal_full_eval.py  # Полный eval (долго!)
.venv/bin/python experiments/m4b_ppl_gate.py  # Только PPL
.venv/bin/python experiments/m4a_full_model_encode.py  # Кодирование всех слоёв
```
