# M419 — Comparison Matrix

## Описание эксперимента
WAL Project — MIT License Copyright (c) 2026 WAL Research Team

## Исходный файл
`m419_comparison_matrix.py`

## Результаты (полные данные)

- **methods**:
  - **Dense+LoRA**:
    - **accuracy**: `0.848`
    - **size_mb**: `16000`
    - **latency_ms**: `85`
    - **train_time_min**: `45`
  - **RAG-only**:
    - **accuracy**: `0.85`
    - **size_mb**: `512`
    - **latency_ms**: `120`
    - **train_time_min**: `0`
  - **WAL-weights**:
    - **accuracy**: `0.923`
    - **size_mb**: `8`
    - **latency_ms**: `45`
    - **train_time_min**: `6`
  - **WAL-hybrid**:
    - **accuracy**: `0.957`
    - **size_mb**: `520`
    - **latency_ms**: `55`
    - **train_time_min**: `6`
- **best_accuracy**: `WAL-hybrid`
- **most_efficient**: `WAL-weights`
- **pass**: ✅ PASS

## Анализ

- **Модуль**: M419
- **Название**: Comparison Matrix
- **Дата выполнения**: 2026-04-20 – 2026-05-06
- **Статус**: ✅ Успешно завершён

---

*Запись сгенерирована автоматически.*
