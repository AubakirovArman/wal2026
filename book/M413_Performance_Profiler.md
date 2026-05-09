# M413 — Performance Profiler

## Описание эксперимента
WAL Project — MIT License Copyright (c) 2026 WAL Research Team

## Исходный файл
`m413_performance_profiler.py`

## Результаты (полные данные)

- **stages**: 8 items
  - {'stage': 'recipe_parse', 'duration_ms': 45, 'memory_mb': 12
  - {'stage': 'dag_build', 'duration_ms': 120, 'memory_mb': 18}
  - {'stage': 'weight_compile', 'duration_ms': 2100, 'memory_mb'
  - {'stage': 'ci_exact', 'duration_ms': 890, 'memory_mb': 32}
  - {'stage': 'ci_para', 'duration_ms': 1200, 'memory_mb': 35}
  - ... и ещё 3
- **total_ms**: `7980`
- **peak_mem_mb**: `64`
- **bottleneck**: `inference_load`
- **pass**: ✅ PASS

## Анализ

- **Модуль**: M413
- **Название**: Performance Profiler
- **Дата выполнения**: 2026-04-20 – 2026-05-06
- **Статус**: ✅ Успешно завершён

---

*Запись сгенерирована автоматически.*
