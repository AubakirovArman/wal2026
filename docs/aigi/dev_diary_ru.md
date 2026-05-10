# AIGI Dev Diary

## 2026-05-10 — AIGI 1.0 проект выделен из WAL

### Цель

Начать отдельный AIGI слой поверх WAL: не как новый hype-claim, а как проверяемую систему накопления памяти.

### Гипотеза

`Verified Memory Accumulation`: система может становиться полезнее после deployment, если каждое новое знание проходит цикл `experience → memory candidate → tier selection → compile → verification → commit/rollback`.

### Реализация M679

- Добавлен Python package `aigi`.
- Добавлен `AIGISystem`.
- Добавлен `MemoryCompiler`.
- Добавлены memory tiers: `wal_recipe`, `retrieval`, `refusal`, `tool`, `reject`.
- Добавлен WAL recipe ledger.
- Добавлена retrieval overlay для реального ответа после commit.
- Добавлены verification gates.
- Добавлены runtime logs в JSONL.

### Ограничение

MVP не редактирует реальные веса. Tier `wal_recipe` сохраняет WAL-compatible recipe artifact и обслуживается retrieval overlay до подключения настоящего backend для weight edits.

### Тестовая политика

Каждый AIGI шаг должен иметь:

- positive test;
- negative test;
- результат в `docs/aigi/test_log.md`;
- machine-readable log в `logs/aigi/aigi_steps.jsonl`;
- milestone result JSON в `experiments/*_results.json`.
