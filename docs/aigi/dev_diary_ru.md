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

## 2026-05-10 — M680-M683 AIGI memory-loop hardening

### Цель

Перевести AIGI слой от SDK skeleton к минимальному проверяемому memory-loop набору: массовое обучение фактам, отказ от плохой памяти, deterministic tier routing и реальный rollback.

### Реализация

- M680: 100 synthetic facts проходят цикл `ask → propose → compile → commit → ask`.
- M681: 20 плохих memory candidates отклоняются без изменения защищённого состояния.
- M682: 9 routing cases проверяют `stable_fact`, `fact_update`, `hard_fact`, `unsafe_request`, `procedure`, `zero_confidence`.
- M683: rollback удаляет последний WAL recipe, восстанавливает предыдущую retrieval memory и корректно падает на пустой истории.

### Результаты

- M680: `100/100` facts passed, tiers `wal_recipe=50`, `retrieval=50`.
- M681: `20/20` bad memories rejected.
- M682: `9/9` routing checks passed.
- M683: `8/8` rollback checks passed.

### Ограничение

Это всё ещё не autonomous AGI и не semantic weight editing. `wal_recipe` остаётся WAL-compatible artifact plus retrieval overlay до подключения настоящего backend для изменения весов.

## 2026-05-10 — M684-M687 feedback contracts

### Цель

Добавить поверх memory-loop первый слой проверяемого feedback learning: контракт поведения, извлечение урока из опыта, verified commit и rollback при нарушении контракта.

### Реализация

- M684: `BehavioralContract` и verifier для `must_answer`, `must_not_answer`, `must_refuse`.
- M685: `LessonExtractor` превращает feedback experience в `MemoryCandidate`.
- M686: `VerifiedLearningLoop` выполняет `experience → candidate → compile → commit → contract check`.
- M687: contract-gated rollback откатывает tentative update, если контракт после commit нарушен.

### Результаты

- M684: `4/4` contract checks passed.
- M685: `8/8` extraction cases passed.
- M686: `25/25` verified feedback episodes passed.
- M687: `5/5` contract rollback checks passed.

### Ограничение

Это deterministic SDK feedback loop. Он не доказывает автономное обучение, не заменяет внешний verifier и не выполняет реальное изменение весов.

## 2026-05-10 — M689-M692 governance layer

### Цель

Добавить контроль цены изменения перед подключением настоящего `wal_recipe` backend: budget, risk ledger, regression suite и decision report.

### Реализация

- M689: `MemoryChangeBudget` и `MemoryBudgetEvaluator` оценивают overwrite, confidence, answer length, tier risk и contract coverage.
- M690: `RiskLedger` записывает active, rejected и rolled-back debt.
- M691: `ContractRegressionSuite` проверяет защищённые факты после feedback update.
- M692: `CommitDecisionReport` фиксирует compile gates, budget status, risk score, regression status и финальное решение.

### Результаты

- M689: `7/7` budget checks passed.
- M690: `8/8` risk-ledger checks passed.
- M691: `6/6` regression checks passed over `10` protected contracts.
- M692: `7/7` decision-report checks passed.

### Ограничение

Это governance для SDK loop, не реальный optimizer и не semantic weight editing. Следующий технологический риск — подключить backend так, чтобы эти gates реально блокировали опасные изменения веса.

## 2026-05-10 — M693 real HF backend gate

### Цель

Перейти от символического fallback к реальному inference backend внутри AIGI SDK.

### Реализация

- Добавлен `HuggingFaceTextBackend` поверх `transformers.AutoModelForCausalLM` и `AutoTokenizer`.
- `AIGISystem.ask()` теперь использует injected model backend, если retrieval/refusal memory не нашла ответ.
- M693 загружает `Qwen/Qwen2.5-0.5B-Instruct`, получает базовый ответ из `hf_model`, затем проверяет memory overlay и rollback.

### Результат

- M693: `9/9` checks passed.
- Модель: `Qwen/Qwen2.5-0.5B-Instruct`.
- Cache: локальный `.hf_cache/`, потому что общий `/mnt/hf_model_weights/.hf_cache` недоступен на запись.

### Ограничение

Это первый реальный inference backend, но ещё не LoRA training и не semantic weight editing. Следующий реальный шаг — M694: настоящий LoRA/adapter update на small model под теми же gates.
