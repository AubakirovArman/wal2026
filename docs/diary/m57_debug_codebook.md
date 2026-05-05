# M57 DEBUG CODEBOOK

## Date
2026 (exact date from git log or experiment run)

## Goal
Debug: verify codebook recon matches raw recon on multiple layers.

## Configuration
K_ATOMS=128

## Method / What was tested
See `experiments/m57_debug_codebook.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m57_debug_codebook.py`
- `experiments/m57_debug_codebook.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- Это означает: WAL-0 — это **кодек**, не **язык**. Программы не "говорят" друг с другом, не образуют предложений, не имеют синтаксиса.
- Сравнение с DRL v2: в DRL v2 тоже был i.i.d. поток (WAL-SS coverage 0.000176). Но DRL v2 имел хотя бы stage structure (12 stages). WAL-0 полностью плоский.
- **Научный вывод**: scalar greedy residual encoding на atom table **не создаёт лингвистической структуры**. Для создания языка нужна **иерархия** (vector/tensor atoms, context dependence, или constraint-based encoding).
- Следующий шаг: M57 — full 70B encode с codebook + PPL, чтобы доказать масштабируемость. Затем M58 — попытка создать structure через constraints или hierarchical atoms.


## Шаг 32. M57: Full 70B Codebook Encode + PPL — 2.7828, 6× faster!

- Full end-to-end encode всех 540 params с codebook layer.
- **PPL: 2.7828** — gap vs baseline 2.7805: **+0.08%** (+0.0023 nats).
- Сравнение: M46=2.7821, M53c=2.7858. Codebook не портит качество.
- **Время encode: 437 секунд (7.3 минуты)** — в 6× быстрее M53c (2225s) и M46 (2715s)!
- **Total unique programs: 609,643** на всю модель, avg 1129 per layer.
- **Баг found и fixed**: `codebook_recon` был проиндексирован по `unique_prog` порядку, а `program_ids` — по `sort_idx` (frequency sort). Это давало catastrophic PPL 299,002 на первом запуске. Fix: использовать `inverse` от `torch.unique` напрямую.
- **Вывод**: WAL-0 с codebook layer масштабируется на полную модель, сохраняет качество, и encode в 6× быстрее.


```

### Mention 2

```text
- Следующий шаг: M57 — full 70B encode с codebook + PPL, чтобы доказать масштабируемость. Затем M58 — попытка создать structure через constraints или hierarchical atoms.


## Шаг 32. M57: Full 70B Codebook Encode + PPL — 2.7828, 6× faster!

- Full end-to-end encode всех 540 params с codebook layer.
- **PPL: 2.7828** — gap vs baseline 2.7805: **+0.08%** (+0.0023 nats).
- Сравнение: M46=2.7821, M53c=2.7858. Codebook не портит качество.
- **Время encode: 437 секунд (7.3 минуты)** — в 6× быстрее M53c (2225s) и M46 (2715s)!
- **Total unique programs: 609,643** на всю модель, avg 1129 per layer.
- **Баг found и fixed**: `codebook_recon` был проиндексирован по `unique_prog` порядку, а `program_ids` — по `sort_idx` (frequency sort). Это давало catastrophic PPL 299,002 на первом запуске. Fix: использовать `inverse` от `torch.unique` напрямую.
- **Вывод**: WAL-0 с codebook layer масштабируется на полную модель, сохраняет качество, и encode в 6× быстрее.


```



## Extracted Metrics (from source)

- Max diff: .2e
- Mean diff: .2e
