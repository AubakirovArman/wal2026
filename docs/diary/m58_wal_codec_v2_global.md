# M58 WAL CODEC V2 GLOBAL

## Date
2026 (exact date from git log or experiment run)

## Goal
M58: WAL-0 Codec v2 — Global atoms + Global codebook + Bit-packed IDs.

## Configuration
K_ATOMS=128, KMEANS_ITERS=5, num_steps=0

## Method / What was tested
Two-pass architecture:
  Pass 1: Collect samples from all layers, build global atom table.
  Pass 2: Encode all layers with global atoms, build global codebook,
          apply precomputed recon, measure PPL and compression.

## Result
PPL evaluation.
Has PASS/FAIL asserts

## Artifacts
- `experiments/m58_wal_codec_v2_global.py`
- `experiments/m58_wal_codec_v2_global.log`

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

