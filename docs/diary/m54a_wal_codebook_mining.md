# M54A WAL CODEBOOK MINING

## Date
2026 (exact date from git log or experiment run)

## Goal
M54a: GPU-native WAL-0 program codebook mining on real 70B weights.

## Configuration
K_ATOMS=128, KMEANS_ITERS=5

## Method / What was tested
Goal: encode a full layer, find unique programs, measure vocabulary and entropy.
Everything stays on GPU.

## Result
Encode test.

## Artifacts
- `experiments/m54a_wal_codebook_mining.py`
- `experiments/m54a_wal_codebook_mining.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text



## Шаг 28. M54a: WAL-0 Codebook Mining — язык существует!

- GPU-native codebook mining на layer 40 self_attn.o_proj (67M weights).
- K=128, lmax=2, per-row normalization.
- **Результат**: всего **1,079 unique programs** из 67,108,864 весов!
- **Vocabulary ratio: 0.0016%** — в 1000× лучше, чем если бы все программы были уникальными.
- **Entropy: 9.13 bits** (max 10.08). Эффективно ~9 бит на программу.
- **Top-1024 coverage: 99.98%** — почти все весы покрываются 1024 программами.
- **Codebook table size: 4,316 bytes** — ничтожно маленький.
- Mining time: **0.088s** на GPU.
- Сравнение с DRL v2: DRL давал ~1500 unique routes на тензор при L_max=12. WAL-0 даёт ~1080 при lmax=2 — и качество в 200× лучше.
- **Вывод**: WAL-0 программы обладают огромной повторяемостью. Это доказывает, что "язык" реально существует — мы можем построить vocabulary из ~1000 слов, которыми выражаются 67M весов.
- Следующий шаг: M54b — pack programs в compact IDs + decode directly from codebook.


```



## Extracted Metrics (from source)

- Time: .1
- Time: .3
