# M56A WAL GRAMMAR ANALYSIS

## Date
2026 (exact date from git log or experiment run)

## Goal
M56a: WAL-0 grammar analysis — structure in the program stream.

## Configuration
K_ATOMS=128, KMEANS_ITERS=5

## Method / What was tested
Analyze n-grams, spatial correlation, and predictability of program sequences.
All GPU-native.

## Result
Encode test.

## Artifacts
- `experiments/m56a_wal_grammar_analysis.py`
- `experiments/m56a_wal_grammar_analysis.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- Следующий шаг: M56 — grammar analysis программного потока. Нужно понять, есть ли структура помимо частот.


## Шаг 31. M56a: Grammar Analysis — WAL-0 программы = i.i.d. поток

- GPU-native grammar analysis на layer 40 o_proj.
- **Униграммная энтропия: 9.298 bits**
- **Условная энтропия bigram: 9.288 bits** — предсказуемость всего 0.010 bits (0.1%)!
- **Spatial autocorrelation: 0.014 (row), 0.006 (col)** — почти ноль.
- **Repeat rate: 0.21% (row), 0.19% (col)** — соседние веса почти никогда не имеют одинаковую программу.
- **Unique bigrams: 991,885 / 1,225,449 возможных (80.9%)** — почти все пары уникальны.
- **Вывод: WAL-0 программы образуют почти идеально независимый поток**. Нет грамматики, нет spatial structure, нет n-gram patterns.
- Это означает: WAL-0 — это **кодек**, не **язык**. Программы не "говорят" друг с другом, не образуют предложений, не имеют синтаксиса.
- Сравнение с DRL v2: в DRL v2 тоже был i.i.d. поток (WAL-SS coverage 0.000176). Но DRL v2 имел хотя бы stage structure (12 stages). WAL-0 полностью плоский.
- **Научный вывод**: scalar greedy residual encoding на atom table **не создаёт лингвистической структуры**. Для создания языка нужна **иерархия** (vector/tensor atoms, context dependence, или constraint-based encoding).
- Следующий шаг: M57 — full 70B encode с codebook + PPL, чтобы доказать масштабируемость. Затем M58 — попытка создать structure через constraints или hierarchical atoms.


```

