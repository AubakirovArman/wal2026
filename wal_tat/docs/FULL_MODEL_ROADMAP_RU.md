# Маршрут WAL-TAT до полной ternary-модели

## Определение финиша

Полная модель считается готовой, когда:

1. все 28 decoder blocks и tied embedding/head имеют зафиксированный формат;
2. независимые PPL и task-benchmarks проходят общий, заранее заданный budget;
3. checkpoint воспроизводим из исходной BF16-модели и WAL;
4. codes реально упакованы, а runtime не материализует всю матрицу в BF16;
5. измерены file size, peak RAM/VRAM, prefill и decode speed.

Основная ветка: ternary `{-1,0,+1}`, logical 1.585 bit и physical Q2-g128
2.125 bpw. Binary Q1-g128 1.125 bpw начинается только после устойчивого
полного ternary recipe.

## Текущий счётчик

```text
decoder blocks:       1 / 28 complete, 27 remain
next block:           layer 24 has Q/K/V/O + 48.14453125% up + 0.5859375% gate + 1.46484375% down
major matrices:       11 / 197 accepted, 186 remain
major matrix weights: 69,230,592 / 1,720,451,072 = 4.023979%
embedding/head:       0 / 1 tied matrix
packed runtime:       0% implemented
```

## Система quality budgets

`1.02` остаётся только диагностическим micro-gate. Он появился как наша
эвристика «не принимать локальный шаг хуже teacher более чем на 2% NLL» и не
является внешним стандартом.

| Уровень | Данные | Назначение |
|---|---|---|
| Micro transaction | маленькие frozen suites | быстрый rollback, limit 1.02 |
| Block development | recovery domains | выбор optimizer/objective |
| Independent block audit | полностью отложенные domains | проверка generalization |
| Cumulative model audit | один неизменяемый большой harness | общий NLL/PPL budget |
| Task audit | reasoning/code/instruction/knowledge | поведенческое качество |
| Runtime audit | packed artifact | память и скорость |

Для эксперимента задаются несколько итоговых кривых: например `+3%`, `+5%` и
`+10% NLL` всей модели. Это проектные цели, а не числа из внешней статьи.
Пропорциональная guide-линия для coverage `c`:

```text
allowed_ratio(c, B) = 1 + B*c
```

Она не считается физическим законом: block errors взаимодействуют нелинейно и
могут быть восстановлены end-to-end QAT. Это консервативный индикатор, который
не позволяет ошибочно назвать успешным рецепт, расходующий весь budget на
первых blocks.

Если цель сформулирована как `+5% PPL`, формула другая, поскольку
`PPL=exp(NLL)`:

```text
allowed_NLL_ratio(domain) = 1 + c*log(1.05)/teacher_NLL(domain)
```

Поэтому в результатах всегда явно указывается, ограничивается NLL или PPL.

## Выполненные фазы

### Фаза 0 — транзакционный механизм

- g128 ternary codes и scales;
- causal activation/Fisher ranking;
- exact commit/rollback;
- hash-chained WAL;
- multi-domain NLL gate;
- matched controls.

### Фаза 1 — MLP compensation

- `down_proj`, `up_proj`, `gate_proj` layer 27 доведены до 100%;
- triadic SwiGLU window связывает gate/up/down;
- scale-only compensation проходила там, где candidate-only откатывалась;
- adaptive transaction sizes позволили пройти локальные frontiers.

### Фаза 2 — attention и полный block 27

- GQA-aware V/O окна конвертировали `v_proj` и `o_proj`;
- GQA-aware K/Q окна конвертировали `k_proj` и `q_proj`;
- все семь крупных матриц layer 27 имеют только ternary codes;
- diverse fixed-code recovery улучшил независимый audit с ~1.05 до максимум
  1.01404.

### Фаза 3 — proxy-code recovery и sensitivity map

- hard forward остаётся точным `{-scale, 0, +scale}`;
- гладкая proxy-переменная используется только для backward;
- layer 27 после proxy recovery проходит два audit suite с ratios ниже `1`;
- два независимых sensitivity scan дали одинаковый порядок blocks
  (Spearman `1.0`);
- следующим выбран наименее чувствительный remaining `layer 24`.

## Текущая фаза 4 — закончить block 24

Q/K/V/O layer 24 приняты и совместно с layer 27 прошли audit-v3/v4. Через
sensitivity-ranked транзакции также приняты 48.14453125% `up_proj`, первые
0.5859375% `gate_proj` и 1.46484375% `down_proj`. Текущее покрытие
`4.023979%`, условная `+5% NLL` guide равна `1.00201199`, худший измеренный
ratio равен `1.000692`.

Masked proxy recovery теперь поддерживает этот частичный block без ложной
тернаризации оставшихся BF16-групп. На текущем frontier он сохранил coverage и
uncommitted master weights, сменил только два уже принятых ternary-кода и
перенастроил их shared scales. Независимый worst ratio снизился с
`1.001531213` до `1.000654804`; при том же gate `1.00195057` доступный
headroom вырос более чем втрое. Это позволяет снова испытывать более крупные
атомы вместо бесконечной последовательности `1/1024`.

Практическая проверка подтвердила пользу recovery: `+6.25% up_proj` лишь
немного не прошёл неизменённый development gate, а автоматически уменьшенный
`+3.125%` атом прошёл development, fresh reload и оба holdout. Он добавил
393,216 ternary weights одним commit и поднял `up_proj` до `35.64453125%`;
лучшим оказался linked BF16 `down/gate` arm с worst ratio `1.000731191`.
Повторный атом того же размера также прошёл оба holdout, добавил ещё 393,216
weights и поднял `up_proj` до `38.76953125%`; linked arm снова выиграл с worst
ratio `1.000755065`. Третий атом прошёл, поднял coverage до `41.89453125%`,
но candidate-only впервые обошёл linked arm на holdout (`1.001118535` против
`1.001146107`), поэтому BF16-компенсация не закрепляется без проверки.
Следующий masked proxy recovery сохранил все discrete codes и все
uncommitted BF16/scale элементы, изменил 467,770 committed scales и улучшил
audit-v3 worst до `1.000297213` без изменения coverage.
Из этого frontier принят `down_proj +0.390625%`: 49,152 новых ternary weights,
оба holdout пройдены, linked arm выиграл с worst ratio `1.000391478`, а
`down_proj` достиг `1.07421875%`.
Затем принят `gate_proj +0.1953125%`: 24,576 новых weights, оба holdout
пройдены, linked arm выиграл с worst `1.000349834`, coverage gate достиг
`0.390625%`.
Следующий round-robin атом `up_proj +3.125%` добавил 393,216 weights и прошёл
оба holdout; linked arm выиграл с worst `1.000441498`, coverage up достиг
`45.01953125%`.
Следующий down-атом добавил 49,152 weights, прошёл оба holdout и поднял
`down_proj` до `1.46484375%`; linked arm worst равен `1.000459295`.
Следующий gate-атом добавил 24,576 weights и поднял `gate_proj` до
`0.5859375%`. Оба arm-а прошли два holdout; candidate-only выиграл с worst
`1.000469347` против `1.000483896` у linked BF16-окна.
Следующий up-атом добавил 393,216 weights и поднял `up_proj` до
`48.14453125%`. Linked arm выиграл с worst `1.000691962` против
`1.000778586` у candidate-only.

Остались три MLP-матрицы. Component ablation показал:

1. `up_proj` — наименее вредная отдельная матрица;
2. `down_proj` и `gate_proj` сильнее ухудшают SQuAD;
3. пары MLP дают нелинейно больший ущерб.

One-shot 100% `up_proj` после hard proxy + continuous recovery дошёл на
audit-v3 до `0.998760 / 1.004714 / 0.978121`, но не прошёл заданный gate.
Incremental WAL нашёл безопасный путь: после 12.5% последовательные шаги
3.125%, 1.5625% и 0.78125% довели `up_proj` до 28.125% и прошли два audit
suite, тогда как крупные очередные шаги 12.5% и 6.25% откатились. Следующий
контроллер должен выбирать размер атома автоматически из holdout margin и
уменьшать его при rollback.

Добавлен новый `candidate_bf16_mlp` arm: candidate остаётся hard ternary, а
связанные ещё не конвертированные `down/gate` получают локальные непрерывные
BF16 окна с точным snapshot/rollback. В matched ablation он улучшил audit-v3
SQuAD с `1.001575` до `1.001471`, не увеличив ternary coverage `down/gate`.
Один linked шаг `+1.5625%` откатился при `1.002169`, но два последовательных
шага по `+0.78125%` достигли того же coverage и прошли оба holdout. Это
подтверждает path-dependence уже для новой компенсационной схемы.

После этого два linked-атома по `+0.390625%` и один адаптивно увеличенный атом
`+0.78125%` также прошли development, fresh reload и оба holdout-аудита. При
сужении audit-margin следующий атом снова уменьшается до `+0.390625%`.

Реальный adaptive campaign runner теперь выполняет recovery, fresh reload,
параллельные audit-v3/v4 для каждого прошедшего arm-а, holdout-selection,
coverage-aware resizing и crash-safe cleanup. Первый автоматический gate-шаг
поднял `gate_proj` до `0.1953125%`; linked BF16-down arm выиграл holdout у
candidate-only (`1.001475638` против `1.001515298`).

Для `down_proj` добавлен отдельный column-structured selector: один столбец
g128-групп соответствует одному входному 128-канальному SwiGLU-блоку и
однозначно задаёт локальные строки компенсации в `up/gate`. Первый атом из 96
групп (`12,288` weights, `0.09765625%` матрицы) прошёл оба holdout. В этом
случае candidate-only оказался немного лучше BF16-компенсации (`1.001527907`
против `1.001550445`) и стал новым frontier. Следующий атом был увеличен в
четыре раза до 384 групп и также прошёл оба holdout; на нём BF16-компенсация
уже выиграла (`1.001487769` против `1.001539105`). Контроллер сузил следующий
размер с `0.390625%` до `0.1953125%` из-за небольшого audit-margin. Этот
следующий атом также прошёл оба holdout, снова с небольшим преимуществом
BF16-компенсации (`1.001531213` против `1.001537834`), после чего контроллер
вернул следующий размер к минимуму `0.09765625%`.

Критерий перехода: принять все три MLP, получить второй полный block и
повторить неизменяемый cumulative audit.

## Фаза 5 — пройти decoder

Blocks выбираются по измеренной чувствительности, не по номеру. После каждого
commit sensitivity пересчитывается, потому что frontier изменился. Всегда
публикуются оба числа:

```text
структурно converted blocks / 28
independently accepted blocks / 28
```

Нельзя считать частичный tensor полноценным block и нельзя ослаблять audit
ради процента покрытия.

## Фаза 6 — tied embedding/LM head

Embedding/head содержит 311,164,928 weights и одновременно является входной
таблицей и output classifier. Для него нужен отдельный frequency-balanced
token suite, logit distillation и, возможно, mixed Q2/Q4 budget. Любое спасение
Q4 публикуется в true average bpw.

## Фаза 7 — full-model validation

- C4/Wiki/code и независимые языковые corpora;
- MMLU/ARC/HellaSwag/PIQA;
- GSM8K/MATH;
- HumanEval/MBPP;
- instruction following, JSON и tool calling;
- русский/казахский и long-context tests;
- несколько seeds и confidence intervals;
- BF16, INT4 и сильный Q2 PTQ baselines.

## Фаза 8 — packed export и llama.cpp

До фиксации representation менять `llama.cpp` преждевременно. После фиксации:

1. exporter пишет packed two-bit codes и FP16 g128 scales;
2. GGUF tensor type совпадает с layout;
3. loader не распаковывает полную BF16-матрицу;
4. CPU/CUDA/Metal dequant/GEMM paths проходят bit-exact tests;
5. измеряются real file size, resident memory и tok/s.

Если останется единый Q2-g128 layout, можно адаптировать существующий близкий
путь. Если внутри tensor будут Q2/Q4 blocks, residual plane или новый decoder,
потребуются новый tensor type, quantizer, loader и kernels.

## Фаза 9 — binary

От устойчивой ternary-модели нули постепенно переводятся в знаковые states.
Цель Q1-g128: `1 + 16/128 = 1.125 bpw`. Binary checkpoint и его quality target
всегда отделяются от ternary результата.
