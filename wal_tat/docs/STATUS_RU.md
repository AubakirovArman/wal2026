# Что такое WAL-TAT сейчас

## Короткий ответ

WAL-TAT уже переводит настоящие матрицы Qwen3-1.7B в жёсткий ternary
`{-scale, 0, +scale}`, восстанавливает качество hard-forward QAT и принимает
результат только после проверки на отложенных данных. Это рабочий
исследовательский процесс, но ещё не готовая полностью 1.58-bit модель.

Теперь отдельно ведутся strict-ternary и mixed low-bit frontier:

| Единица | Strict ternary | Mixed low-bit |
|---|---:|---:|
| Полные decoder blocks | 1 / 28 (`layer 27`) | 5 / 28 (`27`, `25`, `24`, `23`, `22`) |
| Полные крупные матрицы | 11 / 197 | 35 / 197 |
| Ternary weights | 76,673,024 (`4.456565%`) | 81,142,144 (`4.716330%`) |
| Q4-g128 weights | 0 | 71,191,680 (`4.137966%`) |
| Q8-g128 weights | 0 | 99,324,416 (`5.773161%`) |
| Все low-bit weights | 76,673,024 | 251,658,240 (`14.627457%`) |
| Осталось high precision | 1,643,778,048 | 1,468,792,832 (`85.372543%`) |

Strict-ветка по-прежнему не называет `layer 24` полным ternary-блоком. Mixed
artifact честно завершает его как Q2/Q4-блок: attention полностью Q2, а три
MLP-матрицы вместе содержат `40.228950%` Q2 и `59.771050%` Q4 при среднем
`3.320421 bpw`. По всему блоку это `55.171712%` Q2, `44.828288%` Q4 и
`3.021566 bpw`.

Следующий accepted artifact полностью закрывает `layer 23`. После первого
reverse-compression шага `597,632` его Q4-весов стали строгими Q2: блок
содержит `1.187388%` Q2, `61.312612%` Q4 и `37.5%` Q8 при `5.601252 bpw`.
На одноразовом sealed audit-v30 абсолютные C4/SQuAD/sklearn-code ratios равны
`0.992754 / 0.992227 / 0.961534`, incremental ratios против immutable `s0053`
— `1.001822 / 1.003343 / 1.003856 <= 1.005`. Audit-v30 после этого раскрыт.

`layer 22` сначала потребовал дорогого fallback: `19.999949%` Q4 и
`80.000051%` Q8. Первый reverse-compression шаг затем перевёл `956,288`
весов Q4→Q2. Теперь блок содержит `1.899974%` Q2, `18.099976%` Q4 и
`80.000051%` Q8 при `7.287003 bpw`. Sealed audit-v29 прошёл с ratios
`0.991622 / 0.989226 / 0.973866`, incremental worst `1.004860 <= 1.005`.

`layer 25` полностью low-bit: после двух Q8→Q4 и двух Q4→Q2 шагов он
содержит `2.955882%` Q2, `17.204285%` Q4 и `79.839834%` Q8 при
`7.259476 bpw`. Всего в strict ternary переведены `1,487,744` веса.
Последний masked progressive `7→5→3` шаг работал только с нижним `1%`
оставшихся Q4-групп: `683` группы, или `87,424` веса. Fresh development дал
`0.995253 / 1.001842 / 1.019936`; sealed audit-v36 —
`0.992109 / 0.943986 / 0.957007`. Incremental worst равен `1.003765` против
immutable `s0053` и `1.000022` против parent, оба ниже `1.005`.

Следующий rate--distortion шаг вернулся к `layer 24 MLP` и перевёл ещё
1,781 Q4-группу, или `227,968` весов, в strict ternary. Fresh development
дал `0.995290 / 1.001731 / 1.019912`; sealed audit-v35 —
`0.992240 / 0.944613 / 0.966782`. Incremental worst равен `1.004161` против
immutable `s0053` и `1.000162` против parent. Полный Q4-хвост layer 25 через
`7→5→3` collapse был отдельно отклонён: hard-code churn остался нулевым, а
финальный code ratio вырос до `1.034209`.
Тот же progressive collapse на заранее выбранном нижнем `1%` Q4-групп
прошёл: это подтверждает staged codebook как локальный мост Q4→Q2, но не как
основание схлопывать весь чувствительный хвост одновременно.

## Какой это quant

У каждого принятого веса ровно три состояния:

```text
-group_scale, 0, +group_scale
```

- логическая информация: `log2(3) = 1.58496` bit/weight;
- предполагаемый deploy-layout: двухбитный код и FP16 scale на группу 128;
- физическая стоимость: `2 + 16/128 = 2.125 bpw`.

Корректное название: **ternary, logical 1.58-bit, physical Q2-g128
2.125 bpw**. Текущий `.pt` хранит training state и не является компактным
deploy-файлом.

Strict `s0053` содержит 76,673,024 ternary weights. Mixed frontier содержит
81,142,144 Q2-веса, 71,191,680 Q4-весов и 99,324,416 Q8-весов. Текущий
579,580,153-byte `.pt` хранит дублирующиеся int8 training arrays и не является
packed deploy-файлом. Расчётный packed payload этих low-bit весов —
151.766 MiB против 480 MiB в BF16; реальная экономия VRAM появится только
после versioned Q2/Q4/Q8 packer и runtime.

## Что именно преобразовано

`model.layers.27` полностью ternary:

- `mlp.down_proj`, `mlp.up_proj`, `mlp.gate_proj`;
- `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`.

В `model.layers.24` независимо приняты:

- `self_attn.v_proj` и `o_proj`;
- `self_attn.q_proj` и `k_proj`.

В strict `s0053` приняты 75.78125% g128-групп `up_proj`, 10.611979%
`gate_proj` и 22.94921875% `down_proj`. В mixed artifact из оставшихся групп
ещё 11,152 групп (1,427,456 weights) остаются ternary, а 176,272 группы
(22,562,816 weights) используют signed Q4-g128. BF16 в этих трёх MLP-матрицах
больше нет.

`model.layers.23` полностью low-bit: 597,632 весов используют Q2,
30,859,648 — Q4 и 18,874,368 — Q8. BF16 в семи крупных матрицах блока нет.

`model.layers.22` также полностью low-bit: 956,288 весов используют Q2,
9,110,016 — Q4 и 40,265,344 — Q8. BF16 в семи крупных матрицах блока нет.

`model.layers.25` полностью low-bit: 1,487,744 веса используют Q2,
8,659,200 — Q4 и 40,184,704 — Q8. BF16 в семи крупных матрицах блока нет.

## Текущая validation

Актуальный strict parent — `s0053`, SHA-256
`70383ef1b732190c602b8fe1caefc19fa19898df7ada5b15c15994c1d703d0ce`.
Актуальный accepted mixed artifact имеет SHA-256
`01a3176b1971c76868bcd55de068ba5a3058278ad849df4ee7e496a9b98780f4`.
Он прошёл fresh development и sealed audit-v36. Ни Q4/Q8-веса, ни mixed
coverage не прибавляются к strict ternary счётчику.

Две checkpoint-neutral попытки обратно заменить полный Q4 `layer 22` вместо
принятого Q8 fallback отклонены. Малый proxy learning rate почти не менял
hard codes; более высокий дал полезные flips только в очень узкой области и
затем быстро ухудшил все домены. Лучший full-Q4 code ratio равен `1.021875`,
что выше неизменённого development gate `1.02`. Поэтому следующий reverse
шаг был заменён на partial rate-distortion selection. Он успешно перевёл
`956,288` Q4-весов в Q2 и прошёл новый sealed audit-v29; полный collapse
остаётся отклонённым.

Ниже сохранена историческая validation lineage.

Исторический frontier `s0052r1`: coverage-атом `s0052` добавил к
восстановленному `s0051r2` ровно 4,096 g128-групп `layer24.down_proj`, то есть
524,288 strict-ternary weights. На запечатанном one-shot audit-v19 его
cumulative worst ratio равен `0.995173`, а incremental upper-95 worst —
`1.000710` при заранее заданном пределе `1.001306`.

После отклонённой 8,192-group попытки coverage-neutral low-rate recovery
изменил только непринятые BF16-группы layer-24 MLP и сохранил все ternary
codes/scales/masks. На audit-v22 получены:

| Domain | Ratio к BF16 | Incremental upper-95 к `s0052` |
|---|---:|---:|
| C4 train | 0.991022 | 0.998800 |
| SQuAD train context | 1.001470 | 0.998832 |
| NumPy code | 0.972487 | 0.997051 |

Все cumulative point проверки прошли предел `1.002198`, а incremental
upper-95 — заранее зафиксированный предел `1.0005`. Fresh-process standard
verification `s0052r1` дала `wiki=0.943734` и `code=0.953158` relative NLL.
В accepted groups нет BF16 residual, а все codes принадлежат
`{-1,0,+1}`. Recovery не увеличил coverage: он создал запас качества перед
следующей транзакцией.

### Историческая recurring validation parent `s0048r2`

Parent frontier `s0048r2` проверен двумя 131,072-token/domain suites. Audit-v3
использует C4 validation, SQuAD validation contexts и PyTorch code. Audit-v4
заменяет SQuAD/code срезы и использует Transformers code. C4 в этих двух
проверках одинаков. После многократного использования для принятия транзакций
и выбора arm эти suites являются recurring validation, а не независимым
научным holdout.

| Audit | C4 NLL ratio | SQuAD NLL ratio | Code NLL ratio |
|---|---:|---:|---:|
| v3 | 0.996756 | 1.001149 | 0.981170 (PyTorch) |
| v4 | 0.996756 | 0.997175 | 0.982353 (Transformers) |

`ratio < 1` означает, что измеренный candidate NLL ниже teacher на этом
наборе. Это хороший результат, но не доказательство, что тернарная модель
«лучше BF16»: конечные выборки, компенсационное обучение и статистический шум
могут давать локальное улучшение. Для финиша нужны большие PPL suites и
task-benchmarks.

### Paired block-bootstrap audit

Для frontier `s0048r2` дополнительно сохранены NLL каждого из 512 frozen-окон и
посчитан детерминированный 95% CI. Поскольку окна последовательные, bootstrap
пересэмплирует блоки по 8 соседних окон (2,048 токенов), сохраняя BF16 и
candidate строго парными. Использовано 4,096 bootstrap samples.

| Audit/domain | Point ratio | Верхняя 95% граница | Confidence gate |
|---|---:|---:|---|
| v3 C4 | 0.996756 | 0.999155 | pass |
| v3 SQuAD | 1.001149 | 1.006656 | fail |
| v3 PyTorch code | 0.981170 | 0.984688 | pass |
| v4 C4 | 0.996756 | 0.999155 | pass |
| v4 SQuAD | 0.997175 | 1.003054 | fail |
| v4 Transformers code | 0.982353 | 0.986874 | pass |

Обе точечные suite проходят текущий guide `1.00216698`. Кумулятивный
confidence-критерий не проходит только на SQuAD: неопределённость в основном
унаследована от `s0047`. Новый D10-шаг принимался уже по заранее объявленному
двойному правилу: cumulative point-ratio не выше guide и incremental upper-95
не выше `1.0002` на каждом домене v3/v4. Его худшие incremental upper-95 равны
`1.00005088` и `1.00010712`, поэтому шаг принят без ретроактивной настройки.
Для финального научного вывода всё равно нужны непересекающиеся sealed suites.

## Откуда взялся порог 1.02

`1.02` появился в первом коммите WAL-TAT `7f6bd24` как вручную выбранное
значение примера `RatioGate`. Логика была простой: откатывать локальную
транзакцию, если на любом frozen-домене

```text
candidate_NLL / teacher_NLL > 1.02
```

То есть допускалось не более `+2%` **NLL**. Это наша инженерная эвристика, а
не число из BitNet, Prism, CAT-Q, LC-QAT или другой статьи. Оно не было
получено статистической калибровкой.

Важно: `+2% NLL` не равно `+2% PPL`, потому что `PPL = exp(NLL)`. При
teacher NLL около `3.5` ratio `1.02` соответствует примерно `+7.25% PPL`.

Один и тот же `1.02` нельзя разрешать каждому из 28 blocks. Даже грубое
перемножение даёт `1.02^28 = 1.741`, то есть потенциально `+74.1% NLL`.
Поэтому `1.02` остаётся только широким micro-gate для раннего rollback и не
является критерием готовности модели.

## Чем заменяется 1.02

До эксперимента необходимо отдельно объявлять итоговый model budget и метрику:

1. **NLL budget.** Для условной цели `+5% NLL` и coverage `c` используем
   диагностическую линию `1 + 0.05*c`. При текущем `c=0.043339555` это
   `1.00216698`.
2. **PPL budget.** Цель `+5% PPL` означает additive NLL budget
   `log(1.05)`, а не `+5% NLL`. Пропорциональный доменный порог:
   `1 + c*log(1.05)/teacher_NLL`.
3. **Task budget.** MMLU, reasoning, code, instruction following и tool use
   проверяются отдельно: хороший PPL не гарантирует сохранение поведения.

Линия, пропорциональная coverage, тоже является нашей консервативной
диагностикой, а не физическим законом. End-to-end recovery может давать
нелинейную компенсацию. Но она не позволяет потратить весь допустимый ущерб
на первых блоках и ошибочно назвать процесс масштабируемым.

Текущий accepted frontier проходит условную `+5% NLL` guide. На новом
audit-v9 худший point ratio равен `0.995646121`, при guide `1.002167335`.

## Главный технический результат

Первый вариант полного `layer 27` после обычного fixed-code recovery имел
audit ratios около `1.014 / 1.011 / 0.995`. Новый proxy-code процесс оставляет
forward строго ternary, но использует отдельную непрерывную proxy-переменную
для backward. Это позволило пересекать границы между `-1/0/+1` без мягких
весов в реальном forward.

После proxy recovery `layer 27`, а затем attention `layer 24`, совместный
frontier получил ratios ниже `1` на обоих аудитах. Это сильнее старого
результата и впервые укладывается в строгую накопительную guide-линию.

Отрицательный результат тоже важен: one-shot добавление 100% `layer 24
up_proj` хорошо проходило development suite, но на audit-v3 SQuAD ratio после
нескольких recovery стадий остановился на `1.004714`. Вместо ослабления gate
матрица была разбита на sensitivity-ranked транзакции. После первых 12.5%
последовательные порции 3.125%, 1.5625% и 0.78125% прошли оба аудита; общий
принятый coverage `up_proj` на этом этапе достиг 32.51953125%. Попытка добавить 12.5% за один
следующий шаг откатилась при `1.004937`, а 6.25% — при `1.002344` на
development SQuAD. Это подтверждает, что размер commit является частью
оптимизационного пути, а не только инженерным параметром скорости.

Теперь WAL-TAT также умеет обучать связанные BF16-компенсационные окна в
`down_proj/gate_proj`, сохраняя их committed mask нулевым. В matched ablation
это улучшило независимый SQuAD ratio с `1.001575` до `1.001471`. Один linked
прыжок `+1.5625%` откатился при `1.002169`; два последовательных linked шага
по `+0.78125%` достигли того же конечного coverage и прошли оба holdout.
Затем два атома по `+0.390625%` и один адаптивный атом `+0.78125%` также
прошли fresh reload и оба holdout, подняв frontier выше `31.93359375%`.

Дальше начата настоящая конвертация `gate_proj`. Два последовательных атома
по `0.09765625%` приняты; текущий gate coverage равен `0.1953125%`. Новый
campaign runner обучает все arm-ы, параллельно аудитит каждый прошедший arm на
двух holdout и выбирает checkpoint только по независимому худшему ratio.
Coverage-aware gate вычисляется из реальных committed masks, а state
публикуется раньше cleanup старого checkpoint.

Первый настоящий `down_proj` атом также принят: column-structured selector
ограничил 96 новых g128-групп одним входным 128-канальным блоком, то есть
добавил 12,288 hard-ternary weights. Оба matched arm-а прошли development и
два holdout. На holdout победил `candidate_only`: worst ratio `1.001527907`
против `1.001550445` у локальной BF16 `up/gate` компенсации. Поэтому
компенсация не была сохранена только потому, что была сложнее; выбор сделан по
заранее объявленной независимой метрике. Следующая matched-ablation увеличила
атом в четыре раза и приняла ещё 384 группы (`49,152` weights). Здесь уже
BF16 `up/gate` компенсация выиграла holdout: `1.001487769` против
`1.001539105`. Общий `down_proj` coverage стал `0.48828125%`; контроллер из-за
узкого margin уменьшил следующий атом до `0.1953125%`. Этот второй growth-атом
также принят и добавил 192 группы (`24,576` weights). BF16-компенсация снова
выиграла holdout: `1.001531213` против `1.001537834`. Итоговый `down_proj`
coverage равен `0.68359375%`, а следующий размер возвращён к минимальным
`0.09765625%`.

После этого добавлен masked proxy recovery для частично преобразованных
матриц. В отличие от старого proxy он выполняет hard ternary forward и
пропускает градиент только через committed groups, а все непринятые группы
оставляет побитно тем же BF16 master weight. На реальном checkpoint начальный
forward совпал для всех 14 матриц. Recovery не изменил ни одного committed
mask и не затронул uncommitted master weights; изменились два ternary-кода в
`layer24.v_proj` и FP16 scales принятых групп. Худший независимый ratio
улучшился с `1.001531213` до `1.000654804`, то есть запас до cumulative gate
вырос более чем втрое. Scale-only control откатился при SQuAD development
ratio `1.008575`, а сильнее заякоренный proxy прошёл, но был хуже выбранного
варианта (`1.001461038` на audit-v3). Восстановленный запас затем позволил
проверить крупные `up_proj` атомы: `+6.25%` честно откатился буквально на
`0.0000043` выше development gate в лучшем linked arm-е, после чего
контроллер уменьшил шаг до `+3.125%`. Этот шаг добавил 393,216 новых ternary
weights и прошёл оба holdout; linked arm выиграл с worst `1.000731191` против
`1.000791568` у candidate-only. Повторный атом того же размера добавил ещё
393,216 weights, прошёл оба holdout и снова выбрал linked arm: worst
`1.000755065` против `1.000816277`. Coverage `up_proj` вырос до
`38.76953125%`, то есть крупный шаг воспроизвёлся два раза подряд. Третий
атом `+3.125%` тоже прошёл оба holdout и поднял coverage до `41.89453125%`,
но на этот раз candidate-only оказался лучше linked arm: worst
`1.001118535` против `1.001146107`. Это подтверждает, что компенсационное
окно надо выбирать по holdout для каждой транзакции, а не включать постоянно.
Повторный masked proxy recovery на этом frontier не изменил ни одного
тернарного кода и ни одного uncommitted BF16/scale элемента, но перенастроил
467,770 committed scales. Он снизил audit-v3 worst с `1.001118535` до
`1.000297213`, а на audit-v4 оставил все ratios ниже единицы. Coverage при
этом остался строго `41.89453125% up_proj`.
Из восстановленного frontier затем принят sensitivity-ranked атом
`down_proj +0.390625%` (`49,152` weights). Linked arm прошёл оба holdout и
победил candidate-only с worst `1.000391478` против `1.000426078`; coverage
`down_proj` вырос до `1.07421875%`.
Следующий matched атом `gate_proj +0.1953125%` добавил 24,576 weights и тоже
прошёл оба holdout. Linked arm выиграл с worst `1.000349834` против
`1.000396429`; coverage `gate_proj` удвоился до `0.390625%`.
Round-robin вернулся к `up_proj`: ещё один атом `+3.125%` добавил 393,216
weights и прошёл оба holdout. Linked arm выиграл с worst `1.000441498`
против `1.000666787`; coverage `up_proj` достиг `45.01953125%`.
Следующий `down_proj +0.390625%` атом добавил 49,152 weights. Linked arm
прошёл оба holdout и выиграл с worst `1.000459295`; coverage `down_proj`
достиг `1.46484375%`.
Следующий `gate_proj +0.1953125%` атом добавил 24,576 weights и также прошёл
оба holdout. На этот раз candidate-only немного лучше linked arm:
`1.000469347` против `1.000483896`. Coverage `gate_proj` достиг
`0.5859375%`; это подтверждает, что BF16-компенсация выбирается по holdout,
а не сохраняется автоматически.
Round-robin снова добавил `up_proj +3.125%`, то есть 393,216 weights.
Linked arm выиграл у candidate-only (`1.000691962` против `1.000778586`) и
поднял coverage `up_proj` до `48.14453125%`.
Следующий `down_proj +0.390625%` атом добавил 49,152 weights. Linked arm
выиграл (`1.000677098` против `1.000722296`) и поднял coverage `down_proj`
до `1.85546875%`.
Следующий `gate_proj +0.1953125%` атом добавил 24,576 weights. Linked arm
выиграл (`1.000642781` против `1.000709085`) и поднял coverage `gate_proj`
до `0.78125%`.
Ещё один `up_proj +3.125%` атом добавил 393,216 weights и впервые перевёл
больше половины матрицы: coverage `51.26953125%`. Linked arm выиграл
(`1.000961833` против `1.000986563`), но normalized headroom сузился до
`0.525152`, поэтому следующий крупный up-атом откладывается до очередного
round-robin/recovery решения.
Следующий `down_proj +0.390625%` атом добавил 49,152 weights и поднял его
coverage до `2.24609375%`. Оба arm-а прошли development и два holdout;
candidate-only был лучше на development, но linked BF16 MLP arm на
независимом holdout выиграл с минимальным перевесом (`1.001010915` против
`1.001012296`). Normalized headroom равен `0.501272`, поэтому после одного
следующего gate-атома выполняется masked proxy recovery всего частичного MLP.
Запланированный `gate_proj +0.1953125%` атом добавил 24,576 weights и поднял
coverage до `0.9765625%`. Оба arm-а прошли два holdout; linked arm снова едва
выиграл (`1.001018946` против `1.001023001`). Normalized headroom
`0.497488`, поэтому следующий этап — masked proxy recovery без роста coverage.
Третий masked proxy recovery сохранил все committed masks и все непринятые
BF16 master weights. Он изменил один активный ternary-код в `layer24.v_proj`,
476,026 committed scales и 495 неактивных codes под нулевой mask; последние
не участвуют в forward. Худший независимый ratio улучшился с `1.001018946`
до `1.000115086`, а normalized headroom вырос до `0.943243` без изменения
coverage. Это открывает следующий `up_proj +3.125%` атом.
Этот атом добавил 393,216 hard-ternary weights и поднял `up_proj` до
`54.39453125%`. Оба arm-а прошли оба holdout; candidate-only оказался лучше
linked BF16 MLP (`1.000404500` против `1.000427594`). Normalized headroom
остался высоким — `0.801631`, поэтому контроллер сохраняет размер следующего
up-атома `3.125%`.
Следующий `down_proj +0.390625%` атом добавил 49,152 weights и поднял
coverage до `2.63671875%`. Linked arm выиграл независимый holdout у
candidate-only (`1.000412414` против `1.000453858`), normalized headroom
остался высоким — `0.797892`.
Следующий `gate_proj +0.1953125%` атом добавил 24,576 hard-ternary weights и
поднял coverage до `1.171875%`. Оба arm-а снова прошли два holdout, но теперь
candidate-only уверенно выиграл и development, и независимую проверку:
`1.000480310` против `1.000712729` у linked BF16 MLP. Динамический gate равен
`1.002041273`, normalized headroom остаётся высоким — `0.764701`. Поэтому
round-robin возвращается к `up_proj +3.125%`, без промежуточного recovery.
Этот `up_proj` атом добавил 393,216 hard-ternary weights и поднял coverage до
`57.51953125%`. Оба arm-а прошли оба holdout. Candidate-only был лучше на
development, но linked arm минимально выиграл независимый worst:
`1.000862859` против `1.000871839`. Normalized headroom уменьшился до
`0.579647`, поэтому размер следующего up-атома остаётся `3.125%`, а
round-robin сначала переходит к `down_proj +0.390625%`.
Этот down-атом добавил 49,152 hard-ternary weights и поднял coverage до
`3.02734375%`. Оба arm-а прошли оба holdout; linked arm выиграл с worst
`1.000977218` против `1.000998762` у candidate-only. Normalized headroom
равен `0.524267`, поэтому следующий round-robin шаг — один
`gate_proj +0.1953125%` атом перед решением о recovery.
Этот gate-атом добавил 24,576 hard-ternary weights и поднял coverage до
`1.3671875%`. Candidate-only выиграл development и оба holdout; worst равен
`1.000953523` против `1.000995915` у linked arm. Normalized headroom равен
`0.535963`: следующая контролируемая попытка — `up_proj +3.125%`; при отказе
или заметном сужении запаса выполняется masked proxy recovery.
Попытка `up_proj +3.125%` затем была честно отклонена ещё на development:
candidate-only дал SQuAD ratio `1.002592097`, linked arm — `1.002554625`,
тогда как coverage-aware gate равен `1.002066271`. Новые weights не приняты,
оба временных состояния отброшены, исходный checksum сохранён. Контроллер
уменьшил следующую up-долю до `1.5625%`; перед ней выполняется masked proxy
recovery без роста coverage.
Masked proxy recovery v4 прошёл оба cumulative holdout и сохранил coverage,
все committed/uncommitted codes и все непринятые BF16 master weights. Он
изменил только 481,135 committed scales. Худший независимый ratio снизился с
`1.000953523` до `1.000301811`, normalized headroom вырос до `0.853122`.
Следующая up-попытка использует уже уменьшенную контроллером долю `1.5625%`.
Уменьшенная up-транзакция прошла оба holdout и добавила 196,608 hard-ternary
weights: `up_proj` достиг `59.08203125%`. Candidate-only выиграл development,
но linked arm минимально выиграл независимый worst (`1.000243471` против
`1.000255685`). Normalized headroom остался высоким — `0.881842`; следующий
round-robin шаг — `down_proj +0.390625%`.
Следующий down-атом прошёл оба holdout и добавил 49,152 hard-ternary weights:
`down_proj` достиг `3.41796875%`. Development предпочёл linked arm, но
независимый worst был лучше у candidate-only (`1.000307810` против
`1.000332156`), поэтому сохранён candidate-only checkpoint. Normalized
headroom равен `0.850722`; следующий round-robin шаг —
`gate_proj +0.1953125%`.
Следующий gate-атом также прошёл оба holdout и добавил 24,576 hard-ternary
weights: `gate_proj` достиг `1.5625%`. Candidate-only выиграл development,
но linked arm выиграл независимый worst (`1.000336331` против
`1.000357894`). Normalized headroom равен `0.836946`; round-robin
возвращается к уменьшенному `up_proj +1.5625%`.
Следующая уменьшенная up-транзакция прошла оба holdout и добавила 196,608
hard-ternary weights: `up_proj` достиг `60.64453125%`. Candidate-only выиграл
и development, и независимый worst (`1.000550819` против `1.000677756`).
Normalized headroom снизился до `0.733700`, поэтому размер следующего
up-атома не растёт; round-robin сначала переходит к
`down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 hard-ternary weights и
поднял `down_proj` до `4.58984375%`. Candidate-only минимально выиграл у
linked arm (`1.000247348` против `1.000264837`), normalized headroom равен
`0.881399`. Следующий round-robin шаг — `gate_proj +0.1953125%`.
Следующий down-атом прошёл оба holdout и добавил 49,152 hard-ternary weights:
`down_proj` достиг `3.80859375%`. Candidate-only выиграл development, но
linked arm выиграл независимый worst (`1.000555450` против `1.000598652`).
Normalized headroom равен `0.731646`; следующий round-robin шаг —
`gate_proj +0.1953125%`.
Следующий gate-атом прошёл оба holdout и добавил 24,576 hard-ternary weights:
`gate_proj` достиг `1.7578125%`. Candidate-only минимально выиграл и
development, и независимый worst (`1.000559869` против `1.000562804`).
Normalized headroom равен `0.729604`; следующий контролируемый опыт —
`up_proj +1.5625%`, с recovery при отказе или tight headroom.
Следующий up-атом прошёл оба holdout и добавил 196,608 hard-ternary weights:
`up_proj` достиг `62.20703125%`. Linked arm выиграл development, но
candidate-only выиграл независимый worst (`1.000713586` против
`1.000765682`). Normalized headroom снизился до `0.656314`; следующий
round-robin шаг — `down_proj +0.390625%`, с recovery при отказе или tight
headroom.
Этот down-атом прошёл оба holdout и добавил 49,152 hard-ternary weights:
`down_proj` достиг `4.19921875%`. Candidate-only выиграл development, но
linked BF16-MLP arm выиграл независимый worst (`1.000811028` против
`1.000830712`). Normalized headroom равен `0.609651`; следующий
round-robin шаг — `gate_proj +0.1953125%`, затем masked proxy recovery перед
следующим крупным up-атомом.
Этот gate-атом прошёл оба holdout и добавил 24,576 hard-ternary weights:
`gate_proj` достиг `1.953125%`. Candidate-only выиграл development, но linked
BF16-MLP arm выиграл независимый worst (`1.000820492` против `1.000865155`).
Normalized headroom равен `0.605231`. Следующий шаг — coverage-neutral masked
proxy recovery перед следующим крупным up-атомом.
Masked proxy recovery v5 сохранил coverage и все committed masks, изменил
только два уже принятых ternary-кода, перенастроил 486,161 committed scales и
не затронул uncommitted BF16 master weights. Независимый worst снизился с
`1.000820492` до `1.000171060`, normalized headroom вырос до `0.917697`.
Следующий контролируемый шаг — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 hard-ternary weights и
поднял `up_proj` до `63.76953125%`. Linked BF16-MLP arm выиграл и
development, и независимый worst (`1.000192249` против `1.000249673`).
Normalized headroom остался высоким: `0.907755`. Следующий round-robin шаг —
`down_proj +0.390625%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 hard-ternary weights и
поднял `gate_proj` до `2.1484375%`. Candidate-only выиграл у linked arm
(`1.000247756` против `1.000354786`), normalized headroom равен `0.881245`.
Следующий round-robin шаг — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 hard-ternary weights и
поднял `up_proj` до `66.89453125%`. Candidate-only выиграл у linked arm
(`1.000487872` против `1.000599281`), normalized headroom равен `0.767662`.
Следующий round-robin шаг — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 hard-ternary weights и
поднял `down_proj` до `5.37109375%`. Candidate-only выиграл у linked arm
(`1.000605718` против `1.000739123`), normalized headroom равен `0.711737`.
Следующий round-robin шаг — `gate_proj +0.1953125%`.
Этот gate-атом тоже прошёл оба holdout, добавил 24,576 hard-ternary weights и
поднял `gate_proj` до `2.5390625%`. Candidate-only выиграл у linked arm
(`1.000627927` против `1.000815486`), normalized headroom равен `0.701269`.
Следующий round-robin шаг — `up_proj +1.5625%` без предварительного recovery.
Попытка `up_proj +1.5625%` была откатана: candidate-only превысил development
gate на `0.00002647` ratio, linked arm — на `0.00009850`. Уменьшенный атом
`+0.78125%` прошёл оба holdout и добавил 98,304 hard-ternary weights:
`up_proj` достиг `67.67578125%`. Holdout выбрал linked arm вместо
development-победителя candidate-only (`1.000638609` против `1.000662713`),
normalized headroom равен `0.696599`. Следующий шаг — `down_proj +0.390625%`.
Следующий down-атом прошёл оба holdout, добавил 49,152 hard-ternary weights и
поднял `down_proj` до `5.76171875%`. Независимый аудит выбрал candidate-only
вместо development-победителя linked arm (`1.000809223` против
`1.000834380`), normalized headroom равен `0.615802`. Следующий round-robin
шаг — `gate_proj +0.1953125%`.
Следующий gate-атом прошёл оба holdout, добавил 24,576 hard-ternary weights и
поднял `gate_proj` до `2.734375%`. Candidate-only выиграл у linked arm
(`1.000730373` против `1.000794258`), normalized headroom равен `0.653356`.
Следующий контролируемый шаг — `up_proj +0.78125%`; recovery выполняется при
отказе или дальнейшем переходе headroom в tight-зону.
Этот up-атом прошёл оба holdout, добавил 98,304 hard-ternary weights и поднял
`up_proj` до `68.45703125%`. Candidate-only выиграл у linked arm
(`1.000762124` против `1.000833618`), normalized headroom равен `0.638776`.
Следующий round-robin шаг — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 hard-ternary weights и
поднял `down_proj` до `6.15234375%`. Candidate-only выиграл у linked arm
(`1.000850378` против `1.000987898`), normalized headroom равен `0.597219`.
Следующий round-robin шаг — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 hard-ternary weights и
поднял `gate_proj` до `2.9296875%`. Holdout минимально предпочёл linked arm
вместо candidate-only (`1.000883711` против `1.000884797`), normalized headroom
равен `0.581573`. Следующий контролируемый шаг — `up_proj +0.78125%`.
Этот up-атом прошёл оба holdout, добавил 98,304 hard-ternary weights и поднял
`up_proj` до `69.23828125%`. Candidate-only выиграл у linked arm
(`1.001048501` против `1.001158013`), normalized headroom равен `0.504217`.
Следующий round-robin шаг — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 hard-ternary weights и
поднял `down_proj` до `6.54296875%`. Holdout выбрал candidate-only вместо
development-победителя linked arm (`1.001052724` против `1.001137076`),
normalized headroom равен `0.502556`. Следующий шаг — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 hard-ternary weights и
поднял `gate_proj` до `3.125%`. Candidate-only выиграл у linked arm
(`1.001044083` против `1.001077155`), normalized headroom немного вырос до
`0.506806`. Перед следующим up-атомом выполняется coverage-neutral recovery.
Masked proxy recovery v6 сохранил все committed masks, coverage и все
непринятые BF16 master weights. Изменился один активный ternary-код в
`o_proj`, 494,947 committed scales и 18,954,746 элементов master weights
только внутри уже принятых групп. Worst независимого аудита снизился до
`1.000346068`, normalized headroom вырос до `0.836528`. Следующий шаг —
повторить `up_proj +0.78125%` из восстановленного frontier.
Повтор прошёл оба holdout, добавил 98,304 hard-ternary weights и поднял
`up_proj` до `70.01953125%`. Хотя development предпочёл linked BF16 MLP arm,
независимый holdout выбрал candidate-only: worst `1.000616263` против
`1.000722317`. Normalized headroom равен `0.709288`; следующий round-robin
шаг — `down_proj +0.390625%`.
Следующий полный round-robin цикл также прошёл без ослабления gate. Down-атом
добавил 49,152 weights и поднял `down_proj` до `6.93359375%` (worst
`1.000653014`), gate-атом добавил 24,576 weights и поднял `gate_proj` до
`3.3203125%` (worst `1.000688123`), up-атом добавил 98,304 weights и поднял
`up_proj` до `70.80078125%` (worst `1.000783641`). Во всех трёх транзакциях
holdout выбрал candidate-only; итоговый normalized headroom равен `0.631200`.

После этого приняты ещё девять транзакций: `down/gate s0017`, три up-атома
`s0023..s0025` и связанные `down/gate s0018..s0019`. Они добавили 516,096
hard-ternary weights. Итоговое покрытие layer 24 достигло `73.14453125% up`,
`8.10546875% down` и `3.90625% gate`. Последний gate-step выбрал linked
BF16-MLP arm по двум holdout (`1.001126802` против `1.001156927`), не меняя
ternary coverage. Его normalized headroom равен `0.473417`; следующий атом —
`up_proj +0.78125%` с обязательным rollback при отказе любого holdout.

Следующий `up s0026 +0.78125%` был отклонён на development SQuAD
(`1.002279340` и `1.002317004` при dynamic gate `1.002142694`), поэтому
фронтир не изменился, а размер атома был автоматически уменьшен вдвое.
`up s0027 +0.390625%`, затем `down s0020 +0.390625%` и
`gate s0020 +0.1953125%` прошли development, fresh reload и оба holdout.
Все три holdout-решения выбрали candidate-only. Круг добавил 122,880
hard-ternary weights; финальный worst равен `1.001287470`, normalized
headroom — `0.399335`.
Этот up-атом прошёл оба holdout, добавил 196,608 hard-ternary weights и
поднял `up_proj` до `65.33203125%`. Candidate-only минимально выиграл у
linked arm (`1.000363082` против `1.000371670`), normalized headroom равен
`0.826441`. Следующий round-robin шаг — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 hard-ternary weights и
поднял `down_proj` до `4.98046875%`. Holdout выбрал candidate-only вместо
development-победителя linked arm (`1.000442839` против `1.000456372`);
normalized headroom равен `0.788461`. Следующий шаг —
`gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 hard-ternary weights и
поднял `gate_proj` до `2.34375%`. Candidate-only выиграл у linked arm
(`1.000418275` против `1.000454195`), normalized headroom равен `0.800263`.
Следующий round-robin шаг — `up_proj +1.5625%`.

Продолжение после evidence v70 добавило 147,456 hard-ternary weights без
ослабления cumulative gate. `up s0028 +0.390625%`, `down s0022 +0.1953125%`
и `gate s0021 +0.1953125%` прошли оба holdout; крупный `down s0021` был
корректно отклонён. Затем `up s0029 +0.390625%` также откатился на
development, а уменьшенный `up s0030 +0.1953125%` прошёл. Из-за узкого
headroom следующие `down s0023` и `gate s0022` использовали минимальные
атомы по `0.09765625%`. Оба приняты после fresh reload и двух holdout.
Итоговые coverage: `74.12109375% up`, `8.7890625% down` и `4.39453125%
gate`. Финальный worst равен `1.001663819` при dynamic guide `1.002147694`;
normalized headroom равен `0.225300`. Перед дальнейшим ростом выполняется
coverage-neutral masked proxy recovery v7.

Повторный запуск после восстановления хоста успешен. Masked proxy recovery
v7 не изменил coverage и committed masks, изменил один код внутри уже
принятой `layer24.o_proj`, перенастроил 501,729 committed scales и не затронул
ни одного непринятого BF16 weight/scale. Худший независимый ratio снизился с
`1.001663819` до `1.001092666`. Затем полностью принят минимальный
round-robin цикл `up s0031`, `down s0024`, `gate s0023` и следующий
`up s0032`: четыре транзакции добавили 49,152 hard-ternary weights. Все
прошли fresh reload и оба holdout. Текущий worst равен `1.001076137` при
guide `1.002149122`, normalized headroom — `0.499267`. Coverage layer 24:
`74.31640625% up`, `8.88671875% down`, `4.4921875% gate`.

Следующий последовательный цикл `down s0025`, `gate s0024`, `up s0033`
прошёл при строго одной активной экспериментальной сессии за раз. Он добавил
36,864 hard-ternary weights. Все три шага прошли development, fresh reload и
оба holdout; каждый раз holdout выбрал candidate-only. Итоговый worst равен
`1.001147515` при guide `1.002150193`, normalized headroom — `0.466320`.
Coverage layer 24 достиг `74.4140625% up`, `8.984375% down` и
`4.58984375% gate`.

Ещё один полный последовательный цикл `down s0026`, `gate s0025`, `up s0034`
также прошёл при одной управляющей сессии за раз и добавил 36,864
hard-ternary weights. Все шесть независимых audit-запусков приняли
candidate-only checkpoint. Итоговый worst равен `1.001148209` при guide
`1.002151265`, normalized headroom — `0.466263`. Coverage layer 24 теперь
равен `74.51171875% up`, `9.08203125% down` и `4.6875% gate`; общий
подтверждённый coverage модели — `4.302529331%`.

Цикл 27 впервые выполнен новым единым orchestrator-процессом: `down s0027`,
`gate s0026` и `up s0035` последовательно отработали внутри одной терминальной
сессии при максимуме одной активной кампании. Все три транзакции и все шесть
независимых audit-запусков прошли; holdout каждый раз выбрал candidate-only.
Цикл добавил 36,864 hard-ternary weights. Финальный worst равен
`1.001155576` при guide `1.002152336`, normalized headroom — `0.463106`.
Coverage layer 24 теперь `74.609375% up`, `9.1796875% down` и
`4.78515625% gate`; общий coverage модели — `4.304672025%`.

После цикла исправлено сохранение safe-streak адаптивного контроллера между
одношаговыми дочерними процессами. Раньше счётчик roomy-pass сбрасывался при
каждом запуске и автоматический рост атома был фактически недостижим. Теперь
он сохраняется в campaign state; отдельный unit-тест воспроизводит границу
процесса. Текущие три решения остаются `middle_headroom_hold`, поэтому сам
размер атома пока корректно остаётся минимальным.

Три следующих круга 28–30 выполнены тем же единым orchestrator-процессом.
Все девять транзакций и 36 независимых audit-запусков для matched arms прошли;
добавлено 122,880 hard-ternary weights. Реальный safe-streak подтвердил
исправление: после двух roomy-проходов `gate_proj` увеличил атом с `1/1024`
до `1/512`, и увеличенный `gate s0029` также прошёл оба holdout, добавив
24,576 weights одним commit. Его holdout выбрал `candidate_bf16_mlp` с worst
`1.001250899`. Down/up остались на минимальном размере, потому что их
normalized headroom опустился ниже 45%. Финальный `up s0038` принят с worst
`1.001280233` при guide `1.002155907`; общее покрытие — `4.311814338%`.

Три следующих круга 31–33 также выполнены одним последовательным
orchestrator-процессом. Все девять транзакций приняты и добавили 147,456
hard-ternary weights. Увеличенный `gate`-атом `1/512` прошёл три раза, но к
концу серии normalized headroom снизился до `0.311371`, поэтому контроллер
вернул все три проекции к минимальному атому `1/1024`. Финальный `up s0041`
принят с worst `1.001487571` при guide `1.002160193`; общее покрытие стало
`4.320385114%`. За серию автоматически удалено 4,265,229,656 bytes
промежуточных checkpoint; WAL, команды и метрики сохранены.

Круги 34–36 добавили ещё девять принятых транзакций и 110,592
hard-ternary weights. Они довели `layer 24` до `75.48828125% up`,
`10.05859375% down` и `6.0546875% gate`; общий coverage стал
`4.326813195%`. Все кампании шли строго последовательно, максимум одна за
раз. Новый frontier `s0044` прошёл оба точечных holdout; paired bootstrap
снова уверенно пропускает C4/code, а SQuAD остаётся статистически
неопределённым. Orchestrator удалил 17 промежуточных checkpoint на
5,179,207,448 bytes; после проверки ссылок также удалён superseded `s0041`
на 304,658,813 bytes. В `wal2/checkpoints` оставлен один текущий frontier.

### Проверка transform-space и исправление интерпретации

На ещё нетронутой BF16-матрице `layer 23 q_proj` выполнен
checkpoint-neutral тест обычной g128-тернаризации против blockwise randomized
Hadamard transform. На development identity дал worst ratio `1.028783`, а
лучший заранее выбранный RHT seed 307 — `1.002092`. Фиксированный seed затем
без перенастройки проверен на двух holdout: v3 `1.001779` против `1.017127`
у identity, v4 `1.002547` против `1.025458`. Это уменьшение наблюдаемого
worst-domain ущерба примерно на 90% в обеих репликациях.

Важное исправление: эти числа являются **инкрементальными** ratios относительно
уже тернаризованного frontier `s0044`, а guide `1.002163` определена для
кумулятивного сравнения с исходной BF16-моделью. Поэтому прежнее прямое
сравнение RHT-чисел с этой guide было некорректным. Кроме того, после
многократного адаптивного использования v3/v4 считаются recurring validation,
а не независимыми финальными holdout.

Короткий hard-forward proxy/scale recovery поверх seed 307 выполнен и
заморожен только по development. Code churn оказался нулевым: recovery
улучшил scales, но не ternary codes. На кумулятивном paired audit-v3 SQuAD
ratio стал `1.002892` с верхней 95% границей `1.008465`, поэтому кандидат
отклонён. На v4 point ratio `0.998729` прошёл, но верхняя граница `1.004656`
не прошла confidence gate. Accepted coverage и основной checkpoint не
изменились. RHT остаётся перспективной инициализацией, но не подтверждённым
frontier commit. При deploy RHT всё равно должен исполняться вместе с packed
ternary codes; обратное восстановление BF16-матрицы уничтожило бы сжатие.

### Ограниченные round-robin циклы 37--39

Один последовательный orchestrator выполнил девять транзакций
`down → gate → up`. Все девять прошли заранее заданный point gate и добавили
110,592 hard-ternary weights. Первые два шага выбрали linked BF16-MLP arm,
остальные семь — candidate-only. Финальный coverage `layer 24` равен
`75.78125% up / 10.3515625% down / 6.34765625% gate`, общий coverage —
`4.333241277%`. Максимум одновременно работала одна campaign. Удалено 13
superseded/rejected checkpoint на 3,960,569,665 bytes; оставлен один frontier.

Финальный paired recurring-validation audit подтвердил point gate, но не
confidence gate. На v3 worst point `1.001560871`, SQuAD upper-95
`1.007022346`; на v4 worst point ниже 1, но SQuAD upper-95 `1.003624753`.
Поэтому `s0047` сохраняется по старой predeclared policy, но дальнейший рост
по этой же policy остановлен до statistical gate и новых sealed suites.

### Prospective D10-реплика и frontier `s0048`

До запуска нового кандидата сохранена policy `evidence_v83_policy.json`.
Она зафиксировала непересекающиеся 96 D10-групп `layer 24 down_proj`,
`activation_wls`, 256 recovery steps, нулевое обучение norm/старых committed
групп и двойной gate: cumulative point `<= 1.002166978` плюс incremental
paired upper-95 `<= 1.0002` на каждом домене recurring v3/v4. Ранее
просмотренный D10-кандидат явно исключён из принятия по этой новой policy.

Сырая инициализация дала отрицательный результат для универсальности WLS:
на новой реплике worst development ratio был `1.000181289`, тогда как absmean
на тех же группах дал `1.000039946`. Рецепт не менялся после просмотра.
Hard-forward recovery не сменил ни одного ternary-кода, но перенастроил scales
и снизил worst development ratio до `1.000082403`.

На v3 худший incremental upper-95 составил `1.000050878`, на v4 —
`1.000107123`; cumulative point gate также пройден. Поэтому 96 групп, или
12,288 weights, приняты. Новый coverage: `74,563,584 / 1,720,451,072 =
4.333955508%`, `down_proj = 10.44921875%`. Новый checkpoint отдельно загружен,
проверен по development suite, а artifact mask/codes/scales сверены побитно.
Старый `s0047` удалён только после этих проверок.

### Новые rotating validation и recovery v8--v11

После `s0048` построены два новых набора v5/v6. Их C4, SQuAD и code token
windows не пересекаются с v1--v4 и друг с другом. Они уже раскрыты в
evidence v84, поэтому являются **rotating validation**, а не sealed final
audit. На v5 сам parent `s0048` дал SQuAD ratio `1.005233579`; на v6 —
`1.001054693`. Новая absmean D10-реплика была почти инкрементально нейтральна
(upper-95 не хуже `1.000099`), но отклонена из-за cumulative v5 SQuAD point.
Coverage не изменился.

Для recovery создан ещё один train/development suite v2 с 98,304 calibration
tokens, SQuAD-весом x4, interleaved domains и новыми локальными
Accelerate/Datasets code sources. Точные token-window пересечения со всеми
recovery-v1 и v5/v6 равны нулю. Этот suite воспроизвёл более крупный
предсуществующий SQuAD gap `1.013493297`, то есть проблема v5 не является
единичным случайным срезом.

Четыре coverage-neutral recovery были заранее объявлены и безопасно
отклонены:

- v8: block-wide KD/proxy recovery; каждый trained snapshot хуже source,
  выбран `step 0`;
- v9: MLP scale-only recovery; каждый snapshot хуже source, выбран `step 0`;
- v10: hard-code recovery со frozen scales; к шагу 576 появились первые
  переключения, а к 768 churn достиг `0.2861%` в `down_proj` и ухудшил SQuAD
  до `1.020728`;
- v11: `down_proj`-only staged recovery (`proxy move -> code freeze -> scale
  polish`); churn `0.01156%` уже ухудшил selection, а scale polish вернул лишь
  малую часть SQuAD loss и дополнительно ухудшил C4/code.

В proxy diagnostics теперь записываются расстояния до порогов `±0.5`, code
entropy, zero fraction, proxy displacement, scale percentiles и доля scale на
clamp. Они подтвердили, что прежний `code_churn = 0` не означал отсутствие
обучения: proxy двигались к границам, но первые реальные переключения были
вредными. Ни один из этих broad recovery checkpoint не опубликован; frontier
на этом этапе оставался ровно `s0048`.

### Matched control, counterfactual teacher и frontier `s0048r1`

После broad recovery выполнен checkpoint-neutral counterfactual аудит. Он
показал, что основная локальная причина SQuAD gap находится не в fallback и не
в norm extras, а в уже принятых группах `layer 24 up_proj`: восстановление
только этой committed-области в исходный BF16 возвращало около 74% наблюдаемого
разрыва, а восстановление всех committed MLP-групп — около 92%.

Чистый matched BF16 control с raw исходной моделью в роли teacher не помог:
такой teacher одновременно убирал полезное состояние layer 27. Новый
target-local counterfactual teacher оставил весь `s0048` без изменений и
восстановил в BF16 только committed-группы target `up_proj`. С ним continuous
residual recovery прошёл заранее объявленный development gate на шаге 384:
worst-ratio улучшился на `0.000530185`. Этот residual был только обучающим
контролем и не принимался как deploy-состояние.

Затем residual спроецирован обратно в строгий Q2-g128. Codes остались
неизменными (`code churn = 0`), менялись только общие FP16 scales. Базовая
амплитуда `1x` улучшала два recurring среза, но не достигла заранее заданного
selection-порога. Амплитуда `8x` прошла selection и провалила confirmation.
Фиксированный `4x` кандидат прошёл оба: `+0.000051848` на selection и
`+0.000024568` на confirmation.

После заморозки кандидат проверен на 512 новых окнах каждого домена audit-v3.
Относительно parent `s0048` point ratios равны `0.999988306 / 0.999931357 /
0.999970879` для C4/SQuAD/code. Верхние paired 95% границы равны
`1.000031683 / 0.999989735 / 1.000039381`, то есть все прошли заранее
объявленный incremental limit `1.0001`. Persistent BF16 residual отсутствует.

Опубликован lineage checkpoint `s0048r1`. В нём изменено 72,573 group-scales,
ни одного ternary code value и ни одного committed mask. Fresh-load проверка
дала `wiki=0.950322018`, `code=0.961868530` относительно BF16 на стандартном
gate. Parent `s0048` удалён только после проверки. Coverage поэтому не вырос,
но качество принятого strict-Q2 состояния стало лучше.

### Прямой strict-scale QAT после matched control

После публикации `s0048r1` проверен более прямой путь: обучались только
положительные g128 scales уже принятых групп `layer24.up_proj`, а ternary
codes, masks, BF16 fallback и norms оставались замороженными. Teacher был тем
же target-local counterfactual состоянием. Первый FP32-вариант оказался
невалидным для deployment: небольшое улучшение до сериализации превращалось в
ухудшение после обязательного округления scales в FP16.

Поэтому forward был изменён на deployment-faithful FP16 fake quant с STE.
Лучший snapshot на шаге 512 улучшил worst selection ratio на `0.000080637`,
но заранее объявленный порог публикации был `0.0001`. Детерминированное
продление до 768 шагов не улучшило step 512; более поздние шаги начали
ухудшать SQuAD. Артефакт и новый checkpoint не публиковались, coverage и
frontier остались неизменными.

Вывод: общий scale-only градиент полезен, но на этой границе уже упёрся в
дискретную геометрию. Дальше нужен boundary-aware выбор небольшого числа
ternary-code переходов либо новый prospective tail, а не очередной перебор
только LR/steps.

### Boundary-aware sparse recode и frontier `s0048r2`

Следующий эксперимент измерил градиент target-local KD/block loss по уже
принятым весам `layer24.up_proj`, но не сохранял непрерывный residual. Для
каждой g128-группы разрешался максимум один соседний переход
`-1 <-> 0 <-> +1`; кандидаты ранжировались first-order score. Все остальные
codes, masks, BF16 fallback и нормы оставались замороженными.

Первый search выбрал 512 code edits без изменения scales. На development и
отдельном confirmation он улучшил worst ratio на `0.000171719` и
`0.000348307`. Однако frozen paired audit честно отклонил его: все point
ratios улучшились, но верхняя 95% граница SQuAD была `1.000152929` при заранее
объявленном incremental limit `1.0001`.

После этого до повторного запуска был фиксирован более консервативный вариант:
128 групп и counterfactual-LS scale только в этих группах. Он прошёл
development (`+0.000149923`) и confirmation (`+0.000295983`). На 512 frozen
окнах каждого audit-v3 домена incremental point ratios относительно `s0048r1`
равны `0.999573014 / 0.999745906 / 0.998906676`, а верхние paired 95% границы
— `0.999669949 / 0.999923784 / 0.999032864`. Все ниже `1.0001`.

Отдельный sparse-recode commit проверил точные hashes, разрешил ровно 128
code changes и 128 scale changes только внутри committed mask и опубликовал
`s0048r2`. Fresh-load дал `wiki=0.949896401`, `code=0.961393079`; во всех
матрицах codes остаются строго `{-1,0,+1}`. Coverage не изменился: это
улучшение геометрии уже принятого Q2, а не рост числа тернарных весов.

После публикации текущий checkpoint повторно измерен на уже раскрытых v5/v6.
На v5 C4/code остаются лучше BF16 (`0.995501 / 0.988905`), а SQuAD ratio
снизился с прежних `1.005234` примерно до `1.005052`, но всё ещё выше
cumulative guide `1.002167`. На v6 все point ratios проходят:
`0.998432 / 1.000922 / 0.997408`. Confidence intervals на SQuAD остаются
широкими. Поэтому v5/v6 используются только как recurring diagnostics, но их
результат запрещает немедленно принимать новый coverage atom по текущей
строгой policy.

### Joint MLP recovery и frontier `s0048r3`

Для выхода из этого quality frontier создан matched development suite с
непересекающимися selection/confirmation диапазонами. Joint initializer
ablation выбрала `threshold_ls` для `gate_proj` и `activation_wls` для
`down_proj`; кандидат улучшил оба development-среза, но был отклонён на
audit-v7 из-за cumulative SQuAD ratio `1.004673 > 1.002167`.

Следующий эксперимент оставил committed masks неизменными, выполнял уже
принятые группы строго через ternary codes и FP16 g128 scales, а градиенты
разрешил только ещё не принятым BF16-группам `up/gate/down`. Raw-BF16 teacher
дал отрицательный control; target-local counterfactual teacher выбрал шаг 704
и улучшил worst selection/confirmation на `0.001203 / 0.001223`.

Замороженный artifact проверен на заранее построенном audit-v8. Все три
incremental upper-95 ratios оказались ниже единицы, source восстановился
побитно, а исходный checkpoint не мутировал. Atomic commit опубликовал
`s0048r3`; aggregate relative fallback delta равна `0.00315059`, coverage
остаётся `74,563,584` weights. После fresh-process verification parent и
промежуточные бинарные artifacts удалены, а JSON evidence и hashes сохранены.

### Новый prospective atom и frontier `s0049`

До выбора следующего кандидата построен audit-v9. Первая заранее объявленная
попытка диапазона была технически невозможна из-за длины SQuAD stream и не
создала suite. Повторная policy выбрала отдельные свободные диапазоны для
C4/SQuAD/torch. Улучшенный overlap checker восстановил интервалы legacy
audit-v2--v6 из offsets и source identities; v9 имеет ноль exact-window и
range overlaps со всеми 18 retained suites.

На development заново просканированы 96 D1-групп каждой оставшейся MLP
матрицы после fallback-QAT. Победил `gate_proj + threshold_ls` с worst
incremental ratio `1.000053006`. Hard-forward proxy-QAT не превзошёл исходную
инициализацию, поэтому корректно сохранил `step 0`, churn codes равен нулю.

Точный frozen artifact единожды проверен на audit-v9. Incremental upper-95
C4/SQuAD/code равны `1.000039151 / 1.000027537 / 1.000000139`, все ниже
заранее заданного `1.0002`. Atomic commit `s0049` добавил 96 g128-групп, или
12,288 strict-ternary weights. Fresh reload прошёл; parent `s0048r3` и четыре
промежуточных artifact-файла удалены только после этого.

### Ускорение атома и frontier `s0050`

До разработки следующего кандидата построен audit-v10 на новых диапазонах
C4/SQuAD/PyTorch code. Его hash и policy зафиксированы заранее, а overlap
checker доказал нулевое пересечение со всеми 19 retained suites.

Development scan увеличил атом с 96 до 2,048 g128-групп: 262,144 weights за
шаг, или в 21.33 раза больше. Среди девяти заранее объявленных комбинаций
`up/down/gate x absmean/threshold-LS/activation-WLS` победил
`gate_proj + activation_wls` с worst incremental ratio `1.000140889`.
Proxy recovery выбрал шаг 128, улучшил worst full-development ratio до
`1.000094505` и сохранил code churn равным нулю.

Точный frozen artifact единожды проверен на audit-v10. Incremental upper-95
C4/SQuAD/code равны `1.000039423 / 1.000335574 / 1.000147976`, все ниже
заранее заданного size-adjusted лимита `1.000923760`. Atomic commit `s0050`
добавил 2,048 g128-групп. Fresh reload дал `wiki=0.947148` и `code=0.959176`;
accepted coverage теперь 74,838,016 weights, или 4.349907%.

### Coverage-neutral recovery и frontier `s0050r1`

Новый audit-v11 отклонил следующий 2,048-group candidate и одновременно
показал, что сам `s0050` имеет SQuAD-train ratio `1.007967` на этом новом
срезе. Кандидат не был принят. Recovery обучал только непринятые BF16-группы
`layer24.up/gate/down`, не меняя ни одного принятого ternary-code, scale или
mask. Первый запуск узко не прошёл заранее заданный confirmation improvement;
второй использовал новую development-suite с нулевым overlap и прошёл.

Замороженный artifact один раз проверен на audit-v12. Абсолютные C4/SQuAD/
Transformers-code ratios равны `0.994325 / 0.995866 / 0.990334`, а
incremental upper-95 относительно `s0050` —
`0.998983 / 0.998855 / 0.997520`. Atomic checkpoint `s0050r1` сохранил
coverage `74,838,016` и fresh reload дал `wiki=0.946518`, `code=0.957625`.
После проверки parent и полностью интегрированные/rejected artifacts удалены.

### Новый coverage atom и frontier `s0051`

До просмотра модели на audit-v13 был построен новый C4/SQuAD/Transformers-code
suite и доказано нулевое пересечение с 24 retained suites. Development scan
сравнил девять `up/down/gate x initializer` arms и выбрал 2,048 D1-групп
`gate_proj + activation_wls`. Proxy recovery сохранил code churn равным нулю;
худший development incremental ratio равен `1.000075`.

На единственном audit-v13 cumulative worst point ratio равен `1.000515` при
пределе `1.002183`, а incremental worst upper-95 равен `1.000245` при заранее
заданном пределе `1.000924`. Atomic checkpoint `s0051` добавил 262,144
strict-ternary weights, поднял coverage до `75,100,160` (`4.365144%`) и
fresh-verifies at `wiki=0.946587`, `code=0.957546`. После fresh reload parent
и четыре интегрированных/rejected scan artifacts удалены с сохранением их
SHA-256 и полного JSON lineage.

После этого был проверен ускоренный атом из 8,192 групп `down_proj`. Он прошёл
development и cumulative point gates audit-v14, но code incremental upper-95
оказался `1.001969` при пределе `1.001848`; кандидат отклонён. Заранее
уменьшенный вдвое атом из 4,096 групп прошёл incremental audit-v15 с максимумом
`1.001094 <= 1.001306`, однако audit-v15 показал pre-existing SQuAD ratio
`1.004861` у самого `s0051`; candidate ratio `1.005260` нарушил cumulative
предел `1.002198`. Этот кандидат был отклонён.

Две coverage-neutral recovery стадии на непересекающихся development/audit
наборах сформировали `s0051r2`. Из него тот же заранее зафиксированный
4,096-group `down_proj` атом прошёл audit-v19: cumulative worst observed ratio
`0.995173`, incremental upper-95 worst `1.000710 <= 1.001306`. Checkpoint
`s0052` добавил 524,288 strict-ternary weights, подняв `down_proj` до
`14.615885%`, а общий coverage — до `75,624,448` (`4.395617%`).

Следующая 8,192-group попытка прошла incremental gate audit-v20, но была
отклонена из-за cumulative SQuAD `1.005280 > 1.002198`. Двухстадийный
coverage-neutral recovery затем был проверен на новых audit-v21/v22. Первая
скорость обучения всё ещё не прошла cumulative SQuAD; продолжение с половинным
LR прошло audit-v22 с C4/SQuAD/NumPy-code point ratios
`0.991022 / 1.001470 / 0.972487` и incremental upper-95 worst `0.998832`.
Опубликованный `s0052r1` не изменил coverage или ternary codes и fresh-verifies
at `wiki=0.943734`, `code=0.953158`.

### Reference Q2-g128 packer и реальный bpw

Добавлен versioned binary format без pickle overhead. Он хранит mapping
`00=-1`, `01=0`, `10=+1`, `11=reserved`, FP16 scale на группу, а для частичной
матрицы — bit-packed committed mask и только BF16 fallback-группы. Loader
сразу отклоняет reserved code, проверяет длины payload и точно восстанавливает
deployed weight.

На настоящем `layer24.up_proj` (его mask coverage не изменилось в `s0050r1`):

- shape: `6144 x 2048`, всего 12,582,912 weights;
- committed: 74,496 из 98,304 групп, то есть 75.78125%;
- фактический файл: `8,640,072` bytes;
- payload: `8,640,000` bytes, versioned header: 72 bytes;
- true file bpw: `5.493209839`;
- codes, scales, mask и итоговая deployed-матрица после read-back совпали
  побитно/точно.

Почему здесь не 2.125 bpw: четверть этой конкретной матрицы всё ещё хранится
как BF16 fallback. Полностью committed g128-матрица имеет ровно `2.125` payload
bpw плюс исчезающе малый header. Для всего текущего frontier только 4.395617%
major weights уже Q2, поэтому честная проекция major-weight storage пока
`15.390108 bpw`, или около `3.082 GiB` вместо `3.205 GiB` BF16. При 100%
Q2-g128 те же 1,720,451,072 major weights занимали бы около `0.426 GiB` без
runtime/KV-cache. Это storage projection, не текущая VRAM тренировки и не
готовый `llama.cpp` kernel.

## Лучший воспроизводимый checkpoint

```text
wal2/checkpoints/wal-tat-block24_mlp_laterange_s0052r1.pt
```

- размер: `304,659,922` bytes;
- SHA-256: `9a7fb038694f2b850eb95b83fd96204aa882c064fbe89df30e8f1f7f9ed67411`;
- содержание: полный ternary block 27, Q/K/V/O block 24, 75.78125%
  `up_proj`, 10.611979% `gate_proj` и 14.615885% `down_proj`;
- формат: training checkpoint, не packed artifact.

Промежуточные и провалившие audit checkpoints удалены; их метрики и команды
сохранены в `results/`. Повторное получение требует перезапуска эксперимента.

## Следующий технический шаг

1. построить свежие development-v8 и sealed audit-v23, доказав их нулевое
   пересечение со всеми ранее использованными suites;
2. повторить 8,192-group `down_proj` atom из восстановленного `s0052r1`, не
   раскрывая audit-v23 до заморозки кандидата;
3. при отказе уменьшить атом или выполнить ещё один coverage-neutral recovery,
   не меняя принятые ternary code/scale/mask;
4. перенести prospective dual gate из отдельного commit validator в основной
   campaign controller;
5. расширить готовый reference Q2-g128 packer до full-checkpoint manifest и
   проверить logit/NLL equality после загрузки нескольких связанных матриц;
6. завершить второй block и затем проверить ранний чувствительный block, а не
   идти только по easy-first карте.
