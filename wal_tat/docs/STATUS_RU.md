# Что такое WAL-TAT сейчас

## Короткий ответ

WAL-TAT уже переводит настоящие матрицы Qwen3-1.7B в жёсткий ternary
`{-scale, 0, +scale}`, восстанавливает качество hard-forward QAT и принимает
результат только после проверки на отложенных данных. Это рабочий
исследовательский процесс, но ещё не готовая полностью 1.58-bit модель.

Текущий независимо подтверждённый frontier:

| Единица | Принято | Осталось |
|---|---:|---:|
| Полностью ternary decoder blocks | 1 / 28 (`layer 27`) | 27 |
| Матрицы в `layer 24` | Q/K/V/O + 70.01953125% up + 3.125% gate + 6.54296875% down | 29.98046875% up + 96.875% gate + 93.45703125% down |
| Крупные матрицы, включая tied embedding/head | 11 / 197 | 186 |
| Крупные matrix weights | 72,941,568 / 1,720,451,072 | 1,647,509,504 |

Покрытие крупных matrix weights равно `4.239677%`. Нельзя округлять частично
готовый `layer 24` до второго законченного блока: честный счётчик остаётся
`1 полный block + 4/7 матриц следующего`.

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

Принятые 72,941,568 weights занимают 139.125 MiB в BF16. После настоящей
Q2-g128 упаковки их расчётный payload составит 18.4775 MiB, экономия —
120.6475 MiB. Экономия VRAM появится только после packed runtime; fake-quant
обучение её не даёт.

## Что именно преобразовано

`model.layers.27` полностью ternary:

- `mlp.down_proj`, `mlp.up_proj`, `mlp.gate_proj`;
- `self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj`.

В `model.layers.24` независимо приняты:

- `self_attn.v_proj` и `o_proj`;
- `self_attn.q_proj` и `k_proj`.

В `layer 24` приняты 70.01953125% g128-групп `up_proj`, 3.125%
`gate_proj` и 6.54296875% `down_proj`. Остальные 29.98046875% `up_proj`,
96.875% `gate_proj` и 93.45703125% `down_proj` пока остаются BF16.

## Текущий независимый аудит

Сохранённый frontier проверен двумя 131,072-token/domain suites. Audit-v3
использует C4 validation, SQuAD validation contexts и PyTorch code. Audit-v4
заменяет SQuAD/code срезы и использует Transformers code. C4 в этих двух
аудитах одинаков, остальные два домена независимы.

| Audit | C4 NLL ratio | SQuAD NLL ratio | Code NLL ratio |
|---|---:|---:|---:|
| v3 | 0.996645 | 1.000616 | 0.979997 (PyTorch) |
| v4 | 0.996645 | 0.997063 | 0.980045 (Transformers) |

`ratio < 1` означает, что измеренный candidate NLL ниже teacher на этом
наборе. Это хороший результат, но не доказательство, что тернарная модель
«лучше BF16»: конечные выборки, компенсационное обучение и статистический шум
могут давать локальное улучшение. Для финиша нужны большие PPL suites и
task-benchmarks.

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
   диагностическую линию `1 + 0.05*c`. При текущем `c=0.04239677` это
   `1.00211984`.
2. **PPL budget.** Цель `+5% PPL` означает additive NLL budget
   `log(1.05)`, а не `+5% NLL`. Пропорциональный доменный порог:
   `1 + c*log(1.05)/teacher_NLL`.
3. **Task budget.** MMLU, reasoning, code, instruction following и tool use
   проверяются отдельно: хороший PPL не гарантирует сохранение поведения.

Линия, пропорциональная coverage, тоже является нашей консервативной
диагностикой, а не физическим законом. End-to-end recovery может давать
нелинейную компенсацию. Но она не позволяет потратить весь допустимый ущерб
на первых блоках и ошибочно назвать процесс масштабируемым.

Текущий accepted frontier проходит условную `+5% NLL` guide: худший ratio
`1.000616`, при guide `1.00211984`.

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

## Лучший воспроизводимый checkpoint

```text
wal2/checkpoints/wal-tat-block24_up_headroom_growth_s0021-candidate_only.pt
```

- размер: `304,658,813` bytes;
- SHA-256: `6b8428871d6f73ecfc71d7828752363eae61cb558b90dd5ab79885ac56d19f8f`;
- содержание: полный ternary block 27, Q/K/V/O block 24, 70.01953125%
  `up_proj`, 3.125% `gate_proj` и 6.54296875% `down_proj`;
- формат: training checkpoint, не packed artifact.

Промежуточные и провалившие audit checkpoints удалены; их метрики и команды
сохранены в `results/`. Повторное получение требует перезапуска эксперимента.

## Следующий технический шаг

1. выполнить `down_proj +0.390625%` из синхронизированного frontier;
2. выбрать candidate-only или linked arm только по обоим holdout;
3. при отказе уменьшить атом, а при tight headroom снова выполнить recovery;
4. чередовать up/gate/down по holdout headroom, а не доводить одну матрицу
   вслепую до 100%;
5. завершить второй полный block и повторить cumulative audit;
6. идти по карте чувствительности: `23`, `25`, `22`, `26`, `21`, ...;
7. после нескольких устойчивых blocks зафиксировать layout;
8. только затем добавить exporter и `llama.cpp` loader/kernels.
