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
next block:           layer 24 has Q/K/V/O + 74.4140625% up + 4.58984375% gate + 8.984375% down
major matrices:       11 / 197 accepted, 186 remain
major matrix weights: 73,986,048 / 1,720,451,072 = 4.300387%
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
sensitivity-ranked транзакции также приняты 74.4140625% `up_proj`, первые
4.58984375% `gate_proj` и 8.984375% `down_proj`. Текущее покрытие
`4.300387%`, условная `+5% NLL` guide равна `1.00215019`, худший измеренный
ratio равен `1.001148`.

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
Следующий down-атом добавил 49,152 weights и поднял `down_proj` до
`1.85546875%`. Linked arm выиграл с worst `1.000677098` против
`1.000722296` у candidate-only.
Следующий gate-атом добавил 24,576 weights и поднял `gate_proj` до
`0.78125%`. Linked arm выиграл с worst `1.000642781` против `1.000709085` у
candidate-only.
Следующий up-атом добавил 393,216 weights и поднял `up_proj` до
`51.26953125%`. Linked arm выиграл с worst `1.000961833` против
`1.000986563`; normalized headroom сузился до `0.525152`.
Следующий down-атом добавил 49,152 weights и поднял `down_proj` до
`2.24609375%`. Оба arm-а прошли оба holdout; linked BF16 MLP arm едва
выиграл по независимому worst ratio (`1.001010915` против `1.001012296`).
Normalized headroom равен `0.501272`: следующим выполняется один gate-атом,
затем masked proxy recovery перед новым крупным up-ростом.
Этот gate-атом добавил 24,576 weights и поднял `gate_proj` до `0.9765625%`.
Linked arm выиграл holdout (`1.001018946` против `1.001023001`), normalized
headroom равен `0.497488`. Следующий шаг — masked proxy recovery без изменения
coverage, затем новый cumulative audit.
Recovery прошёл оба cumulative holdout и улучшил worst ratio до
`1.000115086`, сохранив masks, coverage и все непринятые BF16 master weights.
Изменился один активный ternary-код и 476,026 committed scales; 495 изменений
codes под нулевыми masks не участвуют в forward. Normalized headroom вырос до
`0.943243`, поэтому следующий безопасный опыт — `up_proj +3.125%`.
Этот опыт прошёл оба holdout и добавил 393,216 weights: `up_proj` достиг
`54.39453125%`. Candidate-only выиграл у linked arm (`1.000404500` против
`1.000427594`), normalized headroom остался `0.801631`. Следующий шаг
round-robin — `down_proj +0.390625%`.
Этот down-атом также прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `2.63671875%`. Linked arm выиграл (`1.000412414` против
`1.000453858`), normalized headroom равен `0.797892`. Следующий round-robin
атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.3671875%`. Candidate-only выиграл (`1.000953523` против
`1.000995915`), normalized headroom равен `0.535963`. Следующая попытка —
`up_proj +3.125%`; при отказе или узком запасе выполняется recovery.
Попытка не прошла development gate: candidate-only SQuAD ratio равен
`1.002592097`, linked — `1.002554625`, gate — `1.002066271`. Coverage и
принятый checkpoint не изменились, следующий up-размер автоматически уменьшен
до `1.5625%`. Перед повтором выполняется masked proxy recovery.
Recovery v4 сохранил coverage, все codes и непринятые BF16 master weights,
изменив 481,135 committed scales. Оба cumulative holdout пройдены; worst
улучшился с `1.000953523` до `1.000301811`, normalized headroom вырос до
`0.853122`. Следующая up-попытка использует уменьшенную долю `1.5625%`.
Эта уменьшенная транзакция прошла оба holdout, добавила 196,608 weights и
подняла `up_proj` до `59.08203125%`. Linked arm минимально выиграл у
candidate-only (`1.000243471` против `1.000255685`), normalized headroom
равен `0.881842`. Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `3.41796875%`. Candidate-only выиграл у linked arm по
независимому worst (`1.000307810` против `1.000332156`), normalized headroom
равен `0.850722`. Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.5625%`. Linked arm выиграл у candidate-only по независимому
worst (`1.000336331` против `1.000357894`), normalized headroom равен
`0.836946`. Следующий round-robin атом — уменьшенный `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `60.64453125%`. Candidate-only выиграл у linked arm по независимому worst
(`1.000550819` против `1.000677756`), normalized headroom равен `0.733700`.
Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `5.37109375%`. Candidate-only выиграл у linked arm
(`1.000605718` против `1.000739123`), normalized headroom равен `0.711737`.
Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом тоже прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.5390625%`. Candidate-only выиграл у linked arm
(`1.000627927` против `1.000815486`), normalized headroom равен `0.701269`.
Следующий round-robin атом — `up_proj +1.5625%` без предварительного recovery.
Попытка `up_proj +1.5625%` была откатана на development-gate. Уменьшенный
атом `+0.78125%` прошёл оба holdout, добавил 98,304 weights и поднял
`up_proj` до `67.67578125%`. Holdout выбрал linked arm вместо candidate-only
(`1.000638609` против `1.000662713`), normalized headroom равен `0.696599`.
Следующий round-robin атом — `down_proj +0.390625%`.
Следующий down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `5.76171875%`. Независимый аудит выбрал candidate-only вместо
development-победителя linked arm (`1.000809223` против `1.000834380`),
normalized headroom равен `0.615802`. Следующий round-robin атом —
`gate_proj +0.1953125%`.
Следующий gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.734375%`. Candidate-only выиграл у linked arm
(`1.000730373` против `1.000794258`), normalized headroom равен `0.653356`.
Следующий контролируемый атом — `up_proj +0.78125%`.
Этот up-атом прошёл оба holdout, добавил 98,304 weights и поднял `up_proj`
до `68.45703125%`. Candidate-only выиграл у linked arm (`1.000762124` против
`1.000833618`), normalized headroom равен `0.638776`. Следующий round-robin
атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `6.15234375%`. Candidate-only выиграл у linked arm
(`1.000850378` против `1.000987898`), normalized headroom равен `0.597219`.
Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.9296875%`. Holdout минимально предпочёл linked arm
вместо candidate-only (`1.000883711` против `1.000884797`), normalized headroom
равен `0.581573`. Следующий контролируемый атом — `up_proj +0.78125%`.
Этот up-атом прошёл оба holdout, добавил 98,304 weights и поднял `up_proj`
до `69.23828125%`. Candidate-only выиграл у linked arm (`1.001048501` против
`1.001158013`), normalized headroom равен `0.504217`. Следующий round-robin
атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `6.54296875%`. Holdout выбрал candidate-only вместо
development-победителя linked arm (`1.001052724` против `1.001137076`),
normalized headroom равен `0.502556`. Следующий атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `3.125%`. Candidate-only выиграл у linked arm
(`1.001044083` против `1.001077155`), normalized headroom немного вырос до
`0.506806`. Перед следующим up-атомом выполняется coverage-neutral recovery.
Masked proxy recovery v6 сохранил coverage, masks и все непринятые BF16
master weights. Изменился один активный ternary-код в `o_proj`, 494,947
committed scales и только committed master weights. Независимый worst снизился
до `1.000346068`, normalized headroom вырос до `0.836528`. Следующий
контролируемый атом — `up_proj +0.78125%` из восстановленного frontier.
Этот атом прошёл оба holdout, добавил 98,304 hard-ternary weights и поднял
`up_proj` до `70.01953125%`. Development предпочёл linked BF16 MLP arm, но
независимый holdout выбрал candidate-only (`1.000616263` против
`1.000722317`). Normalized headroom равен `0.709288`; следующий round-robin
атом — `down_proj +0.390625%`.
Следующий down/gate/up цикл добавил ещё 172,032 hard-ternary weights. Coverage
последовательно достиг `6.93359375% down`, `3.3203125% gate` и
`70.80078125% up`; все шесть arm-а прошли два holdout, а каждая транзакция
независимо выбрала candidate-only. Финальный worst равен `1.000783641`,
normalized headroom — `0.631200`. Следующий round-robin атом —
`down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `4.98046875%`. Holdout выбрал candidate-only вместо
development-победителя linked arm (`1.000442839` против `1.000456372`),
normalized headroom равен `0.788461`. Следующий round-robin атом —
`gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.34375%`. Candidate-only выиграл у linked arm
(`1.000418275` против `1.000454195`), normalized headroom равен `0.800263`.
Следующий round-robin атом — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `66.89453125%`. Candidate-only выиграл у linked arm
(`1.000487872` против `1.000599281`), normalized headroom равен `0.767662`.
Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `3.80859375%`. Linked arm выиграл у candidate-only по
независимому worst (`1.000555450` против `1.000598652`), normalized headroom
равен `0.731646`. Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.7578125%`. Candidate-only минимально выиграл у linked arm
по независимому worst (`1.000559869` против `1.000562804`), normalized
headroom равен `0.729604`. Следующий контролируемый атом —
`up_proj +1.5625%`; при отказе или tight headroom выполняется recovery.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `62.20703125%`. Candidate-only выиграл у linked arm по независимому worst
(`1.000713586` против `1.000765682`), normalized headroom равен `0.656314`.
Следующий round-robin атом — `down_proj +0.390625%`; при отказе или tight
headroom выполняется recovery.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `4.19921875%`. Linked BF16-MLP arm выиграл у candidate-only по
независимому worst (`1.000811028` против `1.000830712`), normalized headroom
равен `0.609651`. Следующий round-robin атом — `gate_proj +0.1953125%`, после
него выполняется masked proxy recovery перед следующим крупным up-атомом.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.953125%`. Linked BF16-MLP arm выиграл у candidate-only по
независимому worst (`1.000820492` против `1.000865155`), normalized headroom
равен `0.605231`. Следующий шаг — coverage-neutral masked proxy recovery,
после которого можно безопаснее пробовать следующий крупный up-атом.
Masked proxy recovery v5 сохранил coverage и committed masks, изменил только
два уже принятых ternary-кода и перенастроил shared scales. Независимый worst
снизился с `1.000820492` до `1.000171060`, normalized headroom вырос до
`0.917697`. Следующий контролируемый атом — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `63.76953125%`. Linked BF16-MLP arm выиграл у candidate-only по
независимому worst (`1.000192249` против `1.000249673`), normalized headroom
равен `0.907755`. Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `4.58984375%`. Candidate-only минимально выиграл у linked arm
(`1.000247348` против `1.000264837`), normalized headroom равен `0.881399`.
Следующий round-robin атом — `gate_proj +0.1953125%`.
Этот gate-атом прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `2.1484375%`. Candidate-only выиграл у linked arm
(`1.000247756` против `1.000354786`), normalized headroom равен `0.881245`.
Следующий round-robin атом — `up_proj +1.5625%`.
Этот up-атом прошёл оба holdout, добавил 196,608 weights и поднял `up_proj`
до `65.33203125%`. Candidate-only минимально выиграл у linked arm
(`1.000363082` против `1.000371670`), normalized headroom равен `0.826441`.
Следующий round-robin атом — `down_proj +0.390625%`.
Этот gate-атом также прошёл оба holdout, добавил 24,576 weights и поднял
`gate_proj` до `1.171875%`. Candidate-only выиграл и development, и holdout:
`1.000480310` против `1.000712729` у linked arm. При динамическом gate
`1.002041273` normalized headroom равен `0.764701`, поэтому следующий
round-robin атом — `up_proj +3.125%`.
Этот up-атом прошёл оба holdout, добавил 393,216 weights и поднял `up_proj`
до `57.51953125%`. Linked arm минимально выиграл у candidate-only по
независимому worst (`1.000862859` против `1.000871839`), normalized headroom
равен `0.579647`. Следующий round-robin атом — `down_proj +0.390625%`.
Этот down-атом прошёл оба holdout, добавил 49,152 weights и поднял
`down_proj` до `3.02734375%`. Linked arm выиграл (`1.000977218` против
`1.000998762`), normalized headroom равен `0.524267`. Следующий round-robin
атом — `gate_proj +0.1953125%`.

Последнее проверенное продолжение добавило девять транзакций после frontier
`s0022`: `down/gate s0017`, `up s0023`, полный цикл
`up s0024 -> down/gate s0018` и полный цикл
`up s0025 -> down/gate s0019`. Всего добавлено 516,096 hard-ternary weights.
Каждый accepted checkpoint был fresh-reload проверен на development suite и
на двух независимых holdout suites. Финальный `gate s0019` выбрал
`candidate_bf16_mlp` с worst `1.001126802` против `1.001156927` у
candidate-only; coverage обоих arm одинаков. Все три campaign-state
синхронизированы на checkpoint SHA
`49210f7ad73b22bb7a0beec96cf516ad33fcce6aab0e36cc3869bc2defef1930`.
Следующий round-robin атом — `up_proj +0.78125%` (`s0026`).

`s0026` был корректно отклонён на development gate и не изменил frontier.
Автоматически уменьшенный `up s0027 +0.390625%`, затем
`down s0020 +0.390625%` и `gate s0020 +0.1953125%` прошли оба holdout и
добавили 122,880 hard-ternary weights. Финальный checkpoint имеет SHA
`eeb870dbce2d65b73899945454bda00b22ee9e16a9808591aedb3f6d3e00fdf3`;
следующий атом — `up_proj s0028 +0.390625%`.

Продолжение `s0028..s0030`, `down s0021..s0023` и `gate s0021..s0022`
добавило 147,456 hard-ternary weights. Две слишком крупные транзакции
(`down s0021` и `up s0029`) были отклонены до изменения frontier; шесть
уменьшенных транзакций прошли fresh reload и оба holdout. Текущий checkpoint
имеет SHA `f61762fcefef7bac94b3cd9f2dbb1072087904500a109a2f66ef0145a2035215`,
coverage `74.12109375% up / 8.7890625% down / 4.39453125% gate` и worst
`1.001663819` при guide `1.002147694`. Normalized headroom `0.225300`, поэтому
следующий этап — coverage-neutral masked proxy recovery v7, затем минимальные
атомы `0.09765625%`.

Recovery v7 прошёл оба holdout без изменения coverage, committed masks и
непринятых BF16 weights. Он снизил worst до `1.001092666`. После него четыре
минимальные транзакции `up s0031`, `down s0024`, `gate s0023`, `up s0032`
добавили 49,152 hard-ternary weights и также прошли оба holdout. Текущий
checkpoint имеет SHA
`b5228f1ce45f292030042f1b2d504703b89404bf434b7c20780f73c1d55bb20f`,
coverage `74.31640625% up / 8.88671875% down / 4.4921875% gate`, worst
`1.001076137` при guide `1.002149122` и normalized headroom `0.499267`.
Следующий round-robin атом — `down_proj s0025 +0.09765625%`.

Следующий цикл `down s0025`, `gate s0024`, `up s0033` добавил ещё 36,864
hard-ternary weights и полностью прошёл два holdout. Все три раза выбран
candidate-only checkpoint. Текущий SHA —
`491fdf27f6b149637358d9b94454deb2ac03e92c6322a05165e1b112f57e4ff8`,
coverage `74.4140625% up / 8.984375% down / 4.58984375% gate`, worst
`1.001147515`, guide `1.002150193`, normalized headroom `0.466320`.
Следующий атом — `down_proj s0026 +0.09765625%`.

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
