# Atomic compensation campaign: итог по layer 27

Гипотеза подтверждена на полном transformer block Qwen3-1.7B. Транзакционные
SwiGLU- и GQA-aware окна последовательно довели все семь крупных матриц layer
27 до 100% hard ternary coverage.

```text
MLP:       down 100%, up 100%, gate 100%
Attention: q 100%, k 100%, v 100%, o 100%
Total:     50,331,648 ternary weights
```

## Что доказала компенсация

- одинаковая порция `up_proj` без связанного `down_proj` дала code ratio
  `1.020203` и rollback, а linked-scale window дала `1.019334` и commit;
- triadic gate/up/down window прошла локальный frontier, где linked-down alone
  получила `1.020076`, а полное окно — `1.019665`;
- GQA-aware attention windows сохранили соответствие между KV-head и двумя
  query-head groups;
- большие шаги откатывались, а дробление транзакции позволяло продолжать без
  изменения локального порога.

## Независимая проверка и отрицательный результат

Сразу после полного block independent audit-v1 был плохим: примерно
`1.051 / 1.047 / 1.047`. Это показало, что успешные маленькие gates
переобучились к своим suites и не гарантируют перенос.

Локализация дала основной дополнительный ущерб в Q/K. Diverse CE recovery,
проверка на полностью других offsets/repos и уменьшение learning rate снизили
audit-v3 ratios до:

| C4 validation | SQuAD contexts | PyTorch code |
|---:|---:|---:|
| 1.014037 | 1.011498 | 0.995163 |

Это проходит старый gate `1.02`, но ещё не проходит строгую пропорциональную
guide-линию полного model budget. Поэтому campaign доказывает работоспособность
одного блока и recovery, а не готовность повторить его 28 раз.

## Прогресс на момент завершения этой campaign

```text
converted decoder blocks:     1 / 28
remaining decoder blocks:    27 / 28
converted major matrices:     7 / 197
remaining major matrices:   190 / 197
major weight coverage:        2.925%
```

Следующая фаза — sensitivity scan, улучшенная activation-aware ternary
инициализация и attention-relation recovery до строгого cumulative budget.

## Последующий результат

После этой campaign был добавлен hard-forward proxy-code recovery. Он улучшил
layer 27, не нарушая ternary forward. Два sensitivity scan затем независимо
выбрали layer 24 следующим. В нём приняты Q/K/V/O, поэтому общий текущий
frontier теперь равен:

```text
complete decoder blocks:      1 / 28
layer 24 matrices:            4 / 7 accepted
major matrices:              11 / 197 accepted
major weight coverage:        3.656864%
```

Audit-v3 ratios: `0.996518 / 0.998454 / 0.976451`.
Audit-v4 ratios: `0.996518 / 0.995346 / 0.974915`.

Подробности и checkpoint зафиксированы в `evidence_v4.json`. Этот раздел
добавлен как продолжение; исходные результаты layer-27 campaign выше сохранены
как историческая трасса.
