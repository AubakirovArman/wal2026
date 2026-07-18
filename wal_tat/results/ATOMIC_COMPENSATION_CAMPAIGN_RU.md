# Итог atomic compensation campaign

## Результат

Гипотеза подтверждена. WAL-TAT теперь умеет атомарно добавлять ternary-группы в
одну MLP-матрицу и компенсировать их ошибку через связанные g128-scales
`down_proj`.

Текущий frontier Qwen3-1.7B, layer 27:

| Матрица | Ternary coverage | Projected bpw |
|---|---:|---:|
| `down_proj` | 100% | 2.125 |
| `up_proj` | 100% | 2.125 |
| `gate_proj` | 75% | 5.59375 mixed |

Fresh gate на 32,768 токенах каждого домена:

| Модель | Wiki PPL | Code PPL | Wiki NLL ratio | Code NLL ratio |
|---|---:|---:|---:|---:|
| BF16 | 35.7241 | 29.6245 | 1.000000 | 1.000000 |
| WAL-TAT | 33.0867 | 31.5144 | 0.978552 | 1.018250 |

Порог `<=1.02` выполнен.

## Почему это не просто удачный QAT

На одинаковых 1,536 новых группах `up_proj`:

- candidate-only: code ratio `1.020203`, rollback;
- linked-down: `1.019334`, commit.

При попытке сразу перевести `gate_proj` с 50% на 75%:

- candidate-only: `1.021983`, rollback;
- linked-down: `1.020987`, rollback.

Дробление транзакции на 12.5% и затем 6.25% позволило дойти до 75%, не меняя
порог. Значит результат объясняется сразу тремя механизмами: causal order,
linked compensation и адаптивный размер транзакции.

В связанных группах `down_proj` ternary-коды не менялись; менялись scales.
Поэтому обнаруженный новый процесс можно описать как:

```text
candidate ternarization
  + linked scale-only compensation
  + strict multi-domain gate
  + atomic commit/rollback
```

## Что осталось

Ближайшая задача — `gate_proj` 75→100%. После завершения MLP следует attention
окно `Q/K/V → O`, затем остальные decoder blocks. Полная модель, embeddings,
LM head, task-benchmarks, GGUF и runtime пока не завершены.
