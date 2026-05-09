# WAL v2: Архитектура Constraint-Based Spatial Encoding

## 1. Мотивация

WAL-0 (M46-M57) доказал, что скалярное кодирование с k-means атомами достигает PPL 2.7828 при базовом 2.7805 (+0.08%). Однако WAL-0 — это кодек, а не язык. Программы WAL-0 i.i.d., не имеют грамматики, не редактируются.

WAL v2 решает эту проблему через:
1. **Более простые программы**: один вызов атома с непрерывным коэффициентом вместо двух вызовов с тернарными знаками.
2. **Human-readable синтаксис**: текстовый формат с парсером и ассемблером.
3. **Exact round-trip**: бинарный ↔ текстовый формат без потерь.

## 2. ISA v2

### 2.1 Структура программы

```
Program ::= AtomCall [Residual]
AtomCall ::= ATOM atom_id COEF coeff_id
Residual ::= LITERAL float16

atom_id: uint8 [0, K-1]
coeff_id: uint4 [0, C-1]
```

Каждый вес кодируется **одним вызовом атома** с коэффициентом и опциональным остатком.

### 2.2 Почему single-call работает

- K=256 атомов × C=16 коэффициентов = **4096 возможных значений** при 12 битах на вес.
- DRL v2 с K=2048 lookup давал PPL 2.40 (M43) — 2048 значений при ~11 битах.
- WAL v2: 4096 значений при 12 битах → **больше expressiveness** при чуть большем размере.
- Результат: **PPL 2.7781** — качество идентично baseline.

### 2.3 Atom table и Coeff table

**Atom table**: K скалярных значений, полученных k-means++ на распределении весов.

**Coeff table**: C уровней квантизации, полученных Lloyd-Max на отношениях `w / best_atom`. Уровни адаптивны и хранятся как float32.

## 3. Encoder v2: Two-Pass Algorithm

### Pass 1: Atom Discovery
- Сэмплирование 1M весов из слоя.
- K-means++ (K=256, 5 итераций).

### Pass 2: Coeff Discovery + Encoding
- Для каждого веса находим лучший атом: `argmin_k |w - atom[k]|`.
- Вычисляем отношение `ratio = w / atom[k]`.
- Запускаем Lloyd-Max на сэмпле отношений (2M сэмплов, 5 итераций) для получения C=16 уровней.
- Для каждого веса: находим лучший `(atom_id, coeff_id)` минимизируя `|w - atom[k] * coeff[c]|`.

### Spatial Smoothness (заготовка для Phase 3)
В текущей версии кодирование независимое по весам. Phase 3 добавит регуляризацию гладкости: поощрение одинаковых программ у соседних весов.

## 4. Grammar v0.1

### BNF
```bnf
<program_stream>   ::= <header> <program>*
<header>           ::= "K" <uint> "C" <uint> "SHAPE" <uint> <uint>
<program>          ::= <atom_call> [<residual>]
<atom_call>        ::= "ATOM" <atom_id> "COEF" <float>
<residual>         ::= "RESIDUAL" <float>
```

### Text Format Example
```wal
; WAL v2 v0.1 — 67,108,864 programs
K 256
C 16
SHAPE 8192 8192

ATOM 120 COEF 0.771360
ATOM 70 COEF -0.716460
ATOM 90 COEF -0.716460
```

### Assembler
- **Input**: WAL text + AtomTable + CoeffTable
- **Output**: ProgramBufferV2
- **Quantization**: `coeff_value` → ближайший `coeff_id` через `np.abs(coeff_values - value).argmin()`

### Disassembler
- **Input**: ProgramBufferV2 + AtomTable + CoeffTable
- **Output**: WAL text
- **Modes**: `full` (все программы) или `unique` (сводка по уникальным)

## 5. Round-Trip Contract

```
binary → disassemble → text → assemble → binary'
assert binary == binary'  # bit-exact
```

M62 доказал: для 10K программ round-trip максимальная ошибка **0.0**. Это означает, что грамматика WAL v2 полная и однозначная.
