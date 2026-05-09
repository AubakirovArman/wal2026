# WAL v2: VM и Runtime

## 1. WAL VM v2 Specification

### 1.1 Регистры
```
ACC : float32   # аккумулятор
PC  : uint32    # программный счётчик (индекс веса)
```

### 1.2 Память
```
ATOM_TABLE[K]     # [K] float32 — общая read-only таблица атомов
COEFF_TABLE[C]    # [C] float32 — общая read-only таблица коэффициентов
PROGRAMS[N]       # N программ (atom_id, coeff_id, residual, has_residual)
OUTPUT[N]         # [N] float32 — восстановленные веса
ROW_SCALES[M]     # [M] float32 — нормализация по строкам
```

### 1.3 Цикл исполнения
```
for PC in 0..N-1:
    ATOM  = ATOM_TABLE[PROGRAMS[PC].atom_id]
    COEF  = COEFF_TABLE[PROGRAMS[PC].coeff_id]
    ACC   = ATOM * COEF
    if PROGRAMS[PC].has_residual:
        ACC += PROGRAMS[PC].residual
    row = PC // COLS
    OUTPUT[PC] = ACC * ROW_SCALES[row]
```

WAL v2 не требует стека — достаточно одного аккумулятора.

## 2. Реализации Runtime

### 2.1 PyTorch Decode (`prog.decode`)
- Векторизованный gather через `atoms[atom_ids] * coeffs[coeff_ids]`
- Наиболее быстрый путь (2.4 Gw/s на H200)

### 2.2 VM Reference Interpreter (`vm_execute`)
- Python-реализация формальной спецификации VM
- Векторизованное исполнение всех программ сразу
- 833 Mw/s — приемлемо для reference interpreter

### 2.3 Triton Kernel (`wal_v2_decode_triton`)
- Каждый thread обрабатывает один вес
- Прямая загрузка atom_id/coeff_id + lookup в таблицах
- 49-115 Mw/s — базовая версия, требует оптимизации shared memory

## 3. Валидация (M63)

### Bit-Exact Equivalence
| Сравнение | Совпадение | Макс. ошибка |
|-----------|-----------|-------------|
| PyTorch vs VM | ✅ | 0.0 |
| PyTorch vs Triton | ✅ | 0.0 |
| PyTorch vs Triton+RS | ✅ | 0.0 |

**Все три пути декодирования дают бит-точно идентичный результат.** Это доказывает:
1. Семантика VM совпадает с PyTorch-реализацией
2. Triton kernel соответствует referenceexactly
3. WAL v2 имеет согласованную модель исполнения на всех backend'ах

### Пропускная способность (слой 40 o_proj, 67M весов)
| Путь | Время | Пропускная способность |
|------|-------|------------------------|
| PyTorch | 0.028s | **2,409.5 Mw/s** |
| VM | 0.081s | 833.5 Mw/s |
| Triton | 1.378s | 48.7 Mw/s |
| Triton + row scales | 0.585s | 114.7 Mw/s |

## 4. Анализ производительности

### Почему PyTorch быстрее Triton
PyTorch использует `torch.index_select` — высокооптимизированную CUDA-операцию gather. Triton kernel делает gather через `tl.load` с per-thread varying addresses, что менее эффективно на GPU.

### Оптимизация Triton
Таблицы atom_table (256 floats = 1KB) и coeff_table (16 floats = 64B) достаточно малы для кэширования в shared memory. Cooperative загрузка таблиц в начале каждого блока должна значительно ускорить kernel.

### Практический вывод
Для production inference **PyTorch decode** (2.4 Gw/s) — оптимальный путь. Triton kernel служит:
1. Формальной спецификацией исполнения
2. Базой для fusion с другими операциями (decode + matmul)
3. Fallback'ом при отсутствии PyTorch

## 5. Артефакты
- `src/wal/v2/vm.py` — VM state + reference interpreter
- `src/wal/v2/triton_kernels.py` — Triton decode kernels
- `experiments/m63_wal_v2_vm_runtime.py`
