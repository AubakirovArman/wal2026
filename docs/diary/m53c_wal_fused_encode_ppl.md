# M53C WAL FUSED ENCODE PPL

## Date
2026 (exact date from git log or experiment run)

## Goal
M53c: WAL-0 with fused Triton encode + PPL on Llama 3.3 70B.

## Configuration
K_ATOMS=128, KMEANS_ITERS=5, num_steps=0

## Method / What was tested
See `experiments/m53c_wal_fused_encode_ppl.py` for implementation details.

## Result
PPL evaluation.

## Artifacts
- `experiments/m53c_wal_fused_encode_ppl.py`
- `experiments/m53c_wal_fused_encode_ppl.log`

## Notes from dev_diary_ru.md
No specific mention in dev_diary_ru.md.

## Detailed Notes from dev_diary_ru.md

### Mention 1

```text

- M53a/b: Fused Triton encode kernel (один kernel делает полный greedy search).
- **Результат**: 7.7 Gw/s vs 24.9 Mw/s PyTorch — **309× speedup** на уровне kernel.
- M53c: Full 70B encode с fused kernel.
- **Проблема**: `invalid resource handle` при переключении GPU (cuda:2 → cuda:3).
- **Fix**: `with torch.cuda.device(device):` перед запуском Triton kernel.
- **Результат M53c**: PPL 2.7858, encode time 2225s (vs 2715s PyTorch). **Ускорение 1.22× overall**.
- **Главный bottleneck**: k-means (построение atom table), не сам encode.
- **Вывод**: fused kernel работает на полном масштабе. Но для реального ускорения нужно оптимизировать k-means (cross-layer sharing, global atoms).
- Полный отчёт: `docs/diary/m53_fused_triton_encode.md`

---

## Текущий фокус: WAL-0 → Язык (M54+)

M46-M53 доказали: WAL-0 scalar — отличный **кодек**. Качество уровня dense, decode быстрый, encode масштабируется.

Но это ещё не **язык**. Нужно три вещи:
```

### Mention 2

```text
- M53c: Full 70B encode с fused kernel.
- **Проблема**: `invalid resource handle` при переключении GPU (cuda:2 → cuda:3).
- **Fix**: `with torch.cuda.device(device):` перед запуском Triton kernel.
- **Результат M53c**: PPL 2.7858, encode time 2225s (vs 2715s PyTorch). **Ускорение 1.22× overall**.
- **Главный bottleneck**: k-means (построение atom table), не сам encode.
- **Вывод**: fused kernel работает на полном масштабе. Но для реального ускорения нужно оптимизировать k-means (cross-layer sharing, global atoms).
- Полный отчёт: `docs/diary/m53_fused_triton_encode.md`

---

## Текущий фокус: WAL-0 → Язык (M54+)

M46-M53 доказали: WAL-0 scalar — отличный **кодек**. Качество уровня dense, decode быстрый, encode масштабируется.

Но это ещё не **язык**. Нужно три вещи:

1. **Codebook layer** (M54): unique programs → IDs. Как в DRL v2: route → ID.
2. **Variable length** (M55): early stopping, stop_depth. Не фиксированный lmax=2.
```

### Mention 3

```text

- Full end-to-end encode всех 540 params с codebook layer.
- **PPL: 2.7828** — gap vs baseline 2.7805: **+0.08%** (+0.0023 nats).
- Сравнение: M46=2.7821, M53c=2.7858. Codebook не портит качество.
- **Время encode: 437 секунд (7.3 минуты)** — в 6× быстрее M53c (2225s) и M46 (2715s)!
- **Total unique programs: 609,643** на всю модель, avg 1129 per layer.
- **Баг found и fixed**: `codebook_recon` был проиндексирован по `unique_prog` порядку, а `program_ids` — по `sort_idx` (frequency sort). Это давало catastrophic PPL 299,002 на первом запуске. Fix: использовать `inverse` от `torch.unique` напрямую.
- **Вывод**: WAL-0 с codebook layer масштабируется на полную модель, сохраняет качество, и encode в 6× быстрее.


```


## Known Results (from project context)

**Result:** Fused encode + full 70B PPL validation

**Notes:** Verified that fused Triton encode produces identical results to reference encoder.


## Extracted Metrics (from source)

- Elapsed: .0
