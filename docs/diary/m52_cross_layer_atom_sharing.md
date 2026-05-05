# M52 CROSS LAYER ATOM SHARING

## Date
2026 (exact date from git log or experiment run)

## Goal
M52: Cross-layer atom sharing — analyze atom similarity and shared quality.

## Configuration
K=128, iters=5

## Method / What was tested
See `experiments/m52_cross_layer_atom_sharing.py` for implementation details.

## Result
Encode test.

## Artifacts
- `experiments/m52_cross_layer_atom_sharing.py`
- `experiments/m52_cross_layer_atom_sharing.log`

## Notes from dev_diary_ru.md
```
- **Почему**: больше данных → лучше k-means → лучше atoms.
- **Вывод**: глобальный atom table жизнеспособен. Это путь к единому языку для всей модели.
- Полный отчёт: `docs/diary/m52_cross_layer_sharing.md`

## Шаг 27. M53: Fused Triton Encode — 309× speedup, multi-GPU fix, full PPL
```


## Detailed Notes from dev_diary_ru.md

### Mention 1

```text
- **Вывод**: для K=128 compile-time specialization не нужен. Для K=8-32 мог бы дать выигрыш, но не приоритет.
- Полный отчёт: `docs/diary/m51_wal_compiler.md`

## Шаг 26. M52: Cross-layer atom sharing — shared atoms бьют per-layer

- Shared atoms (пул из 8 слоёв) vs per-layer atoms.
- **Результат**: shared atoms **лучше** per-layer до 7.7× по relMSE.
- **Почему**: больше данных → лучше k-means → лучше atoms.
- **Вывод**: глобальный atom table жизнеспособен. Это путь к единому языку для всей модели.
- Полный отчёт: `docs/diary/m52_cross_layer_sharing.md`

## Шаг 27. M53: Fused Triton Encode — 309× speedup, multi-GPU fix, full PPL

- M53a/b: Fused Triton encode kernel (один kernel делает полный greedy search).
- **Результат**: 7.7 Gw/s vs 24.9 Mw/s PyTorch — **309× speedup** на уровне kernel.
- M53c: Full 70B encode с fused kernel.
- **Проблема**: `invalid resource handle` при переключении GPU (cuda:2 → cuda:3).
- **Fix**: `with torch.cuda.device(device):` перед запуском Triton kernel.
```


## Known Results (from project context)

**Result:** Cross-layer atom sharing beats per-layer atoms.

**Notes:** Shared atoms reduce total atom table size without quality degradation.
