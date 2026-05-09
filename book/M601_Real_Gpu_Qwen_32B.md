# M601 — Real Gpu Qwen 32B

## Описание эксперимента
Qwen-VL probe is unsupported through the current AutoModelForCausalLM path; this is not a PASS.

## Исходный файл
`m601_real_gpu_qwen_32b.py`

## Результаты

- **schema_version**: `wal.results.v1`
- **status**: 🚫 UNSUPPORTED
- **pass**: `False`
- **model_loaded**: `False`
- **inference_done**: `False`
- **error**: `Unrecognized configuration class <class 'transformers.models.qwen3_vl.configuration_qwen3_vl.Qwen3VLConfig'> for this kind of AutoModel: AutoModelForCausalLM. Model type should ...`
- **reason**: `UNSUPPORTED_CONFIG`

## Анализ

- **Модуль**: M601
- **Название**: Real Gpu Qwen 32B
- **Дата обновления**: 2026-05-09
- **Статус**: 🚫 UNSUPPORTED

## Лог аудита

- Обновлено вручную в рамках cleanup release M621–M623.
- Исторические данные сохранены; статусная интерпретация приведена к WAL result schema v1.

---

*Запись обновлена после технического аудита.*
