# M622 — Result Schema Gate

## Описание эксперимента
Runs the WAL result schema validator across experiment result files.

## Исходный файл
`m622_result_schema_gate.py`

## Результаты

- **schema_version**: `wal.results.v1`
- **status**: ✅ PASS
- **pass**: `True`
- **total**: `431`
- **valid**: `431`
- **invalid**: `0`
- **warnings**: `576`
- **status_counts**: `5 keys` (`PASS=414`, `FAIL=5`, `BLOCKED=8`, `SIMULATED=3`, `UNSUPPORTED=1`)
- **experiment**: `M622`
- **gate**: `result_schema`
- **invalid_files**: `0 items`
- **warning_files**: `383 items`

## Анализ

- **Модуль**: M622
- **Название**: Result Schema Gate
- **Дата обновления**: 2026-05-09
- **Статус**: ✅ PASS

---

*Запись обновлена после полного sweep-аудита.*
