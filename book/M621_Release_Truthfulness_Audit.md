# M621 — Release Truthfulness Audit

## Описание эксперимента
Audits public wording and false-positive GPU statuses before public release.

## Исходный файл
`m621_release_truthfulness_audit.py`

## Результаты

- **schema_version**: `wal.results.v1`
- **status**: ✅ PASS
- **pass**: `True`
- **checks_total**: `37`
- **checks_passed**: `37`
- **checks_failed**: `0`
- **checks**: `37 items`

## Анализ

- **Модуль**: M621
- **Название**: Release Truthfulness Audit
- **Дата обновления**: 2026-05-09
- **Статус**: ✅ PASS

---

*Запись обновлена после полного sweep-аудита.*

## Дополнение M626-M627

M621 теперь также проверяет current public claim files: `README.md`, `PROJECT_SUMMARY.md`, `TECHNICAL_REPORT.md`, `docs/demo_playbook.md`, `docs/blocked_script_taxonomy.md`, `docs/controlled_runners.md`, `docs/public_claim_policy.md`, `docs/docs_command_smoke.md`, `docs/wal_status_summary.md`, `wal_studio_v01/README.md`, `FINAL_REPORT.html`, `FINAL_REPORT.json`, `WAL_EXPORT.json`, `MILESTONE_v1.2.json`, `MILESTONE_v1.4.json`.

Regex escaping was corrected so the public-file scan detects forbidden active claims outside README as well.
