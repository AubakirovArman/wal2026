"""Result-file validation and normalization helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .status import ALLOWED_STATUSES, derive_status, normalize_status, status_to_pass


SCHEMA_VERSION = "wal.results.v1"


@dataclass
class ResultValidation:
    total: int = 0
    valid: int = 0
    invalid: int = 0
    warnings: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    invalid_files: list[dict[str, str]] = field(default_factory=list)
    warning_files: list[dict[str, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.invalid == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "warnings": self.warnings,
            "status_counts": dict(sorted(self.status_counts.items())),
            "invalid_files": self.invalid_files,
            "warning_files": self.warning_files,
            "pass": self.passed,
            "status": "PASS" if self.passed else "FAIL",
        }


def result_files(root: str | Path) -> list[Path]:
    """Return experiment result JSON files under ``root``."""
    root_path = Path(root)
    return sorted(root_path.glob("*_results.json"))


def normalize_result_payload(payload: Any, source: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """Normalize legacy result payloads into the WAL result schema."""
    warnings: list[str] = []
    if isinstance(payload, list):
        warnings.append("legacy_list_wrapped")
        record_statuses = [
            derive_status(item)
            for item in payload
            if isinstance(item, dict)
        ]
        status = "PASS" if all(status_to_pass(status) for status in record_statuses) else "FAIL"
        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "pass": status_to_pass(status),
            "record_count": len(payload),
            "records": payload,
            "source": source,
        }, warnings

    if not isinstance(payload, dict):
        raise TypeError(f"top-level JSON must be object or list, got {type(payload).__name__}")

    normalized = dict(payload)
    if "status" in normalized:
        status = normalize_status(normalized.get("status"))
        if status is None:
            status = str(normalized.get("status"))
        elif normalized.get("status") != status:
            warnings.append("status_normalized")
            normalized["status"] = status
    else:
        status = derive_status(normalized)
        normalized["status"] = status
    if "pass" not in normalized:
        warnings.append("pass_derived")
        normalized["pass"] = status_to_pass(status)
    if "schema_version" not in normalized:
        warnings.append("schema_version_added")
        normalized["schema_version"] = SCHEMA_VERSION
    return normalized, warnings


def validate_result_file(path: str | Path, strict: bool = False) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """Validate one result file and return normalized data, warnings, and error."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return None, [], f"json_parse_error: {exc}"

    try:
        normalized, warnings = normalize_result_payload(payload, source=path.name)
    except Exception as exc:
        return None, [], str(exc)

    status = normalize_status(normalized.get("status"))
    if status not in ALLOWED_STATUSES:
        return None, warnings, f"invalid_status: {normalized.get('status')!r}"

    if strict:
        strict_missing = [
            key
            for key in ("schema_version", "status", "pass")
            if key not in normalized
        ]
        if strict_missing:
            return None, warnings, "missing_required_keys: " + ",".join(strict_missing)

    return normalized, warnings, None


def validate_results(root: str | Path, strict: bool = False) -> ResultValidation:
    """Validate all result files under ``root``."""
    summary = ResultValidation()
    root_path = Path(root)
    for path in result_files(root):
        summary.total += 1
        normalized, warnings, error = validate_result_file(path, strict=strict)
        try:
            rel_path = str(path.relative_to(root_path))
        except ValueError:
            rel_path = str(path)
        if error:
            summary.invalid += 1
            summary.invalid_files.append({"path": rel_path, "error": error})
            continue

        assert normalized is not None
        status = str(normalized["status"])
        summary.status_counts[status] = summary.status_counts.get(status, 0) + 1
        summary.valid += 1
        if warnings:
            summary.warnings += len(warnings)
            summary.warning_files.append({"path": rel_path, "warnings": ",".join(warnings)})
    return summary


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
