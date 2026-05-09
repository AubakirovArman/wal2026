"""Canonical WAL experiment status taxonomy."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping


class ExperimentStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"
    UNSUPPORTED = "UNSUPPORTED"
    SIMULATED = "SIMULATED"
    NO_DATA = "NO_DATA"
    DOC_ONLY = "DOC_ONLY"


ALLOWED_STATUSES = frozenset(status.value for status in ExperimentStatus)
FAILURE_STATUSES = frozenset(
    {
        ExperimentStatus.FAIL.value,
        ExperimentStatus.BLOCKED.value,
        ExperimentStatus.UNSUPPORTED.value,
        ExperimentStatus.NO_DATA.value,
    }
)


def normalize_status(value: Any) -> str | None:
    """Normalize legacy status spellings to the canonical taxonomy."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    upper = raw.upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "TRUE": ExperimentStatus.PASS.value,
        "OK": ExperimentStatus.PASS.value,
        "HEALTHY": ExperimentStatus.PASS.value,
        "SUCCESS": ExperimentStatus.PASS.value,
        "FALSE": ExperimentStatus.FAIL.value,
        "ERROR": ExperimentStatus.FAIL.value,
        "FAILED": ExperimentStatus.FAIL.value,
        "PRE_ALPHA": ExperimentStatus.SIMULATED.value,
        "PRE_ALPHA,_PUBLICATION_READY": ExperimentStatus.SIMULATED.value,
        "PRE_ALPHA,_SYSTEM_VALIDATED,_PUBLICATION_READY": ExperimentStatus.SIMULATED.value,
    }
    candidate = aliases.get(upper, upper)
    if candidate in ALLOWED_STATUSES:
        return candidate
    return None


def derive_status(record: Mapping[str, Any]) -> str:
    """Derive a conservative canonical status for legacy result records."""
    explicit = normalize_status(record.get("status"))
    if explicit is not None:
        return explicit

    if isinstance(record.get("pass"), bool):
        return ExperimentStatus.PASS.value if record["pass"] else ExperimentStatus.FAIL.value
    if isinstance(record.get("overall_pass"), bool):
        return ExperimentStatus.PASS.value if record["overall_pass"] else ExperimentStatus.FAIL.value
    if isinstance(record.get("healthy"), bool):
        return ExperimentStatus.PASS.value if record["healthy"] else ExperimentStatus.FAIL.value

    error = str(record.get("error") or "").lower()
    if error:
        if "unsupported" in error or "unrecognized configuration" in error:
            return ExperimentStatus.UNSUPPORTED.value
        if "out of memory" in error or "oom" in error:
            return ExperimentStatus.BLOCKED.value
        return ExperimentStatus.FAIL.value

    if record.get("simulated") is True or record.get("simulation") is True:
        return ExperimentStatus.SIMULATED.value
    if record.get("doc_only") is True:
        return ExperimentStatus.DOC_ONLY.value

    return ExperimentStatus.PASS.value


def status_to_pass(status: str) -> bool:
    """Map canonical status to legacy boolean pass semantics."""
    normalized = normalize_status(status) or status
    return normalized not in FAILURE_STATUSES
