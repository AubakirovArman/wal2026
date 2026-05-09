"""Lightweight cross-model validation helpers.

This module intentionally avoids model loading. It only discovers local model
manifests and records whether a controlled runner can proceed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


WORKFLOW_STEPS = [
    "init",
    "add_recipe",
    "build",
    "exact_check",
    "negative_check",
    "context_check",
    "tag",
    "bad_edit",
    "ci_fail",
    "blame_or_bisect",
    "rollback",
    "release_notes",
]


@dataclass
class ModelCandidate:
    path: str
    size_gb: float
    model_type: str | None
    architectures: list[str]
    matched_terms: list[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_search_roots() -> list[Path]:
    root = repo_root()
    return [
        root / ".hf_cache",
        Path("/mnt") / "hf_model_weights",
    ]


def safe_read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def directory_size_gb(path: Path) -> float:
    total = 0
    for file_path in path.glob("*"):
        if file_path.is_file():
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return round(total / (1024**3), 3)


def iter_config_dirs(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            parent = config.parent
            if parent in seen:
                continue
            seen.add(parent)
            yield parent


def discover_candidates(
    include_terms: list[str],
    exclude_terms: list[str],
    max_size_gb: float,
    roots: Iterable[Path] | None = None,
) -> tuple[list[ModelCandidate], list[ModelCandidate]]:
    candidates: list[ModelCandidate] = []
    near_misses: list[ModelCandidate] = []
    roots = list(roots or default_search_roots())
    for model_dir in iter_config_dirs(roots):
        lower_path = str(model_dir).lower()
        matched = [term for term in include_terms if term.lower() in lower_path]
        if not matched:
            continue
        config = safe_read_json(model_dir / "config.json")
        model_type = config.get("model_type")
        architectures = config.get("architectures") or []
        if not isinstance(architectures, list):
            architectures = [str(architectures)]
        size_gb = directory_size_gb(model_dir)
        candidate = ModelCandidate(
            path=str(model_dir),
            size_gb=size_gb,
            model_type=str(model_type) if model_type is not None else None,
            architectures=[str(item) for item in architectures],
            matched_terms=matched,
        )
        excluded = any(term.lower() in lower_path for term in exclude_terms)
        too_large = size_gb > max_size_gb
        if excluded or too_large:
            near_misses.append(candidate)
        else:
            candidates.append(candidate)
    return sorted(candidates, key=lambda item: (item.size_gb, item.path)), sorted(
        near_misses,
        key=lambda item: (item.size_gb, item.path),
    )


def model_workflow_result(
    module: str,
    name: str,
    family: str,
    candidates: list[ModelCandidate],
    near_misses: list[ModelCandidate],
    required_models: int = 1,
) -> dict[str, object]:
    if len(candidates) >= required_models:
        status = "BLOCKED"
        reason = "REAL_MODEL_WORKFLOW_REQUIRES_CONTROLLED_RUNNER"
    else:
        status = "BLOCKED"
        reason = "LOCAL_SMALL_TEXT_MODEL_NOT_FOUND"
    return {
        "schema_version": "wal.results.v1",
        "module": module,
        "name": name,
        "status": status,
        "pass": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "reason": reason,
        "required_workflow": WORKFLOW_STEPS,
        "required_models": required_models,
        "candidate_count": len(candidates),
        "near_miss_count": len(near_misses),
        "candidates": [asdict(item) for item in candidates],
        "near_misses": [asdict(item) for item in near_misses[:20]],
    }


def write_result(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
