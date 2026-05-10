"""Legacy experiment audit helpers.

The audit layer is intentionally conservative: it classifies old experiment
scripts, records whether modern safe runners can execute them, and separates
historical artifacts from current release claims.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "wal.legacy_audit.v1"

HEAVY_REASONS = {
    "local_model_path",
    "model_load",
    "tokenizer_load",
    "cuda",
    "device_map",
    "triton",
    "dataset_load",
    "hf_download",
    "model_artifact",
}

MUTATION_REASONS = {
    "git_mutation",
    "destructive_file_op",
    "destructive_shell_op",
    "archive_mutation",
    "mass_regeneration",
    "mass_rewrite",
    "mass_export",
}

DOC_REASONS = {
    "public_doc_generator",
    "public_claim_generator",
    "git_metadata_generator",
}

MODEL_PATH_NEEDLES = ("/mnt/hf_model_weights", "models--", ".safetensors", "safe_open")
SEED_NEEDLES = ("manual_seed", "np.random.seed", "random.seed", "set_seed")
RANDOM_NEEDLES = ("torch.randn", "torch.rand", "np.random", "random.")
RESULT_WRITE_NEEDLES = ("json.dump", "write_text", "torch.save")
MAX_FAST_ARTIFACT_HASH_BYTES = 8 * 1024 * 1024


def experiment_number(filename: str) -> int | None:
    match = re.match(r"m(\d+)", filename.lower())
    return int(match.group(1)) if match else None


def experiment_suffix(filename: str) -> str:
    match = re.match(r"m\d+([a-z]*)_", filename.lower())
    return match.group(1) if match else ""


def experiment_order_key(path: Path) -> tuple[int, str, str]:
    number = experiment_number(path.name)
    return (number if number is not None else 999999, experiment_suffix(path.name), path.name)


def experiment_family(number: int | None, filename: str) -> str:
    lowered = filename.lower()
    if number is None:
        return "unversioned_helper"
    if number <= 75:
        return "core_wal_encoding"
    if number <= 150:
        return "edit_and_transform"
    if number <= 220:
        return "spectral_and_wave"
    if number <= 245:
        return "hard_facts_and_memory"
    if number <= 290:
        return "ci_retrieval_and_build"
    if number <= 385:
        return "framework_ops"
    if number <= 620:
        return "platform_and_meta"
    if "claim" in lowered or "readme" in lowered or "docs" in lowered:
        return "release_hardening"
    return "legacy_audit_and_hardening"


def runner_type(blocked_reasons: list[str], static_flags: dict[str, bool], runnable: bool) -> str:
    reason_set = set(blocked_reasons)
    if not runnable:
        if "syntax_error" in reason_set:
            return "invalid_source"
        if "model_small_controlled_runner" in reason_set:
            return "model_small"
        if "runtime_timeout_in_safe_sweep" in reason_set:
            return "slow_safe"
        if reason_set & DOC_REASONS:
            return "docs_public_claims"
        if reason_set & MUTATION_REASONS:
            return "mutation_dry_run"
        if "subprocess" in reason_set:
            return "subprocess_review"
        if reason_set & HEAVY_REASONS:
            return "gpu_or_model_controlled"
        return "blocked_review"
    if static_flags.get("writes_results"):
        return "safe_core_with_artifact"
    return "safe_core"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_if_small(path: Path) -> str | None:
    if path.stat().st_size > MAX_FAST_ARTIFACT_HASH_BYTES:
        return None
    return sha256_file(path)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def load_inventory(repo_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(repo_root / "experiments" / "m624_full_test_inventory_results.json") or {}
    records = payload.get("records", [])
    return {
        record["file"]: record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("file"), str)
    }


def load_sweep(repo_root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(repo_root / "experiments" / "m625_safe_runtime_sweep_results.json") or {}
    records = payload.get("records", [])
    return {
        record["file"]: record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("file"), str)
    }


def static_flags_for(source_text: str) -> dict[str, bool]:
    lowered = source_text.lower()
    return {
        "uses_model_path": any(needle.lower() in lowered for needle in MODEL_PATH_NEEDLES),
        "uses_cuda": "cuda" in lowered or "triton" in lowered,
        "uses_randomness": any(needle.lower() in lowered for needle in RANDOM_NEEDLES),
        "has_explicit_seed": any(needle.lower() in lowered for needle in SEED_NEEDLES),
        "writes_results": any(needle.lower() in lowered for needle in RESULT_WRITE_NEEDLES),
        "uses_fp16_or_bf16": "float16" in lowered or "fp16" in lowered or "bfloat16" in lowered,
        "has_results_schema_v1": "wal.results.v1" in lowered,
    }


def discover_artifacts(repo_root: Path, script_path: Path, source_text: str, limit: int = 24) -> list[dict[str, Any]]:
    stem = script_path.stem
    candidates: list[Path] = []
    for base in (repo_root / "experiments", repo_root / "results"):
        if not base.exists():
            continue
        candidates.extend(sorted(base.glob(f"{stem}*")))
    for match in re.findall(r"results/[A-Za-z0-9_./{}\\-]+(?:\.json|\.pt|\.log)", source_text):
        if "{" in match or "}" in match:
            continue
        candidates.append(repo_root / match)
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.is_file() and candidate not in seen and candidate.name != script_path.name:
            seen.add(candidate)
            unique_paths.append(candidate)

    artifacts = []
    for artifact_path in unique_paths[:limit]:
        payload = read_json(artifact_path) if artifact_path.suffix == ".json" else None
        artifact_hash = sha256_file_if_small(artifact_path)
        artifacts.append({
            "path": str(artifact_path.relative_to(repo_root)),
            "suffix": artifact_path.suffix,
            "size_bytes": artifact_path.stat().st_size,
            "sha256": artifact_hash,
            "sha256_skipped": artifact_hash is None,
            "schema_version": payload.get("schema_version") if payload else None,
            "status": payload.get("status") if payload else None,
            "pass": payload.get("pass") if payload else None,
        })
    return artifacts


def modernization_recommendations(
    static_flags: dict[str, bool],
    artifact_count: int,
    selected_runner: str,
    category: str,
) -> list[str]:
    recommendations: list[str] = []
    if static_flags["uses_randomness"] and not static_flags["has_explicit_seed"]:
        recommendations.append("add_explicit_seed")
    if static_flags["writes_results"] and not static_flags["has_results_schema_v1"]:
        recommendations.append("write_wal_results_v1")
    if artifact_count == 0:
        recommendations.append("reproduce_or_mark_no_data")
    if selected_runner == "gpu_or_model_controlled":
        recommendations.append("move_to_controlled_gpu_or_model_runner")
    if selected_runner == "slow_safe":
        recommendations.append("move_to_slow_runner_with_timeout_budget")
    if selected_runner == "mutation_dry_run":
        recommendations.append("convert_to_tempdir_dry_run")
    if static_flags["uses_model_path"]:
        recommendations.append("parameterize_model_path_and_record_artifact_hash")
    if static_flags["uses_cuda"]:
        recommendations.append("record_hardware_requirements")
    if static_flags["uses_fp16_or_bf16"]:
        recommendations.append("check_fp32_adapter_or_overflow_control")
    if category in {"edit_and_transform", "hard_facts_and_memory", "ci_retrieval_and_build"}:
        recommendations.append("add_negative_context_lure_ci")
    return sorted(set(recommendations))


def review_status(
    parse_status: str,
    selected_runner: str,
    sweep_status: str | None,
    has_schema_v1_artifact: bool,
) -> str:
    if parse_status != "PASS":
        return "invalid_due_to_source_error"
    if sweep_status == "FAIL":
        return "needs_rewrite"
    if selected_runner.startswith("safe_core") and sweep_status == "PASS":
        return "still_valid" if has_schema_v1_artifact else "still_valid_needs_schema_v1"
    if selected_runner == "gpu_or_model_controlled":
        return "blocked_needs_controlled_model_runner"
    if selected_runner == "model_small":
        return "blocked_needs_model_small_runner"
    if selected_runner == "slow_safe":
        return "blocked_needs_slow_runner"
    if selected_runner == "mutation_dry_run":
        return "blocked_needs_dry_run"
    if selected_runner == "docs_public_claims":
        return "doc_or_meta_only"
    if selected_runner == "subprocess_review":
        return "blocked_needs_subprocess_review"
    if sweep_status == "BLOCKED":
        return "blocked_by_policy"
    return "needs_review"


def public_claim_allowed(selected_status: str, selected_runner: str) -> bool:
    return selected_status == "still_valid" and not selected_runner.startswith("docs")


def build_record(repo_root: Path, script_path: Path, inventory: dict[str, dict[str, Any]], sweep: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_text = script_path.read_text(encoding="utf-8", errors="replace")
    number = experiment_number(script_path.name)
    inventory_record = inventory.get(script_path.name, {})
    sweep_record = sweep.get(script_path.name, {})
    blocked_reasons = sorted(set(inventory_record.get("blocked_reasons", [])))
    parse_status = str(inventory_record.get("parse_status", "UNKNOWN"))
    runnable = bool(inventory_record.get("runnable", False))
    static_flags = static_flags_for(source_text)
    artifacts = discover_artifacts(repo_root, script_path, source_text)
    has_schema_v1_artifact = any(artifact.get("schema_version") == "wal.results.v1" for artifact in artifacts)
    selected_runner = runner_type(blocked_reasons, static_flags, runnable)
    family = experiment_family(number, script_path.name)
    selected_status = review_status(parse_status, selected_runner, sweep_record.get("status"), has_schema_v1_artifact)

    return {
        "file": script_path.name,
        "experiment_number": number,
        "suffix": experiment_suffix(script_path.name),
        "family": family,
        "source_sha256": sha256_file(script_path),
        "parse_status": parse_status,
        "m624_runnable": runnable,
        "m624_blocked_reasons": blocked_reasons,
        "m625_status": sweep_record.get("status"),
        "m625_pass": sweep_record.get("pass"),
        "runner_type": selected_runner,
        "static_flags": static_flags,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "review_status": selected_status,
        "public_claim_allowed": public_claim_allowed(selected_status, selected_runner),
        "modernization_recommendations": modernization_recommendations(
            static_flags=static_flags,
            artifact_count=len(artifacts),
            selected_runner=selected_runner,
            category=family,
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(records),
        "by_family": dict(sorted(Counter(record["family"] for record in records).items())),
        "by_runner_type": dict(sorted(Counter(record["runner_type"] for record in records).items())),
        "by_review_status": dict(sorted(Counter(record["review_status"] for record in records).items())),
        "public_claim_allowed": sum(1 for record in records if record["public_claim_allowed"]),
        "with_artifacts": sum(1 for record in records if record["artifact_count"] > 0),
        "with_schema_v1_artifact": sum(
            1 for record in records
            if any(artifact.get("schema_version") == "wal.results.v1" for artifact in record["artifacts"])
        ),
    }


def build_manifest(repo_root: Path, lower: int | None = None, upper: int | None = None) -> dict[str, Any]:
    experiments_dir = repo_root / "experiments"
    inventory = load_inventory(repo_root)
    sweep = load_sweep(repo_root)
    script_paths = sorted(experiments_dir.glob("*.py"), key=experiment_order_key)
    records = []
    for script_path in script_paths:
        number = experiment_number(script_path.name)
        if lower is not None and (number is None or number < lower):
            continue
        if upper is not None and (number is None or number > upper):
            continue
        records.append(build_record(repo_root, script_path, inventory, sweep))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {"lower": lower, "upper": upper},
        "summary": summarize(records),
        "records": records,
    }
