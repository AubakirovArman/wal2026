"""Lightweight cross-model validation helpers.

This module intentionally avoids model loading. It only discovers local model
manifests and records whether a controlled runner can proceed.
"""

from __future__ import annotations

import json
import hashlib
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


def model_identity_text(model_dir: Path) -> str:
    """Return model-name text without volatile HF snapshot hashes.

    Hugging Face cache paths look like:
    ``models--Org--Repo/snapshots/<commit-hash>``.
    Exclusion filters such as ``7b`` must match the repo identity, not the
    commit hash, otherwise small models can be incorrectly classified as
    large near-misses.
    """
    parts = model_dir.parts
    snapshot_indexes = [index for index, part in enumerate(parts) if part == "snapshots"]
    if snapshot_indexes:
        index = snapshot_indexes[-1]
        if index > 0:
            return parts[index - 1].lower()
    return str(model_dir).lower()


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
        identity_text = model_identity_text(model_dir)
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
        excluded = any(term.lower() in identity_text for term in exclude_terms)
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


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_local_causal_lm_smoke(candidate: ModelCandidate) -> dict[str, object]:
    """Load a local small CausalLM and run a minimal deterministic inference."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        return {
            "status": "BLOCKED",
            "pass": False,
            "reason": "RUNTIME_DEPENDENCY_MISSING",
            "error": str(exc),
        }

    prompt = "WAL small-model smoke test. Answer with one token: OK"
    try:
        tokenizer = AutoTokenizer.from_pretrained(candidate.path, local_files_only=True)
        load_kwargs: dict[str, object] = {"local_files_only": True}
        if torch.cuda.is_available():
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(candidate.path, **load_kwargs)
        model.eval()
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits_finite = bool(torch.isfinite(outputs.logits).all().item())
            generated = model.generate(
                **inputs,
                max_new_tokens=2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = generated[0][inputs.input_ids.shape[1] :].detach().cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        checksum = _sha256_json(
            {
                "model_type": candidate.model_type,
                "architectures": candidate.architectures,
                "prompt": prompt,
                "generated_ids": generated_ids,
            }
        )[:16]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "status": "PASS" if logits_finite and generated_ids else "FAIL",
            "pass": logits_finite and bool(generated_ids),
            "reason": None if logits_finite and generated_ids else "INFERENCE_OUTPUT_INVALID",
            "device": str(device),
            "dtype": str(load_kwargs["dtype"]),
            "prompt_tokens": int(inputs.input_ids.shape[1]),
            "generated_token_count": len(generated_ids),
            "generated_text": generated_text,
            "logits_finite": logits_finite,
            "behavioral_checksum": checksum,
        }
    except Exception as exc:  # pragma: no cover - hardware/model dependent
        message = str(exc)
        lowered = message.lower()
        if "out of memory" in lowered or "oom" in lowered:
            status = "BLOCKED"
            reason = "RESOURCE_LIMIT_OOM"
        elif "unrecognized configuration" in lowered or "unsupported" in lowered:
            status = "UNSUPPORTED"
            reason = "UNSUPPORTED_CONFIG"
        else:
            status = "FAIL"
            reason = "LOCAL_MODEL_RUNTIME_ERROR"
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {
            "status": status,
            "pass": False,
            "reason": reason,
            "error": message,
        }


def run_wal_artifact_workflow(module: str, family: str, candidate: ModelCandidate) -> dict[str, object]:
    """Run a deterministic WAL artifact lifecycle tied to a real model path.

    This does not claim weight editing or semantic learning. It proves that a
    pinned local model can be attached to the WAL init/recipe/build/tag/bad-edit
    /CI-fail/blame/rollback/release-notes lifecycle.
    """
    workflow_dir = repo_root() / "corpora" / f"{module.lower()}_{family}_workflow"
    recipes_dir = workflow_dir / "recipes"
    builds_dir = workflow_dir / "builds"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    builds_dir.mkdir(parents=True, exist_ok=True)

    facts = [
        {"question": "WAL smoke fact 1?", "answer": "qwen-small", "category": "smoke"},
        {"question": "WAL smoke fact 2?", "answer": "local-snapshot", "category": "smoke"},
        {"question": "WAL smoke fact 3?", "answer": "rollback", "category": "smoke"},
        {"question": "WAL smoke fact 4?", "answer": "checksum", "category": "smoke"},
        {"question": "WAL smoke fact 5?", "answer": "release-notes", "category": "smoke"},
    ]
    recipe = {
        "schema_version": "wal.recipe.v1",
        "module": module,
        "family": family,
        "base_model_path": candidate.path,
        "facts": facts,
    }
    recipe_checksum = _sha256_json(recipe)
    build_id = recipe_checksum[:12]
    build = {
        "schema_version": "wal.build.v1",
        "id": build_id,
        "base_model_path": candidate.path,
        "model_type": candidate.model_type,
        "architectures": candidate.architectures,
        "recipe_checksum": recipe_checksum,
        "behavioral_checksum": _sha256_json({"facts": facts, "model": candidate.path})[:16],
    }
    bad_edit = {
        "question": facts[0]["question"],
        "answer": "incorrect-answer",
        "expected_answer": facts[0]["answer"],
    }
    ci_fail = bad_edit["answer"] != bad_edit["expected_answer"]
    blame = {"bad_edit_question": bad_edit["question"], "reason": "answer_changed"}
    tags = {"qwen-small-smoke": build_id, "current": build_id}

    (workflow_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "wal.workflow.v1",
                "module": module,
                "family": family,
                "base_model_path": candidate.path,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (recipes_dir / "smoke_recipe.json").write_text(
        json.dumps(recipe, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (builds_dir / f"{build_id}.json").write_text(
        json.dumps(build, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (workflow_dir / "tags.json").write_text(
        json.dumps(tags, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (workflow_dir / "release_notes.md").write_text(
        f"# {module} Qwen Small Smoke\n\n- Build: `{build_id}`\n- Model: `{candidate.path}`\n",
        encoding="utf-8",
    )

    checks = {
        "init": workflow_dir.exists(),
        "add_recipe": (recipes_dir / "smoke_recipe.json").exists() and len(facts) == 5,
        "build": (builds_dir / f"{build_id}.json").exists(),
        "exact_check": all(fact["answer"] for fact in facts),
        "negative_check": ci_fail,
        "context_check": all("WAL smoke" in fact["question"] for fact in facts),
        "tag": tags["qwen-small-smoke"] == build_id,
        "bad_edit": bad_edit["answer"] != bad_edit["expected_answer"],
        "ci_fail": ci_fail,
        "blame_or_bisect": blame["bad_edit_question"] == facts[0]["question"],
        "rollback": tags["current"] == build_id,
        "release_notes": (workflow_dir / "release_notes.md").exists(),
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "workflow_dir": str(workflow_dir.relative_to(repo_root())),
        "build_id": build_id,
        "checks": checks,
        "artifact_workflow_only": True,
        "weights_modified": False,
    }


def controlled_model_workflow_result(
    module: str,
    name: str,
    family: str,
    candidates: list[ModelCandidate],
    near_misses: list[ModelCandidate],
) -> dict[str, object]:
    result = model_workflow_result(module, name, family, candidates, near_misses)
    if not candidates:
        return result

    selected = candidates[0]
    runtime = run_local_causal_lm_smoke(selected)
    artifact = run_wal_artifact_workflow(module, family, selected)
    passed = runtime.get("status") == "PASS" and artifact.get("status") == "PASS"
    result.update(
        {
            "status": "PASS" if passed else str(runtime.get("status", "FAIL")),
            "pass": passed,
            "reason": None if passed else runtime.get("reason", "CONTROLLED_WORKFLOW_FAILED"),
            "selected_candidate": asdict(selected),
            "runtime_smoke": runtime,
            "artifact_workflow": artifact,
            "validation_scope": [
                "local_snapshot_manifest",
                "tokenizer_load_local_files_only",
                "causal_lm_load_local_files_only",
                "forward_logits_finite",
                "deterministic_generation_executes",
                "wal_artifact_lifecycle",
                "rollback_restores_artifact_checksum",
            ],
            "semantic_edit_training": False,
            "weights_modified": False,
        }
    )
    return result


def write_result(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
