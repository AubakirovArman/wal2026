from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import AIGISystem, HuggingFaceTextBackend  # noqa: E402


MODEL_NAME = os.environ.get("AIGI_REAL_HF_MODEL", "/mnt/hf_model_weights/arman/3bit/bk/.hf_cache/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445")
CACHE_DIR = Path(os.environ.get("AIGI_HF_CACHE", str(ROOT / ".hf_cache")))
RESULT_PATH = ROOT / "experiments" / "m693_aigi_real_hf_backend_gate_results.json"
BOOK_PATH = ROOT / "book" / "M693_AIGI_Real_HF_Backend_Gate.md"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    import torch

    records: list[dict[str, object]] = []
    errors: list[str] = []
    started = time.monotonic()
    load_started = time.monotonic()
    backend = None
    try:
        backend = HuggingFaceTextBackend.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            max_new_tokens=24,
        )
    except Exception as exc:
        errors.append(f"model_load_failed:{type(exc).__name__}:{str(exc)[:240]}")
    load_elapsed = round(time.monotonic() - load_started, 3)

    records.append({"name": "real_backend_loaded", "passed": backend is not None})
    records.append({"name": "cuda_available_recorded", "passed": isinstance(torch.cuda.is_available(), bool)})

    before_answer = ""
    after_answer = ""
    rollback_answer = ""
    if backend is not None:
        question = "Say exactly: base model online"
        target = "M693 overlay committed"
        with tempfile.TemporaryDirectory(prefix="aigi_m693_") as tmpdir:
            system = AIGISystem.from_model(MODEL_NAME, workdir=tmpdir, model_backend=backend)
            before_started = time.monotonic()
            before = system.ask(question)
            before_elapsed = round(time.monotonic() - before_started, 3)
            before_answer = before.answer
            records.append({"name": "hf_fallback_source", "passed": before.source == "hf_model", "source": before.source})
            records.append({"name": "hf_generated_text", "passed": bool(before.answer.strip()), "elapsed_sec": before_elapsed})

            candidate = system.propose_memory(
                question=question,
                answer=target,
                source="m693_real_hf_backend_gate",
            )
            report = system.compile(candidate)
            committed = system.commit(report)
            after = system.ask(question)
            after_answer = after.answer
            records.append({"name": "memory_compile_passed", "passed": report.pass_, "tier": report.tier})
            records.append({"name": "memory_commit_passed", "passed": committed})
            records.append({"name": "overlay_answer_exact", "passed": after.answer == target, "source": after.source})

            rolled_back = system.rollback_last()
            rollback = system.ask(question)
            rollback_answer = rollback.answer
            records.append({"name": "rollback_passed", "passed": rolled_back})
            records.append({"name": "rollback_returns_to_hf", "passed": rollback.source == "hf_model", "source": rollback.source})

    failures = [record for record in records if not record.get("passed")]
    status = "PASS" if not failures and not errors else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M693",
        "name": "AIGI Real HF Backend Gate",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "cache_dir": str(CACHE_DIR),
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "load_elapsed_sec": load_elapsed,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "checks_total": len(records),
        "checks_passed": len(records) - len(failures),
        "records": records,
        "failures": failures,
        "errors": errors,
        "before_answer_preview": before_answer[:240],
        "after_answer": after_answer,
        "rollback_answer_preview": rollback_answer[:240],
        "claim_boundary": "Real HF inference backend gate; not LoRA training and not real semantic weight editing yet.",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    BOOK_PATH.write_text(
        "# M693 — AIGI Real HF Backend Gate\n\n"
        "Date: 2026-05-10\n"
        f"Status: {status}\n"
        f"Result: `{RESULT_PATH.relative_to(ROOT)}`\n"
        f"Model: `{MODEL_NAME}`\n\n"
        "## Purpose\n\n"
        "Connect the AIGI SDK to a real HuggingFace causal language model backend instead of only the symbolic fallback.\n\n"
        "## Outcome\n\n"
        f"- Backend load: `{records[0]['passed']}`\n"
        f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
        "- The base answer comes from `hf_model`; committed AIGI memory overrides it; rollback returns to the HF backend.\n"
        "- This is real inference integration, not real LoRA/weight editing yet.\n",
        encoding="utf-8",
    )
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m693_aigi_real_hf_backend_gate",
        "status": status,
        "details": {"model": MODEL_NAME, "checks_total": len(records), "checks_passed": len(records) - len(failures)},
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M693 — Real HF Backend Gate\n\n"
            f"- Status: `{status}`\n"
            f"- Model: `{MODEL_NAME}`\n"
            f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
            "- Boundary: real HF inference backend, no LoRA/weight-edit backend yet.\n"
        )
    print(f"M693 AIGI Real HF Backend Gate: {status}")
    print(f"model={MODEL_NAME} checks={len(records) - len(failures)}/{len(records)} load_sec={load_elapsed}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
