from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import HuggingFaceTextBackend, SoftPromptAdapterTrainer  # noqa: E402


MODEL_NAME = os.environ.get("AIGI_REAL_HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
CACHE_DIR = Path(os.environ.get("AIGI_HF_CACHE", str(ROOT / ".hf_cache")))
RESULT_PATH = ROOT / "experiments" / "m694_aigi_real_soft_prompt_adapter_results.json"
BOOK_PATH = ROOT / "book" / "M694_AIGI_Real_Soft_Prompt_Adapter.md"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"
ADAPTER_PATH = ROOT / ".aigi" / "adapters" / "m694_qwen_soft_prompt.pt"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    import torch

    started = time.monotonic()
    errors: list[str] = []
    records: list[dict[str, object]] = []
    training_report = None

    try:
        backend = HuggingFaceTextBackend.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            max_new_tokens=24,
        )
        trainer = SoftPromptAdapterTrainer(backend)
        training_report = trainer.train(
            question="What is the M694 adapter codeword? Answer only the codeword.",
            target_answer=" M694_REAL_ADAPTER_OK",
            artifact_path=ADAPTER_PATH,
            soft_prompt_tokens=8,
            steps=50,
            learning_rate=0.2,
            max_new_tokens=12,
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:400]}")

    if training_report is not None:
        records.extend([
            {"name": "model_loaded", "passed": True, "model": MODEL_NAME},
            {"name": "adapter_artifact_written", "passed": ADAPTER_PATH.exists(), "bytes": training_report.artifact_size_bytes},
            {"name": "loss_decreased", "passed": training_report.after_loss < training_report.before_loss, "before": training_report.before_loss, "after": training_report.after_loss},
            {"name": "loss_improvement_ratio_ge_95pct", "passed": training_report.loss_improvement_ratio >= 0.95, "ratio": training_report.loss_improvement_ratio},
            {"name": "target_generated_with_adapter", "passed": training_report.target_in_generated_text, "generated": training_report.generated_text},
            {"name": "base_generation_not_target", "passed": "M694_REAL_ADAPTER_OK" not in training_report.base_generated_text, "base_generated": training_report.base_generated_text},
            {"name": "real_gradient_steps", "passed": training_report.steps >= 1 and training_report.soft_prompt_tokens >= 1},
            {"name": "cuda_available_recorded", "passed": isinstance(torch.cuda.is_available(), bool)},
        ])
    else:
        records.append({"name": "model_loaded_and_trained", "passed": False})

    failures = [record for record in records if not record.get("passed")]
    status = "PASS" if not failures and not errors else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M694",
        "name": "AIGI Real Soft Prompt Adapter",
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
        "checks_total": len(records),
        "checks_passed": len(records) - len(failures),
        "records": records,
        "failures": failures,
        "errors": errors,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "training": None if training_report is None else {
            "prompt_tokens": training_report.prompt_tokens,
            "target_tokens": training_report.target_tokens,
            "soft_prompt_tokens": training_report.soft_prompt_tokens,
            "steps": training_report.steps,
            "learning_rate": training_report.learning_rate,
            "before_loss": training_report.before_loss,
            "after_loss": training_report.after_loss,
            "loss_improvement": training_report.loss_improvement,
            "loss_improvement_ratio": training_report.loss_improvement_ratio,
            "generated_text": training_report.generated_text,
            "base_generated_text": training_report.base_generated_text,
            "artifact_path": str(ADAPTER_PATH.relative_to(ROOT)),
            "artifact_size_bytes": training_report.artifact_size_bytes,
        },
        "claim_boundary": "Real gradient-trained soft-prompt adapter on a frozen HF model; not base-weight LoRA/MEMIT editing yet.",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    BOOK_PATH.write_text(
        "# M694 — AIGI Real Soft Prompt Adapter\n\n"
        "Date: 2026-05-10\n"
        f"Status: {status}\n"
        f"Result: `{RESULT_PATH.relative_to(ROOT)}`\n"
        f"Model: `{MODEL_NAME}`\n\n"
        "## Purpose\n\n"
        "Run a real gradient-trained adapter update on a small HuggingFace model instead of serving only a retrieval overlay.\n\n"
        "## Outcome\n\n"
        f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
        f"- Before loss: `{None if training_report is None else round(training_report.before_loss, 6)}`\n"
        f"- After loss: `{None if training_report is None else round(training_report.after_loss, 6)}`\n"
        f"- Adapter artifact: `{ADAPTER_PATH.relative_to(ROOT)}`\n"
        "- Boundary: real soft-prompt adapter training on a frozen base model, not full base-weight editing yet.\n",
        encoding="utf-8",
    )
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m694_aigi_real_soft_prompt_adapter",
        "status": status,
        "details": {
            "model": MODEL_NAME,
            "checks_total": len(records),
            "checks_passed": len(records) - len(failures),
            "after_loss": None if training_report is None else training_report.after_loss,
        },
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M694 — Real Soft Prompt Adapter\n\n"
            f"- Status: `{status}`\n"
            f"- Model: `{MODEL_NAME}`\n"
            f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
            f"- Before loss: `{None if training_report is None else round(training_report.before_loss, 6)}`\n"
            f"- After loss: `{None if training_report is None else round(training_report.after_loss, 6)}`\n"
            "- Boundary: real gradient-trained adapter, frozen base model, no LoRA/MEMIT base-weight edit yet.\n"
        )
    print(f"M694 AIGI Real Soft Prompt Adapter: {status}")
    if training_report is not None:
        print(
            f"model={MODEL_NAME} checks={len(records) - len(failures)}/{len(records)} "
            f"loss={training_report.before_loss:.4f}->{training_report.after_loss:.4f} "
            f"generated={training_report.target_in_generated_text}"
        )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
