from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aigi import HuggingFaceTextBackend, ModuleLoRAAdapterTrainer  # noqa: E402


MODEL_NAME = os.environ.get("AIGI_REAL_HF_MODEL", "/mnt/hf_model_weights/arman/3bit/wal/.hf_cache/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
CACHE_DIR = Path(os.environ.get("AIGI_HF_CACHE", str(ROOT / ".hf_cache")))
RESULT_PATH = ROOT / "experiments" / "m696_aigi_real_module_lora_adapter_results.json"
BOOK_PATH = ROOT / "book" / "M696_AIGI_Real_Module_LoRA_Adapter.md"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"
ADAPTER_PATH = ROOT / ".aigi" / "adapters" / "m696_qwen_mlp_down_proj_lora.pt"


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
    target_module = ""

    try:
        backend = HuggingFaceTextBackend.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            max_new_tokens=24,
        )
        layer_count = len(backend.model.model.layers)
        target_module = f"model.layers.{layer_count - 1}.mlp.down_proj"
        trainer = ModuleLoRAAdapterTrainer(backend)
        training_report = trainer.train(
            question="What is the M696 module LoRA codeword? Answer only the codeword.",
            target_answer=" M696_MODULE_LORA_OK",
            target_module=target_module,
            artifact_path=ADAPTER_PATH,
            rank=8,
            alpha=16.0,
            steps=80,
            learning_rate=0.08,
            max_new_tokens=12,
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:400]}")

    if training_report is not None:
        records.extend([
            {"name": "model_loaded", "passed": True, "model": MODEL_NAME},
            {"name": "target_module_selected", "passed": training_report.target_module.endswith("mlp.down_proj"), "target_module": training_report.target_module},
            {"name": "adapter_artifact_written", "passed": ADAPTER_PATH.exists(), "bytes": training_report.artifact_size_bytes},
            {"name": "loss_decreased", "passed": training_report.after_loss < training_report.before_loss, "before": training_report.before_loss, "after": training_report.after_loss},
            {"name": "loss_improvement_ratio_ge_95pct", "passed": training_report.loss_improvement_ratio >= 0.95, "ratio": training_report.loss_improvement_ratio},
            {"name": "target_generated_with_adapter", "passed": training_report.target_in_generated_text, "generated": training_report.generated_text},
            {"name": "base_generation_not_target", "passed": "M696_MODULE_LORA_OK" not in training_report.base_generated_text, "base_generated": training_report.base_generated_text},
            {"name": "real_module_lora_parameters", "passed": training_report.trainable_parameters > 0 and training_report.rank == 8, "trainable_parameters": training_report.trainable_parameters},
            {"name": "cuda_available_recorded", "passed": isinstance(torch.cuda.is_available(), bool)},
        ])
    else:
        records.append({"name": "model_loaded_and_trained", "passed": False, "target_module": target_module})

    failures = [record for record in records if not record.get("passed")]
    status = "PASS" if not failures and not errors else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M696",
        "name": "AIGI Real Module LoRA Adapter",
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
            "target_module": training_report.target_module,
            "prompt_tokens": training_report.prompt_tokens,
            "target_tokens": training_report.target_tokens,
            "rank": training_report.rank,
            "alpha": training_report.alpha,
            "steps": training_report.steps,
            "learning_rate": training_report.learning_rate,
            "trainable_parameters": training_report.trainable_parameters,
            "before_loss": training_report.before_loss,
            "after_loss": training_report.after_loss,
            "loss_improvement": training_report.loss_improvement,
            "loss_improvement_ratio": training_report.loss_improvement_ratio,
            "generated_text": training_report.generated_text,
            "base_generated_text": training_report.base_generated_text,
            "artifact_path": str(ADAPTER_PATH.relative_to(ROOT)),
            "artifact_size_bytes": training_report.artifact_size_bytes,
        },
        "claim_boundary": "Real LoRA injection into a frozen MLP down_proj module; not full multi-fact training, MEMIT, or production editing yet.",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    BOOK_PATH.write_text(
        "# M696 — AIGI Real Module LoRA Adapter\n\n"
        "Date: 2026-05-10\n"
        f"Status: {status}\n"
        f"Result: `{RESULT_PATH.relative_to(ROOT)}`\n"
        f"Model: `{MODEL_NAME}`\n"
        f"Target module: `{target_module}`\n\n"
        "## Purpose\n\n"
        "Inject a real trainable LoRA adapter into an actual MLP `down_proj` module of a frozen small HuggingFace model.\n\n"
        "## Outcome\n\n"
        f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
        f"- Before loss: `{None if training_report is None else round(training_report.before_loss, 6)}`\n"
        f"- After loss: `{None if training_report is None else round(training_report.after_loss, 6)}`\n"
        f"- Trainable parameters: `{None if training_report is None else training_report.trainable_parameters}`\n"
        f"- Adapter artifact: `{ADAPTER_PATH.relative_to(ROOT)}`\n"
        "- Boundary: real module LoRA injection, still a one-fact controlled gate rather than production model editing.\n",
        encoding="utf-8",
    )
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m696_aigi_real_module_lora_adapter",
        "status": status,
        "details": {
            "model": MODEL_NAME,
            "target_module": target_module,
            "checks_total": len(records),
            "checks_passed": len(records) - len(failures),
            "after_loss": None if training_report is None else training_report.after_loss,
        },
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M696 — Real Module LoRA Adapter\n\n"
            f"- Status: `{status}`\n"
            f"- Model: `{MODEL_NAME}`\n"
            f"- Target module: `{target_module}`\n"
            f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
            f"- Before loss: `{None if training_report is None else round(training_report.before_loss, 6)}`\n"
            f"- After loss: `{None if training_report is None else round(training_report.after_loss, 6)}`\n"
            "- Boundary: real MLP module LoRA injection, one-fact controlled gate.\n"
        )
    print(f"M696 AIGI Real Module LoRA Adapter: {status}")
    if training_report is not None:
        print(
            f"model={MODEL_NAME} target={training_report.target_module} "
            f"checks={len(records) - len(failures)}/{len(records)} "
            f"loss={training_report.before_loss:.4f}->{training_report.after_loss:.4f} "
            f"generated={training_report.target_in_generated_text}"
        )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
