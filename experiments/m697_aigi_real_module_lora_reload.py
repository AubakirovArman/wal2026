from __future__ import annotations

import gc
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
RESULT_PATH = ROOT / "experiments" / "m697_aigi_real_module_lora_reload_results.json"
BOOK_PATH = ROOT / "book" / "M697_AIGI_Real_Module_LoRA_Reload.md"
STEP_LOG = ROOT / "logs" / "aigi" / "aigi_steps.jsonl"
TEST_LOG = ROOT / "docs" / "aigi" / "test_log.md"
ADAPTER_PATH = ROOT / ".aigi" / "adapters" / "m697_qwen_mlp_down_proj_lora_reload.pt"


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def release_backend(*objects: object) -> None:
    import torch

    for item in objects:
        del item
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    import torch

    started = time.monotonic()
    errors: list[str] = []
    records: list[dict[str, object]] = []
    training_report = None
    apply_report = None
    target_module = ""

    try:
        train_backend = HuggingFaceTextBackend.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            max_new_tokens=24,
        )
        layer_count = len(train_backend.model.model.layers)
        target_module = f"model.layers.{layer_count - 1}.mlp.down_proj"
        train_trainer = ModuleLoRAAdapterTrainer(train_backend)
        training_report = train_trainer.train(
            question="What is the M697 reload codeword? Answer only the codeword.",
            target_answer=" M697_RELOAD_OK",
            target_module=target_module,
            artifact_path=ADAPTER_PATH,
            rank=8,
            alpha=16.0,
            steps=80,
            learning_rate=0.08,
            max_new_tokens=12,
        )
        release_backend(train_trainer, train_backend)

        reload_backend = HuggingFaceTextBackend.from_pretrained(
            MODEL_NAME,
            cache_dir=CACHE_DIR,
            max_new_tokens=24,
        )
        reload_trainer = ModuleLoRAAdapterTrainer(reload_backend)
        apply_report = reload_trainer.apply_artifact(
            artifact_path=ADAPTER_PATH,
            max_new_tokens=12,
        )
        release_backend(reload_trainer, reload_backend)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}:{str(exc)[:400]}")

    if training_report is not None:
        records.extend([
            {"name": "training_model_loaded", "passed": True, "model": MODEL_NAME},
            {"name": "training_loss_decreased", "passed": training_report.after_loss < training_report.before_loss, "before": training_report.before_loss, "after": training_report.after_loss},
            {"name": "training_target_generated", "passed": training_report.target_in_generated_text, "generated": training_report.generated_text},
            {"name": "adapter_artifact_written", "passed": ADAPTER_PATH.exists(), "bytes": training_report.artifact_size_bytes},
            {"name": "target_module_recorded", "passed": training_report.target_module.endswith("mlp.down_proj"), "target_module": training_report.target_module},
        ])
    else:
        records.append({"name": "training_completed", "passed": False, "target_module": target_module})

    if apply_report is not None:
        records.extend([
            {"name": "fresh_model_loaded", "passed": True, "model": MODEL_NAME},
            {"name": "artifact_model_matches", "passed": apply_report.artifact_model == MODEL_NAME, "artifact_model": apply_report.artifact_model},
            {"name": "reload_target_module_matches", "passed": apply_report.target_module == target_module, "target_module": apply_report.target_module},
            {"name": "reloaded_generation_target", "passed": apply_report.target_in_generated_text, "generated": apply_report.generated_text},
            {"name": "fresh_base_generation_not_target", "passed": "M697_RELOAD_OK" not in apply_report.base_generated_text, "base_generated": apply_report.base_generated_text},
            {"name": "reloaded_trainable_params_match", "passed": training_report is not None and apply_report.trainable_parameters == training_report.trainable_parameters, "trainable_parameters": apply_report.trainable_parameters},
            {"name": "reload_artifact_size_matches", "passed": training_report is not None and apply_report.artifact_size_bytes == training_report.artifact_size_bytes, "bytes": apply_report.artifact_size_bytes},
            {"name": "cuda_available_recorded", "passed": isinstance(torch.cuda.is_available(), bool)},
        ])
    else:
        records.append({"name": "artifact_applied_to_fresh_model", "passed": False, "target_module": target_module})

    failures = [record for record in records if not record.get("passed")]
    status = "PASS" if not failures and not errors else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M697",
        "name": "AIGI Real Module LoRA Reload",
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
            "rank": training_report.rank,
            "alpha": training_report.alpha,
            "steps": training_report.steps,
            "learning_rate": training_report.learning_rate,
            "trainable_parameters": training_report.trainable_parameters,
            "before_loss": training_report.before_loss,
            "after_loss": training_report.after_loss,
            "loss_improvement_ratio": training_report.loss_improvement_ratio,
            "generated_text": training_report.generated_text,
            "base_generated_text": training_report.base_generated_text,
        },
        "reload": None if apply_report is None else {
            "target_module": apply_report.target_module,
            "rank": apply_report.rank,
            "alpha": apply_report.alpha,
            "trainable_parameters": apply_report.trainable_parameters,
            "generated_text": apply_report.generated_text,
            "base_generated_text": apply_report.base_generated_text,
            "artifact_path": str(ADAPTER_PATH.relative_to(ROOT)),
            "artifact_size_bytes": apply_report.artifact_size_bytes,
            "artifact_model": apply_report.artifact_model,
        },
        "claim_boundary": "Real module-LoRA artifact reload into a fresh model instance; still one-fact controlled persistence, not production multi-fact editing.",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    BOOK_PATH.write_text(
        "# M697 — AIGI Real Module LoRA Reload\n\n"
        "Date: 2026-05-10\n"
        f"Status: {status}\n"
        f"Result: `{RESULT_PATH.relative_to(ROOT)}`\n"
        f"Model: `{MODEL_NAME}`\n"
        f"Target module: `{target_module}`\n\n"
        "## Purpose\n\n"
        "Verify that a trained module-LoRA artifact can be saved, loaded into a fresh model instance, and reproduce the target behavior.\n\n"
        "## Outcome\n\n"
        f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
        f"- Training loss: `{None if training_report is None else round(training_report.before_loss, 6)} → {None if training_report is None else round(training_report.after_loss, 6)}`\n"
        f"- Reload generated target: `{None if apply_report is None else apply_report.target_in_generated_text}`\n"
        f"- Trainable parameters: `{None if apply_report is None else apply_report.trainable_parameters}`\n"
        f"- Adapter artifact: `{ADAPTER_PATH.relative_to(ROOT)}`\n"
        "- Boundary: real artifact persistence/reload gate, still one-fact controlled validation.\n",
        encoding="utf-8",
    )
    append_jsonl(STEP_LOG, {
        "timestamp": result["timestamp"],
        "event": "m697_aigi_real_module_lora_reload",
        "status": status,
        "details": {
            "model": MODEL_NAME,
            "target_module": target_module,
            "checks_total": len(records),
            "checks_passed": len(records) - len(failures),
            "reloaded_generation_target": None if apply_report is None else apply_report.target_in_generated_text,
        },
    })
    with TEST_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## M697 — Real Module LoRA Reload\n\n"
            f"- Status: `{status}`\n"
            f"- Model: `{MODEL_NAME}`\n"
            f"- Target module: `{target_module}`\n"
            f"- Checks: `{len(records) - len(failures)}/{len(records)}`\n"
            f"- Reload generated target: `{None if apply_report is None else apply_report.target_in_generated_text}`\n"
            "- Boundary: real fresh-model artifact reload, one-fact controlled gate.\n"
        )
    print(f"M697 AIGI Real Module LoRA Reload: {status}")
    if training_report is not None and apply_report is not None:
        print(
            f"model={MODEL_NAME} target={apply_report.target_module} "
            f"checks={len(records) - len(failures)}/{len(records)} "
            f"loss={training_report.before_loss:.4f}->{training_report.after_loss:.4f} "
            f"reload_generated={apply_report.target_in_generated_text}"
        )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
