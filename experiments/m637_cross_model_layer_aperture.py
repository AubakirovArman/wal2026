from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from wal.cross_model_protocol import repo_root, write_result


ROOT = repo_root()
RESULT_PATH = ROOT / "experiments" / "m637_cross_model_layer_aperture_results.json"
INPUT_RESULTS = [
    ROOT / "experiments" / "m632_llama_1b_full_workflow_results.json",
    ROOT / "experiments" / "m633_qwen_small_full_workflow_results.json",
    ROOT / "experiments" / "m634_gemma_small_full_workflow_results.json",
    ROOT / "experiments" / "m635_tinyllama_mistral_full_workflow_results.json",
]


def load_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "NO_DATA", "path": str(path)}


def main() -> int:
    inputs = [load_result(path) for path in INPUT_RESULTS]
    candidate_models = sum(int(item.get("candidate_count", 0)) for item in inputs)
    real_passes = sum(1 for item in inputs if item.get("status") == "PASS")
    unique_model_paths = sorted(
        {
            str(item.get("selected_candidate", {}).get("path"))
            for item in inputs
            if item.get("status") == "PASS"
            and isinstance(item.get("selected_candidate"), dict)
            and item.get("selected_candidate", {}).get("path")
        }
    )
    status = "PASS" if real_passes >= 3 and len(unique_model_paths) >= 3 else "BLOCKED"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M637",
        "name": "Cross-Model Layer Aperture",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": None if status == "PASS" else "NEEDS_REAL_MODEL_MANIFESTS",
        "candidate_models": candidate_models,
        "real_passes": real_passes,
        "unique_model_paths": unique_model_paths,
        "unique_model_count": len(unique_model_paths),
        "policy": {
            "do_not_assume_fixed_layer": True,
            "required_mapping": "family-specific layer/target mapping before edit execution",
            "minimum_families": 3,
        },
        "inputs": inputs,
        "docs": "docs/cross_model_validation_plan.md",
    }
    write_result(RESULT_PATH, result)
    print(f"M637 Cross-Model Layer Aperture: {status}")
    print(f"candidate_models={candidate_models} real_passes={real_passes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
