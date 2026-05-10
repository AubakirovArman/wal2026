from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from wal.cross_model_protocol import repo_root, write_result


ROOT = repo_root()
RESULT_PATH = ROOT / "experiments" / "m638_cross_model_ci_behavior_results.json"
INPUT_RESULTS = [
    ROOT / "experiments" / "m632_llama_1b_full_workflow_results.json",
    ROOT / "experiments" / "m633_qwen_small_full_workflow_results.json",
    ROOT / "experiments" / "m634_gemma_small_full_workflow_results.json",
    ROOT / "experiments" / "m635_tinyllama_mistral_full_workflow_results.json",
    ROOT / "experiments" / "m636_cross_model_recipe_replay_results.json",
]


def load_result(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"status": "NO_DATA", "path": str(path)}


def main() -> int:
    inputs = [load_result(path) for path in INPUT_RESULTS]
    real_model_passes = sum(1 for item in inputs[:4] if item.get("status") == "PASS")
    unique_model_paths = sorted(
        {
            str(item.get("selected_candidate", {}).get("path"))
            for item in inputs[:4]
            if item.get("status") == "PASS"
            and isinstance(item.get("selected_candidate"), dict)
            and item.get("selected_candidate", {}).get("path")
        }
    )
    replay_pass = inputs[-1].get("status") == "PASS"
    status = "PASS" if real_model_passes >= 3 and len(unique_model_paths) >= 3 and replay_pass else "BLOCKED"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M638",
        "name": "Cross-Model CI Behavior",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": None if status == "PASS" else "NEEDS_REAL_CROSS_MODEL_WORKFLOW_RESULTS",
        "required_checks": [
            "exact_behavior",
            "negative_behavior",
            "context_behavior",
            "behavioral_checksum",
            "rollback_restores_previous_checksum",
        ],
        "real_model_passes": real_model_passes,
        "unique_model_paths": unique_model_paths,
        "unique_model_count": len(unique_model_paths),
        "replay_pass": replay_pass,
        "inputs": inputs,
        "docs": "docs/cross_model_validation_plan.md",
    }
    write_result(RESULT_PATH, result)
    print(f"M638 Cross-Model CI Behavior: {status}")
    print(f"real_model_passes={real_model_passes} replay_pass={replay_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
