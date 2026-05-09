from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from wal.cross_model_protocol import WORKFLOW_STEPS, repo_root, write_result


ROOT = repo_root()
RESULT_PATH = ROOT / "experiments" / "m636_cross_model_recipe_replay_results.json"
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
    real_passes = [item for item in inputs if item.get("status") == "PASS"]
    blocked = [item for item in inputs if item.get("status") == "BLOCKED"]
    status = "PASS" if len(real_passes) >= 3 else "BLOCKED"
    reason = None if status == "PASS" else "NEEDS_THREE_REAL_SMALL_MODEL_WORKFLOWS"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M636",
        "name": "Cross-Model Recipe Replay",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "required_real_passes": 3,
        "real_passes": len(real_passes),
        "blocked_inputs": len(blocked),
        "workflow_steps": WORKFLOW_STEPS,
        "recipe_contract": {
            "facts": 5,
            "checks": ["exact", "negative", "context"],
            "rollback_required": True,
            "release_notes_required": True,
        },
        "inputs": inputs,
        "docs": "docs/cross_model_validation_plan.md",
    }
    write_result(RESULT_PATH, result)
    print(f"M636 Cross-Model Recipe Replay: {status}")
    print(f"real_passes={len(real_passes)} blocked_inputs={len(blocked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
