from __future__ import annotations

from pathlib import Path

from wal.cross_model_protocol import controlled_model_workflow_result, discover_candidates, repo_root, write_result


ROOT = repo_root()
RESULT_PATH = ROOT / "experiments" / "m632_llama_1b_full_workflow_results.json"


def main() -> int:
    candidates, near_misses = discover_candidates(
        include_terms=["llama-3.2-1b", "llama3.2-1b", "llama-1b", "smollm2-360m", "smollm2"],
        exclude_terms=["70b", "31b", "13b", "8b", "7b", "vision", "vl"],
        max_size_gb=6.0,
    )
    result = controlled_model_workflow_result(
        module="M632",
        name="Llama 1B Full Workflow",
        family="llama_small",
        candidates=candidates,
        near_misses=near_misses,
    )
    result["docs"] = "docs/model_small_protocol.md"
    write_result(RESULT_PATH, result)
    print(f"M632 Llama 1B Full Workflow: {result['status']}")
    print(f"candidates={result['candidate_count']} near_misses={result['near_miss_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
