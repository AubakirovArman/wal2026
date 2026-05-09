from __future__ import annotations

from wal.cross_model_protocol import discover_candidates, model_workflow_result, repo_root, write_result


ROOT = repo_root()
RESULT_PATH = ROOT / "experiments" / "m633_qwen_small_full_workflow_results.json"


def main() -> int:
    candidates, near_misses = discover_candidates(
        include_terms=["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen-0.5b", "qwen-1.5b"],
        exclude_terms=["72b", "32b", "31b", "14b", "8b", "7b", "vision", "vl"],
        max_size_gb=6.0,
    )
    result = model_workflow_result(
        module="M633",
        name="Qwen Small Full Workflow",
        family="qwen_small",
        candidates=candidates,
        near_misses=near_misses,
    )
    result["docs"] = "docs/model_small_protocol.md"
    write_result(RESULT_PATH, result)
    print(f"M633 Qwen Small Full Workflow: {result['status']}")
    print(f"candidates={result['candidate_count']} near_misses={result['near_miss_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
