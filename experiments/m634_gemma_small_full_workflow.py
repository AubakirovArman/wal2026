from __future__ import annotations

from wal.cross_model_protocol import discover_candidates, model_workflow_result, repo_root, write_result


ROOT = repo_root()
RESULT_PATH = ROOT / "experiments" / "m634_gemma_small_full_workflow_results.json"


def main() -> int:
    candidates, near_misses = discover_candidates(
        include_terms=["gemma-2b", "gemma-3-1b", "gemma-1b", "gemma-small"],
        exclude_terms=["31b", "27b", "12b", "9b", "7b", "vision", "vl", "flux"],
        max_size_gb=6.0,
    )
    result = model_workflow_result(
        module="M634",
        name="Gemma Small Full Workflow",
        family="gemma_small",
        candidates=candidates,
        near_misses=near_misses,
    )
    result["docs"] = "docs/model_small_protocol.md"
    write_result(RESULT_PATH, result)
    print(f"M634 Gemma Small Full Workflow: {result['status']}")
    print(f"candidates={result['candidate_count']} near_misses={result['near_miss_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
