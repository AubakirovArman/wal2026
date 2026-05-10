from __future__ import annotations

from wal.cross_model_protocol import controlled_model_workflow_result, discover_candidates, repo_root, write_result


ROOT = repo_root()
RESULT_PATH = ROOT / "experiments" / "m635_tinyllama_mistral_full_workflow_results.json"


def main() -> int:
    candidates, near_misses = discover_candidates(
        include_terms=["tinyllama", "mistral-1b", "mistral-small"],
        exclude_terms=["mixtral", "22b", "12b", "8b", "7b", "vision", "vl"],
        max_size_gb=6.0,
    )
    result = controlled_model_workflow_result(
        module="M635",
        name="TinyLlama or Mistral Small Full Workflow",
        family="tinyllama_or_mistral_small",
        candidates=candidates,
        near_misses=near_misses,
    )
    result["docs"] = "docs/model_small_protocol.md"
    write_result(RESULT_PATH, result)
    print(f"M635 TinyLlama/Mistral Small Full Workflow: {result['status']}")
    print(f"candidates={result['candidate_count']} near_misses={result['near_miss_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
