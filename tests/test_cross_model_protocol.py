import json

from wal.cross_model_protocol import discover_candidates


def test_hf_snapshot_hash_does_not_trigger_large_model_exclusion(tmp_path):
    snapshot = (
        tmp_path
        / ".hf_cache"
        / "models--Qwen--Qwen2.5-0.5B-Instruct"
        / "snapshots"
        / "7b_hash_like_commit"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    )

    candidates, near_misses = discover_candidates(
        include_terms=["qwen2.5-0.5b"],
        exclude_terms=["7b"],
        max_size_gb=6.0,
        roots=[tmp_path],
    )

    assert len(candidates) == 1
    assert near_misses == []
