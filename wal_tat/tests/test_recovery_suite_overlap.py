import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from recovery_suite_overlap_audit import (  # noqa: E402
    intervals_overlap,
    source_ranges,
    stream_identity,
)


def test_source_ranges_prefers_full_stream_identity():
    payload = {
        "full_token_stream_sha256": {"squad": "full"},
        "token_sha256": {"squad": "selected-windows"},
        "ranges": {"calibration": {"squad": [100, 200]}},
    }
    result = source_ranges(payload)
    assert set(result) == {"full"}
    assert result["full"][0]["start"] == 100


def test_adjacent_ranges_do_not_overlap_but_intersecting_ranges_do():
    assert not intervals_overlap({"start": 0, "end": 10}, {"start": 10, "end": 20})
    assert intervals_overlap({"start": 0, "end": 11}, {"start": 10, "end": 20})


def test_legacy_audit_offsets_use_source_file_identity():
    payload = {
        "format": "wal-tat-audit-holdout-v1",
        "model_revision": "revision",
        "sequence_length": 4,
        "offsets": {"squad_context": 20, "code": 40},
        "gates": {"squad_context": [0, 1], "torch_code": [0]},
        "sources": ["/cache/squad-validation.arrow", "/lib/a.py"],
        "source_sha256": {
            "/cache/squad-validation.arrow": "squad-sha",
            "/lib/a.py": "code-sha",
        },
    }
    ranges = source_ranges(payload)
    squad_id = stream_identity(payload, "squad_context")
    code_id = stream_identity(payload, "torch_code")
    assert ranges[squad_id][0]["start"] == 20
    assert ranges[squad_id][0]["end"] == 28
    assert ranges[code_id][0]["start"] == 40
    assert ranges[code_id][0]["end"] == 44
