import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from recovery_suite_overlap_audit import intervals_overlap, source_ranges  # noqa: E402


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
