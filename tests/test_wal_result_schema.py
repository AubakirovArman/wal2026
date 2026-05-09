import json

from wal.results import normalize_result_payload, validate_results
from wal.status import derive_status, status_to_pass


def test_status_derivation_handles_resource_and_unsupported_errors():
    assert derive_status({"error": "CUDA out of memory"}) == "BLOCKED"
    assert derive_status({"error": "Unrecognized configuration class X"}) == "UNSUPPORTED"
    assert status_to_pass("BLOCKED") is False
    assert status_to_pass("PASS") is True


def test_legacy_list_result_is_wrapped_without_losing_records():
    payload = [{"pass": True, "value": 1}, {"pass": True, "value": 2}]

    normalized, warnings = normalize_result_payload(payload, source="legacy_results.json")

    assert normalized["schema_version"] == "wal.results.v1"
    assert normalized["status"] == "PASS"
    assert normalized["record_count"] == 2
    assert normalized["records"] == payload
    assert warnings == ["legacy_list_wrapped"]


def test_validate_results_reports_invalid_status(tmp_path):
    (tmp_path / "good_results.json").write_text(json.dumps({"status": "PASS", "pass": True}))
    (tmp_path / "bad_results.json").write_text(json.dumps({"status": "MAYBE"}))

    summary = validate_results(tmp_path)

    assert summary.valid == 1
    assert summary.invalid == 1
    assert summary.invalid_files[0]["path"].endswith("bad_results.json")
