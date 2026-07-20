"""Audit exact-window and declared-range overlap for recovery suites."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from compensation_window import PROJECT, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, action="append", default=[])
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def digest(window: torch.Tensor) -> str:
    return hashlib.sha256(window.contiguous().numpy().tobytes()).hexdigest()


def window_digests(payload: dict) -> set[str]:
    result = {digest(window) for window in payload.get("calibration", [])}
    for windows in payload.get("gates", {}).values():
        result.update(digest(window) for window in windows)
    return result


def stream_identity(payload: dict, domain: str) -> str | None:
    """Identify the untiled token stream, including legacy audit payloads."""
    sources = [str(path) for path in payload.get("sources", [])]
    source_sha = payload.get("source_sha256", {})
    if domain.startswith("c4"):
        family = "c4"
        selected = [
            path
            for path in sources
            if Path(path).name.startswith("c4-") and path.endswith(".arrow")
        ]
    elif domain.startswith("squad"):
        family = "squad"
        selected = [
            path
            for path in sources
            if Path(path).name.startswith("squad-") and path.endswith(".arrow")
        ]
    else:
        family = "code"
        selected = [path for path in sources if path.endswith(".py")]
    if selected and all(path in source_sha for path in selected):
        stream_view = None
        if family == "c4":
            stream_view = {
                "text_start": int(payload.get("c4_text_start", 0)),
                "text_count": int(payload.get("c4_text_count", 4000)),
            }
        material = {
            "model_revision": payload.get("model_revision"),
            "family": family,
            "sources": [(path, source_sha[path]) for path in selected],
            "stream_view": stream_view,
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        return f"sources:{hashlib.sha256(encoded).hexdigest()}"

    full_sha = payload.get("full_token_stream_sha256", {})
    if domain in full_sha:
        return full_sha[domain]
    return None


def source_ranges(payload: dict) -> dict[str, list[dict]]:
    """Return ranges keyed by the tokenized source SHA rather than domain name."""
    declared = payload.get("ranges")
    result: dict[str, list[dict]] = {}
    if declared is not None:
        for split, domains in declared.items():
            for domain, bounds in domains.items():
                source_id = stream_identity(payload, domain)
                if source_id is None:
                    continue
                intervals = (
                    bounds
                    if bounds and isinstance(bounds[0], (list, tuple))
                    else [bounds]
                )
                for segment, interval in enumerate(intervals):
                    result.setdefault(source_id, []).append(
                        {
                            "split": split,
                            "domain": domain,
                            "segment": segment,
                            "start": int(interval[0]),
                            "end": int(interval[1]),
                        }
                    )
        return result

    if payload.get("format") == "wal-tat-audit-holdout-v1" and payload.get(
        "offsets"
    ):
        length = int(payload["sequence_length"])
        offsets = payload["offsets"]
        for domain, windows in payload.get("gates", {}).items():
            offset_key = domain if domain in offsets else "code"
            if offset_key not in offsets:
                continue
            source_id = stream_identity(payload, domain)
            if source_id is None:
                continue
            start = int(offsets[offset_key])
            result.setdefault(source_id, []).append(
                {
                    "split": "gates",
                    "domain": domain,
                    "start": start,
                    "end": start + len(windows) * length,
                    "inferred_legacy_audit": True,
                }
            )
        return result

    if payload.get("format") != "wal-tat-diverse-recovery-v1":
        return result
    length = int(payload["sequence_length"])
    base_count = int(payload["calibration_per_domain"])
    repeats = {
        "c4_train": int(payload.get("c4_calibration_repeat") or 1),
        "squad_train": int(payload.get("squad_calibration_repeat") or 1),
        "vendor_code_train": int(payload.get("code_calibration_repeat") or 1),
    }
    domains = payload.get("full_token_stream_sha256") or payload.get(
        "token_sha256", {}
    )
    for domain in domains:
        source_id = stream_identity(payload, domain)
        if source_id is None:
            continue
        repeat = repeats.get(domain, 1)
        result.setdefault(source_id, []).append(
            {
                "split": "calibration",
                "domain": domain,
                "start": 10007,
                "end": 10007 + base_count * repeat * length,
                "inferred_legacy_v1": True,
            }
        )
        if domain in payload.get("gates", {}):
            result[source_id].append(
                {
                    "split": "gates",
                    "domain": domain,
                    "start": 80003,
                    "end": 80003 + len(payload["gates"][domain]) * length,
                    "inferred_legacy_v1": True,
                }
            )
    return result


def intervals_overlap(left: dict, right: dict) -> bool:
    return max(left["start"], right["start"]) < min(left["end"], right["end"])


def main() -> None:
    args = parse_args()
    new_path = args.new.resolve()
    prior_paths = [path.resolve() for path in args.prior]
    if new_path in prior_paths or len(set(prior_paths)) != len(prior_paths):
        raise ValueError("suite paths must be unique")
    new_payload = torch.load(new_path, map_location="cpu", weights_only=False)
    new_digests = window_digests(new_payload)
    new_ranges = source_ranges(new_payload)
    comparisons = []
    passed = True
    for prior_path in prior_paths:
        prior = torch.load(prior_path, map_location="cpu", weights_only=False)
        exact = len(new_digests & window_digests(prior))
        range_overlaps = []
        prior_ranges = source_ranges(prior)
        for source_sha in sorted(set(new_ranges) & set(prior_ranges)):
            for left in new_ranges[source_sha]:
                for right in prior_ranges[source_sha]:
                    if intervals_overlap(left, right):
                        range_overlaps.append(
                            {
                                "token_sha256": source_sha,
                                "new": left,
                                "prior": right,
                            }
                        )
        comparison_passed = exact == 0 and not range_overlaps
        passed = passed and comparison_passed
        comparisons.append(
            {
                "prior": str(prior_path),
                "prior_sha256": sha256_file(prior_path),
                "exact_window_overlap": exact,
                "declared_range_overlaps": range_overlaps,
                "passed": comparison_passed,
            }
        )
    result = {
        "schema": "wal-tat-recovery-suite-overlap-audit-v1",
        "new_suite": {
            "path": str(new_path),
            "sha256": sha256_file(new_path),
            "policy": new_payload.get("policy"),
            "ranges": new_payload.get("ranges"),
            "token_sha256": new_payload.get("token_sha256"),
        },
        "comparisons": comparisons,
        "passed": passed,
    }
    output = PROJECT / f"results/{args.tag}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
