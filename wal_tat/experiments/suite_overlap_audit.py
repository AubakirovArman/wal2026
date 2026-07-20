"""Verify that newly built tokenized validation suites do not overlap priors."""
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
    parser.add_argument("--new", type=Path, action="append", required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def window_digest(window: torch.Tensor) -> str:
    return hashlib.sha256(window.contiguous().numpy().tobytes()).hexdigest()


def interval(payload: dict, domain: str) -> tuple[int, int] | None:
    offsets = payload.get("offsets")
    if offsets is None:
        return None
    key = domain if domain in offsets else "code"
    if key not in offsets:
        return None
    start = int(offsets[key])
    length = int(payload["sequence_length"])
    count = int(payload["sequences_per_domain"])
    return start, start + count * length


def main() -> None:
    args = parse_args()
    prior_paths = [path.resolve() for path in args.prior]
    new_paths = [path.resolve() for path in args.new]
    paths = prior_paths + new_paths
    if len(set(paths)) != len(paths):
        raise ValueError("suite paths must be unique")
    payloads = {path: torch.load(path, map_location="cpu", weights_only=False) for path in paths}
    digests = {
        path: {
            domain: {window_digest(window) for window in windows}
            for domain, windows in payload["gates"].items()
        }
        for path, payload in payloads.items()
    }
    comparisons = []
    passed = True
    for new_index, left in enumerate(new_paths):
        peers = prior_paths + new_paths[:new_index]
        for right in peers:
            common_domains = sorted(
                set(payloads[left]["gates"]) & set(payloads[right]["gates"])
            )
            for domain in common_domains:
                exact = len(digests[left][domain] & digests[right][domain])
                left_interval = interval(payloads[left], domain)
                right_interval = interval(payloads[right], domain)
                range_overlap = None
                if left_interval is not None and right_interval is not None:
                    range_overlap = max(left_interval[0], right_interval[0]) < min(
                        left_interval[1], right_interval[1]
                    )
                domain_passed = exact == 0 and range_overlap is not True
                passed = passed and domain_passed
                comparisons.append(
                    {
                        "new": left.name,
                        "other": right.name,
                        "domain": domain,
                        "exact_window_overlap": exact,
                        "new_interval": left_interval,
                        "other_interval": right_interval,
                        "range_overlap": range_overlap,
                        "passed": domain_passed,
                    }
                )
    result = {
        "schema": "wal-tat-suite-overlap-audit-v1",
        "prior_suites": [
            {"path": str(path), "sha256": sha256_file(path)} for path in prior_paths
        ],
        "new_suites": [
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "policy": payloads[path].get("policy"),
                "offsets": payloads[path].get("offsets"),
                "token_sha256": payloads[path].get("token_sha256"),
            }
            for path in new_paths
        ],
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
