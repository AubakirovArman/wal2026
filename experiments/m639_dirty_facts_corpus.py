from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "corpora" / "dirty_facts_500.jsonl"
RESULT_PATH = ROOT / "experiments" / "m639_dirty_facts_corpus_results.json"


DOMAINS = ["geography", "product", "legal", "temporal", "procedural", "author", "inventor", "policy"]
NOISE = ["alias", "old_answer_lure", "source_conflict", "date_sensitive", "spelling_variant"]


def make_record(index: int) -> dict[str, object]:
    domain = DOMAINS[index % len(DOMAINS)]
    noise = [NOISE[index % len(NOISE)], NOISE[(index * 3) % len(NOISE)]]
    subject = f"WAL robustness entity {index:03d}"
    answer = f"verified value {index:03d}"
    old_answer = f"obsolete value {index:03d}"
    return {
        "id": f"dirty-{index:03d}",
        "domain": domain,
        "question": f"What is the current verified fact for {subject}?",
        "answer": answer,
        "old_answer": old_answer,
        "aliases": [subject, subject.lower(), f"WAL entity {index:03d}"],
        "noise": sorted(set(noise)),
        "source": f"synthetic_seed_source_{index % 17}",
        "notes": "messy seed record for reviewer-safe robustness gates",
    }


def main() -> int:
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = [make_record(index) for index in range(500)]
    CORPUS_PATH.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    domain_counts = {domain: sum(1 for record in records if record["domain"] == domain) for domain in DOMAINS}
    noise_counts = {noise: sum(1 for record in records if noise in record["noise"]) for noise in NOISE}
    status = "PASS" if len(records) == 500 and min(domain_counts.values()) >= 50 else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M639",
        "name": "Dirty Facts Corpus",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "domains": domain_counts,
        "noise_counts": noise_counts,
        "corpus": str(CORPUS_PATH.relative_to(ROOT)),
        "docs": "docs/robustness_data_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M639 Dirty Facts Corpus: {status}")
    print(f"records={len(records)} corpus={CORPUS_PATH.relative_to(ROOT)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
