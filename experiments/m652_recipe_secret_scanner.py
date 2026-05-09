from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "corpora" / "security_recipe_secret_scan.json"
RESULT_PATH = ROOT / "experiments" / "m652_recipe_secret_scanner_results.json"


PATTERNS = {
    "api_key": re.compile(r"\bsk-(?:test|live)-[A-Za-z0-9]{16,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "bearer_token": re.compile(r"\bBearer\s+[A-Za-z0-9._-]{20,}\b"),
    "pem_block": re.compile(r"-----BEGIN [A-Z ]+-----"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}


RECIPES = [
    {"id": "clean-001", "question": "What is WAL city 001?", "answer": "Verified city 001."},
    {"id": "secret-001", "question": "Store key sk-test-abcdefghijklmnop1234", "answer": "Do not store secrets."},
    {"id": "secret-002", "question": "AWS AKIAABCDEFGHIJKLMNOP check", "answer": "Reject credential-bearing recipe."},
    {"id": "secret-003", "question": "Use Bearer abcdefghijklmnopqrstuvwxyz012345", "answer": "Reject token-bearing recipe."},
    {"id": "secret-004", "question": "-----BEGIN TEST KEY-----", "answer": "Reject PEM-bearing recipe."},
    {"id": "secret-005", "question": "Contact owner@example.com", "answer": "Reject PII-bearing recipe."},
]


def scan_recipe(recipe: dict[str, str]) -> list[str]:
    text = json.dumps(recipe, ensure_ascii=False)
    return [name for name, pattern in PATTERNS.items() if pattern.search(text)]


def main() -> int:
    records = []
    failures = []
    for recipe in RECIPES:
        findings = scan_recipe(recipe)
        should_flag = recipe["id"].startswith("secret-")
        blocked = bool(findings)
        records.append({**recipe, "findings": findings, "blocked": blocked})
        if should_flag and not blocked:
            failures.append({"id": recipe["id"], "reason": "missed_secret"})
        if not should_flag and blocked:
            failures.append({"id": recipe["id"], "reason": "false_positive"})

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    status = "PASS" if not failures else "FAIL"
    result = {
        "schema_version": "wal.results.v1",
        "module": "M652",
        "name": "Recipe Secret Scanner",
        "status": status,
        "pass": status == "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "recipes_checked": len(RECIPES),
        "blocked_recipes": sum(1 for record in records if record["blocked"]),
        "failures": failures,
        "artifact": str(ARTIFACT_PATH.relative_to(ROOT)),
        "scope": "static recipe secret scanning contract",
        "docs": "docs/security_hardening_protocol.md",
    }
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"M652 Recipe Secret Scanner: {status}")
    print(f"recipes={len(RECIPES)} blocked={result['blocked_recipes']} failures={len(failures)}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
