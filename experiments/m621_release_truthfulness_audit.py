"""M621 — Release Truthfulness Audit.

Checks that public release claims use conservative pre-alpha wording and that
known false-positive GPU probes are no longer marked as PASS.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path):
    return json.loads((ROOT / path).read_text())


checks = []

m501 = load_json("experiments/m501_gpu_inference_results.json")
checks.append({
    "name": "M501 classified as blocked",
    "pass": m501.get("status") == "BLOCKED" and m501.get("pass") is False,
    "observed": {"status": m501.get("status"), "pass": m501.get("pass")},
})

m601 = load_json("experiments/m601_gpu_qwen_results.json")
checks.append({
    "name": "M601 classified as unsupported",
    "pass": m601.get("status") == "UNSUPPORTED" and m601.get("pass") is False,
    "observed": {"status": m601.get("status"), "pass": m601.get("pass")},
})

readme = (ROOT / "README.md").read_text()
for forbidden in ("production-ready", "certified A+", "complete and production"):
    checks.append({
        "name": f"README avoids {forbidden}",
        "pass": re.search(re.escape(forbidden), readme, re.IGNORECASE) is None,
        "observed": forbidden,
    })

public_claim_files = [
    "README.md",
    "PROJECT_SUMMARY.md",
    "TECHNICAL_REPORT.md",
    "docs/demo_playbook.md",
    "docs/blocked_script_taxonomy.md",
    "docs/controlled_runners.md",
    "docs/public_claim_policy.md",
    "docs/docs_command_smoke.md",
    "docs/model_small_protocol.md",
    "docs/cross_model_validation_plan.md",
    "docs/robustness_data_protocol.md",
    "docs/ci_hardening_protocol.md",
    "docs/security_hardening_protocol.md",
    "docs/deployment_reality_protocol.md",
    "docs/product_polish_protocol.md",
    "docs/wal_status_summary.md",
    "docs/aigi/README.md",
    "wal_studio_v01/README.md",
    "FINAL_REPORT.html",
    "FINAL_REPORT.json",
    "WAL_EXPORT.json",
    "MILESTONE_v1.2.json",
    "MILESTONE_v1.4.json",
]
for rel_path in public_claim_files:
    text = (ROOT / rel_path).read_text(encoding="utf-8")
    checks.append({
        "name": f"{rel_path} avoids affirmative production-ready claim",
        "pass": re.search(r"\bproduction[- ]ready\b|PRODUCTION READY", text, re.IGNORECASE) is None,
        "observed": rel_path,
    })
    checks.append({
        "name": f"{rel_path} avoids active A+ grade",
        "pass": re.search(r'"grade"\s*:\s*"A\+"|>A\+<|certified-A\+', text, re.IGNORECASE) is None,
        "observed": rel_path,
    })

checks.append({
    "name": "Known issues documented",
    "pass": (ROOT / "KNOWN_ISSUES.md").exists(),
    "observed": "KNOWN_ISSUES.md",
})
checks.append({
    "name": "Result schema documented",
    "pass": (ROOT / "docs/result_schema.md").exists(),
    "observed": "docs/result_schema.md",
})

passed = sum(1 for check in checks if check["pass"])
failed = [check for check in checks if not check["pass"]]
result = {
    "schema_version": "wal.results.v1",
    "status": "PASS" if not failed else "FAIL",
    "pass": not failed,
    "checks_total": len(checks),
    "checks_passed": passed,
    "checks_failed": len(failed),
    "checks": checks,
}

print("=" * 60)
print("M621 — RELEASE TRUTHFULNESS AUDIT")
print("=" * 60)
print(f"  Passed: {passed}/{len(checks)}")
for check in failed:
    print(f"  ❌ {check['name']}: {check['observed']}")

(ROOT / "experiments/m621_release_truthfulness_audit_results.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n"
)
print(f"\nM621 status={result['status']}")
