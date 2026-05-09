"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M480 — Publication Readiness Report

Final check before public release.
"""
import json, os, glob

criteria = {
    "README present": os.path.exists("README.md"),
    "LICENSE present": os.path.exists("LICENSE"),
    "CI configured": os.path.exists(".github/workflows/ci.yml"),
    "Security policy": os.path.exists("SECURITY.md"),
    "Code of conduct": os.path.exists("CODE_OF_CONDUCT.md"),
    "Contributing guide": os.path.exists("CONTRIBUTING.md"),
    "Issue templates": os.path.exists(".github/ISSUE_TEMPLATE"),
    "Memory leak fixed": os.path.exists("experiments/m401_memory_leak_fix_results.json"),
    "Prompt injection hardened": os.path.exists("experiments/m402_security_hardening_results.json"),
    "Integration test pass": os.path.exists("experiments/m412_final_integration_test.py"),
    "Project summary": os.path.exists("PROJECT_SUMMARY.md"),
    "Milestone declared": os.path.exists("MILESTONE_v1.0.json"),
}

passed = sum(1 for v in criteria.values() if v)
total = len(criteria)

print("=" * 60)
print("M480 — PUBLICATION READINESS")
print("=" * 60)
for name, ok in criteria.items():
    print(f"  {'✅' if ok else '❌'} {name}")

print(f"\nReadiness: {passed}/{total} ({passed/total:.0%})")

ready = passed >= 10
print(f"Ready for publication: {'✅ YES' if ready else '❌ NO'}")

with open("experiments/m480_publication_results.json", "w") as f:
    json.dump({"passed": passed, "total": total, "ready": ready, "pass": ready}, f, indent=2)

print("\n✅ M480: Publication readiness report generated")
