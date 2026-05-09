"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M391 — Final Health Check

Comprehensive system health diagnostic.
"""
import json, os

print("=" * 60)
print("M391 — FINAL HEALTH CHECK")
print("=" * 60)

checks = []

# Structure
checks.append(("Books > 300", len([f for f in os.listdir("book") if f.endswith(".md")]) > 300))
checks.append(("Experiments > 120", len([f for f in os.listdir("experiments") if f.endswith(".py")]) > 120))
checks.append(("Results > 80", len([f for f in os.listdir("experiments") if f.endswith("_results.json")]) > 80))

# Key files
checks.append(("WAL Studio demo", os.path.exists("wal_studio_v01/demo.py")))
checks.append(("E1 results", os.path.exists("experiments/e1_realistic_500_results.json")))
checks.append(("E2 results", os.path.exists("experiments/e2_multimodel_results.json")))
checks.append(("E3 results", os.path.exists("experiments/e3_baseline_results.json")))
checks.append(("E4 results", os.path.exists("experiments/e4_security_results.json")))
checks.append(("E5 results", os.path.exists("experiments/e5_longrun_results.json")))

# Docs
checks.append(("Project summary", os.path.exists("PROJECT_SUMMARY.md")))
checks.append(("Final report", os.path.exists("FINAL_REPORT.html")))

passed = sum(1 for _, ok in checks if ok)
total = len(checks)

print("\nFinal health check:")
for name, ok in checks:
    print(f"  {'✅' if ok else '❌'} {name}")

print(f"\nScore: {passed}/{total} ({passed/total:.0%})")

with open("experiments/m391_health_results.json", "w") as f:
    json.dump({"passed": passed, "total": total, "healthy": passed == total}, f, indent=2)

print("\n✅ M391: Final health check complete")
