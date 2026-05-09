"""
M345 — System Health Check

Comprehensive diagnostics of WAL system.
"""
import json, os

print("=" * 60)
print("M345 — SYSTEM HEALTH CHECK")
print("=" * 60)

checks = []

# Check 1: Directories exist
checks.append(("src directory", os.path.isdir("src")))
checks.append(("book directory", os.path.isdir("book")))
checks.append(("docs directory", os.path.isdir("docs")))
checks.append(("experiments directory", os.path.isdir("experiments")))

# Check 2: Key files exist
checks.append(("USER_GUIDE", os.path.exists("docs/USER_GUIDE.md")))
checks.append(("API_REFERENCE", os.path.exists("docs/API_REFERENCE.md")))
checks.append(("dev_diary", os.path.exists("docs/dev_diary_ru.md")))
checks.append(("RELEASE_NOTES", os.path.exists("RELEASE_NOTES.md")))
checks.append(("CONTRIBUTING", os.path.exists("CONTRIBUTING.md")))

# Check 3: Sufficient content
checks.append(("books > 200", len([f for f in os.listdir("book") if f.endswith(".md")]) > 200))
checks.append(("experiments > 50", len([f for f in os.listdir("experiments") if f.endswith(".py")]) > 50))
checks.append(("results > 30", len([f for f in os.listdir("experiments") if f.endswith("_results.json")]) > 30))

# Check 4: ROADMAP exists
checks.append(("ROADMAP exists", len([f for f in os.listdir(".") if f.startswith("ROADMAP")]) > 0))

print("\nHealth checks:")
passed = 0
for name, ok in checks:
    status = "✅" if ok else "❌"
    if ok:
        passed += 1
    print(f"  {status} {name}")

print(f"\nHealth score: {passed}/{len(checks)} ({passed/len(checks):.0%})")

status = "HEALTHY" if passed == len(checks) else "DEGRADED" if passed >= len(checks) * 0.8 else "CRITICAL"
print(f"Status: {status}")

with open("experiments/m345_health_results.json", "w") as f:
    json.dump({"passed": passed, "total": len(checks), "status": status}, f, indent=2)

print("\n✅ M345: Health check complete")
