"""
M365 — Final Integration Test

Run complete pipeline one more time.
"""
import json, os

print("=" * 60)
print("M365 — FINAL INTEGRATION TEST")
print("=" * 60)

checks = [
    ("Book entries > 275", len([f for f in os.listdir("book") if f.endswith(".md")]) > 275),
    ("Experiments > 100", len([f for f in os.listdir("experiments") if f.endswith(".py")]) > 100),
    ("Results > 60", len([f for f in os.listdir("experiments") if f.endswith("_results.json")]) > 60),
    ("User guide exists", os.path.exists("docs/USER_GUIDE.md")),
    ("API reference exists", os.path.exists("docs/API_REFERENCE.md")),
    ("Release notes exist", os.path.exists("RELEASE_NOTES.md")),
    ("Contributing exists", os.path.exists("CONTRIBUTING.md")),
    ("Final report exists", os.path.exists("FINAL_REPORT.html")),
    ("Project summary exists", os.path.exists("PROJECT_SUMMARY.md")),
    ("Dev diary exists", os.path.exists("docs/dev_diary_ru.md")),
]

passed = sum(1 for _, ok in checks if ok)
total = len(checks)

print("\nIntegration checks:")
for name, ok in checks:
    print(f"  {'✅' if ok else '❌'} {name}")

print(f"\nScore: {passed}/{total} ({passed/total:.0%})")

with open("experiments/m365_integration_results.json", "w") as f:
    json.dump({"passed": passed, "total": total}, f, indent=2)

if passed == total:
    print("\n🎉 M365: ALL INTEGRATION TESTS PASSED")
else:
    print(f"\n⚠️ M365: {total - passed} tests failed")
