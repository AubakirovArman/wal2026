"""
M353 — Book Integrity Check

Verify all book entries are valid.
"""
import json, os

print("=" * 60)
print("M353 — BOOK INTEGRITY CHECK")
print("=" * 60)

books = [f for f in os.listdir("book") if f.endswith(".md") and f != "README.md"]

issues = []
for book in books:
    path = f"book/{book}"
    with open(path) as f:
        content = f.read()
    
    # Check required sections
    if "# " not in content:
        issues.append(f"{book}: missing title")
    if "## Date" not in content:
        issues.append(f"{book}: missing date")
    if "## Verdict" not in content:
        issues.append(f"{book}: missing verdict")

print(f"\nChecked {len(books)} book entries")
print(f"Issues found: {len(issues)}")

if issues:
    for issue in issues[:5]:
        print(f"  ⚠️ {issue}")
    if len(issues) > 5:
        print(f"  ... and {len(issues) - 5} more")
else:
    print("  ✅ All book entries valid")

with open("experiments/m353_integrity_results.json", "w") as f:
    json.dump({"total": len(books), "issues": len(issues)}, f, indent=2)

print("\n✅ M353: Book integrity check complete")
