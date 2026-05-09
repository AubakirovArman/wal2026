"""
M403 — GitHub Repository Validation

Validates that all files needed for GitHub publication exist.
"""
import os, json

required = {
    ".github/workflows/ci.yml": "GitHub Actions CI",
    "LICENSE": "MIT License",
    ".gitignore": "Git ignore rules",
    "README.md": "Project README",
    "PROJECT_SUMMARY.md": "Project summary",
    "CITATION.bib": "Academic citation",
    "CONTRIBUTING.md": "Contributing guide",
    "wal_studio_v01/README.md": "WAL Studio README",
    "wal_studio_v01/demo.py": "WAL Studio demo",
}

optional = {
    "CODE_OF_CONDUCT.md": "Code of conduct",
    "SECURITY.md": "Security policy",
}

print("=" * 60)
print("M403 — GITHUB REPO VALIDATION")
print("=" * 60)

missing = []
for path, desc in required.items():
    ok = os.path.exists(path)
    print(f"  {'✅' if ok else '❌'} {desc}: {path}")
    if not ok:
        missing.append(path)

print("\nOptional:")
for path, desc in optional.items():
    ok = os.path.exists(path)
    print(f"  {'✅' if ok else '⚪'} {desc}: {path}")

score = len(required) - len(missing)
print(f"\nScore: {score}/{len(required)}")

with open("experiments/m403_github_validation_results.json", "w") as f:
    json.dump({"required_present": score, "required_total": len(required), "missing": missing, "pass": len(missing) == 0}, f, indent=2)

if len(missing) == 0:
    print("\n✅ M403: Repository ready for GitHub")
else:
    print(f"\n⚠️ M403: Missing {len(missing)} required files")
