"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M449 — Version Compatibility Check

Ensures WAL weights are compatible with model version.
"""
import json

def compatible(model_version, wal_version):
    # Major version must match
    return model_version.split(".")[0] == wal_version.split(".")[0]

tests = [
    ("3.1.0", "3.0.1", True),
    ("3.1.0", "2.0.0", False),
    ("4.0.0", "4.1.2", True),
]

print("=" * 60)
print("M449 — VERSION COMPATIBILITY")
print("=" * 60)

passed = 0
for mv, wv, expected in tests:
    ok = compatible(mv, wv)
    status = ok == expected
    if status:
        passed += 1
    print(f"  Model {mv} + WAL {wv}: {'✅' if status else '❌'} (compatible={ok})")

assert passed == len(tests)
with open("experiments/m449_compatibility_results.json", "w") as f:
    json.dump({"passed": passed, "total": len(tests), "pass": True}, f, indent=2)

print("\n✅ M449: Version compatibility check working")
