"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M346 — Version Compatibility

Ensure backward compatibility between versions.
"""
import json

print("=" * 60)
print("M346 — VERSION COMPATIBILITY")
print("=" * 60)

# Version history
versions = {
    "v1.0": {"recipes": [1, 2, 3], "format": "legacy"},
    "v1.1": {"recipes": [1, 2, 3, 4], "format": "legacy"},
    "v2.0": {"recipes": [1, 2, 3, 4, 5], "format": "current"},
}

def is_compatible(old_ver, new_ver):
    """Check if old recipes work with new version."""
    old = versions[old_ver]
    new = versions[new_ver]
    # Compatible if old recipes are subset of new
    return set(old["recipes"]).issubset(set(new["recipes"]))

print("\nCompatibility matrix:")
print(f"{'Old \\ New':>10s}", end="")
for v in versions:
    print(f"{v:>8s}", end="")
print()

for old in versions:
    print(f"{old:>10s}", end="")
    for new in versions:
        compat = is_compatible(old, new)
        print(f"{'✅' if compat else '❌':>8s}", end="")
    print()

with open("experiments/m346_compat_results.json", "w") as f:
    json.dump({"versions": len(versions)}, f, indent=2)

print("\n✅ M346: Version compatibility checked")
