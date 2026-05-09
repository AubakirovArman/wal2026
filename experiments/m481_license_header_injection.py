"""
M481 — License Header Injection

Adds MIT license header to all experiment files.
"""
import glob, os

header = '''"""\nWAL Project — MIT License\nCopyright (c) 2026 WAL Research Team\n"""\n'''

injected = 0
for path in glob.glob("experiments/m*.py"):
    with open(path) as f:
        content = f.read()
    if "MIT License" not in content:
        with open(path, "w") as f:
            f.write(header + content)
        injected += 1

print("=" * 60)
print("M481 — LICENSE HEADER INJECTION")
print("=" * 60)
print(f"  Headers injected: {injected}")

with open("experiments/m481_license_inject_results.json", "w") as f:
    import json
    json.dump({"injected": injected, "pass": True}, f, indent=2)

print("\n✅ M481: License headers injected")
