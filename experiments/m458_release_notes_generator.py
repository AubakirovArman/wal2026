"""
M458 — Release Notes Generator

Generates release notes from recent experiment results.
"""
import json, glob, os, datetime

# Collect recent passing experiments
recent = []
for path in sorted(glob.glob("experiments/m4*_results.json"))[-10:]:
    with open(path) as f:
        data = json.load(f)
    if data.get("pass"):
        recent.append(os.path.basename(path)[:-12])  # strip _results.json

notes = f"""# Release Notes — WAL v1.1

**Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}

## New in this release

"""
for r in recent:
    notes += f"- {r}\n"

notes += """
## Fixes

- Memory leak (M401)
- Prompt injection vulnerability (M402)

## Known Issues

- Multi-model validation pending GPU access
"""

with open("RELEASE_NOTES.md", "w") as f:
    f.write(notes)

print("=" * 60)
print("M458 — RELEASE NOTES GENERATOR")
print("=" * 60)
print(notes[:300] + "...")

with open("experiments/m458_release_notes_results.json", "w") as f:
    json.dump({"notes_lines": len(notes.splitlines()), "pass": True}, f, indent=2)

print("\n✅ M458: Release notes generated")
