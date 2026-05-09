"""
M397 — Validate JSON Results

Check all JSON result files are valid.
"""
import json, os, glob

valid = 0
invalid = 0
for path in glob.glob("experiments/*_results.json"):
    try:
        with open(path) as f:
            json.load(f)
        valid += 1
    except Exception as e:
        print(f"  ❌ Invalid: {path} — {e}")
        invalid += 1

print(f"✅ M397: {valid} valid, {invalid} invalid JSON files")
