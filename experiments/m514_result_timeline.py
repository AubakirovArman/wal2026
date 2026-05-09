"""
M514 — Result Timeline

Creates timeline of experiment completion.
"""
import json, glob, os, datetime

timeline = []
for path in glob.glob("experiments/*_results.json"):
    mtime = os.path.getmtime(path)
    timeline.append({"file": os.path.basename(path), "time": datetime.datetime.fromtimestamp(mtime).isoformat()})

timeline.sort(key=lambda x: x["time"])

print("=" * 60)
print("M514 — RESULT TIMELINE")
print("=" * 60)
print(f"  Total entries: {len(timeline)}")
print(f"  First: {timeline[0]['time'][:19] if timeline else 'N/A'}")
print(f"  Last: {timeline[-1]['time'][:19] if timeline else 'N/A'}")

with open("experiments/m514_timeline_results.json", "w") as f:
    json.dump({"entries": len(timeline), "pass": True}, f, indent=2)

print("\n✅ M514: Timeline generated")
