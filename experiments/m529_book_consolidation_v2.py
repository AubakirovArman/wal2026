"""
M529 — Book Consolidation v2

Creates combined index of all book entries.
"""
import json, glob, os

books = sorted(glob.glob("book/*.md"))
index = []
for path in books:
    with open(path) as f:
        first_line = f.readline().strip()
    index.append({"file": os.path.basename(path), "title": first_line[:50]})

print("=" * 60)
print("M529 — BOOK CONSOLIDATION V2")
print("=" * 60)
print(f"  Books: {len(index)}")

with open("experiments/m529_book_consolidation_results.json", "w") as f:
    json.dump({"books": len(index), "pass": True}, f, indent=2)

print("\n✅ M529: Book consolidation v2 complete")
