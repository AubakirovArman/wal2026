"""
M546 — Documentation Word Count

Counts words in all documentation.
"""
import json, glob

total = 0
for path in glob.glob("docs/**/*.md", recursive=True):
    with open(path) as f:
        total += len(f.read().split())

print("=" * 60)
print("M546 — DOC WORD COUNT")
print("=" * 60)
print(f"  Total words: {total}")

with open("experiments/m546_word_count_results.json", "w") as f:
    json.dump({"words": total, "pass": True}, f, indent=2)

print("\n✅ M546: Word count complete")
