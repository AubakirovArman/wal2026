"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M327 — Index Generation

Generate index of all book entries.
"""
import json, os

print("=" * 60)
print("M327 — INDEX GENERATION")
print("=" * 60)

# Collect all book entries
books = [f for f in os.listdir("book") if f.endswith(".md") and f != "README.md"]
books.sort()

print(f"\nIndexing {len(books)} book entries...")

# Group by experiment range
index = {}
for book in books:
    # Extract experiment number
    parts = book.split("_")
    if parts[0].isdigit():
        num = int(parts[0])
        if num < 100:
            group = "Early (M1-M99)"
        elif num < 150:
            group = "Phase 1 (M100-M149)"
        elif num < 200:
            group = "Phase 2 (M150-M199)"
        elif num < 250:
            group = "Phase 3 (M200-M249)"
        else:
            group = "Phase 4 (M250+)"
    else:
        group = "Other"
    
    if group not in index:
        index[group] = []
    index[group].append(book)

# Print index
print("\nBook Index:")
for group, entries in sorted(index.items()):
    print(f"\n  {group}: {len(entries)} entries")
    for entry in entries[:3]:
        print(f"    - {entry}")
    if len(entries) > 3:
        print(f"    ... and {len(entries) - 3} more")

# Generate INDEX.md
with open("book/INDEX.md", "w") as f:
    f.write("# WAL Book Index\n\n")
    f.write(f"Total entries: {len(books)}\n\n")
    for group, entries in sorted(index.items()):
        f.write(f"## {group}\n\n")
        for entry in entries:
            title = entry.replace(".md", "").replace("_", " ")
            f.write(f"- [{title}]({entry})\n")
        f.write("\n")

print(f"\nGenerated book/INDEX.md with {len(books)} entries")

results = {
    "total_books": len(books),
    "groups": len(index),
    "index_generated": True,
}

with open("experiments/m327_index_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M327: Index generation complete")
