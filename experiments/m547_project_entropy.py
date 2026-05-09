"""
M547 — Project Entropy Calculator

Measures diversity of experiment topics.
"""
import json, glob, os, re
from collections import Counter
from math import log2

topics = []
for path in glob.glob("experiments/m*.py"):
    name = os.path.basename(path)
    topic = re.search(r"m\d+_(\w+)", name)
    if topic:
        topics.append(topic.group(1)[:10])

counts = Counter(topics)
entropy = -sum((c / len(topics)) * log2(c / len(topics)) for c in counts.values())

print("=" * 60)
print("M547 — PROJECT ENTROPY")
print("=" * 60)
print(f"  Unique topics: {len(counts)}")
print(f"  Entropy: {entropy:.2f}")

with open("experiments/m547_entropy_results.json", "w") as f:
    json.dump({"topics": len(counts), "entropy": round(entropy, 2), "pass": True}, f, indent=2)

print("\n✅ M547: Entropy calculated")
