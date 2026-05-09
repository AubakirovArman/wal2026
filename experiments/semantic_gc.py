"""
Wild Idea #6 — Semantic Garbage Collector

Remove edits with low utility, high drift, or obsolete domain.
"""
import json, time

class SemanticGC:
    def __init__(self, drift_threshold=2.0, age_threshold=3600):
        self.drift_threshold = drift_threshold
        self.age_threshold = age_threshold
    
    def should_collect(self, edit):
        """Check if edit should be garbage collected."""
        reasons = []
        
        # High PPL drift
        if edit.get("ppl_drift", 0) > self.drift_threshold:
            reasons.append("high_drift")
        
        # Low survival
        if edit.get("survival_rate", 1.0) < 0.5:
            reasons.append("low_survival")
        
        # Too old
        age = time.time() - edit.get("timestamp", time.time())
        if age > self.age_threshold:
            reasons.append("obsolete")
        
        # Never tested
        if edit.get("last_tested", 0) == 0:
            reasons.append("untested")
        
        return len(reasons) > 0, reasons

edits = [
    {"id": 0, "ppl_drift": 0.1, "survival_rate": 1.0, "timestamp": time.time(), "last_tested": time.time()},
    {"id": 1, "ppl_drift": 3.5, "survival_rate": 0.3, "timestamp": time.time() - 4000, "last_tested": 0},
    {"id": 2, "ppl_drift": 0.2, "survival_rate": 0.9, "timestamp": time.time() - 5000, "last_tested": time.time()},
    {"id": 3, "ppl_drift": 0.05, "survival_rate": 1.0, "timestamp": time.time(), "last_tested": 0},
]

print("=" * 60)
print("SEMANTIC GARBAGE COLLECTOR")
print("=" * 60)

gc = SemanticGC()
collected = []
kept = []

for edit in edits:
    should, reasons = gc.should_collect(edit)
    if should:
        collected.append({"id": edit["id"], "reasons": reasons})
        print(f"  🗑️  Edit #{edit['id']}: COLLECTED ({', '.join(reasons)})")
    else:
        kept.append(edit["id"])
        print(f"  💾 Edit #{edit['id']}: KEPT")

print(f"\n🎯 GC: {len(collected)} collected, {len(kept)} kept")
with open("experiments/semantic_gc_results.json", "w") as f:
    json.dump({"collected": collected, "kept": kept}, f, indent=2)
