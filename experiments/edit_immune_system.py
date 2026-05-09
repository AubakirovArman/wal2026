"""
Wild Idea #3 — Edit Immune System

CI negative tests act as immunity against harmful edits.
An edit that fails negative tests is "rejected" like a pathogen.
"""
import json

# Simulate immune system
class EditImmuneSystem:
    def __init__(self):
        self.antibodies = ["negative_test", "ppl_gate", "no_nan"]
        self.immunity_log = []
    
    def test_edit(self, edit):
        """Test edit against immune system."""
        infections = []
        
        if not edit.get("negative_pass", False):
            infections.append("negative_test")
        if edit.get("ppl", 999) > 3.0:
            infections.append("ppl_gate")
        if edit.get("has_nan", True):
            infections.append("no_nan")
        
        is_healthy = len(infections) == 0
        self.immunity_log.append({
            "edit_id": edit.get("id"),
            "healthy": is_healthy,
            "infections": infections,
        })
        return is_healthy, infections

immune = EditImmuneSystem()

edits = [
    {"id": 0, "negative_pass": True, "ppl": 2.0, "has_nan": False},
    {"id": 1, "negative_pass": False, "ppl": 2.0, "has_nan": False},  # Bad edit
    {"id": 2, "negative_pass": True, "ppl": 5.0, "has_nan": False},  # PPL too high
    {"id": 3, "negative_pass": True, "ppl": 1.5, "has_nan": True},   # NaN
    {"id": 4, "negative_pass": True, "ppl": 1.8, "has_nan": False},  # Good
]

print("=" * 60)
print("EDIT IMMUNE SYSTEM")
print("=" * 60)

for edit in edits:
    healthy, infections = immune.test_edit(edit)
    status = "✅ HEALTHY" if healthy else "❌ REJECTED"
    print(f"  Edit #{edit['id']}: {status}")
    if infections:
        print(f"    Infections: {', '.join(infections)}")

healthy_count = sum(1 for log in immune.immunity_log if log["healthy"])

print(f"\n🎯 Immune system: {healthy_count}/{len(edits)} edits accepted")
with open("experiments/edit_immune_system_results.json", "w") as f:
    json.dump(immune.immunity_log, f, indent=2)
