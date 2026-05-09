"""
M415 — Fact Lifecycle Management

Tracks facts through creation, validation, deployment, deprecation.
"""
import json
from datetime import datetime, timedelta

class FactLifecycle:
    STATES = ["draft", "validated", "deployed", "deprecated", "archived"]

    def __init__(self, fact_id):
        self.id = fact_id
        self.state = "draft"
        self.history = [{"state": "draft", "at": datetime.now().isoformat()}]

    def transition(self, to_state):
        if to_state not in self.STATES:
            raise ValueError(f"Invalid state: {to_state}")
        idx_from = self.STATES.index(self.state)
        idx_to = self.STATES.index(to_state)
        if idx_to < idx_from:
            raise ValueError(f"Cannot transition {self.state} → {to_state}")
        self.state = to_state
        self.history.append({"state": to_state, "at": datetime.now().isoformat()})

print("=" * 60)
print("M415 — FACT LIFECYCLE MANAGEMENT")
print("=" * 60)

fact = FactLifecycle("fact_001")
fact.transition("validated")
fact.transition("deployed")
print(f"  Fact {fact.id}: {fact.state}")
print(f"  History: {[h['state'] for h in fact.history]}")

assert fact.state == "deployed"
assert len(fact.history) == 3

with open("experiments/m415_lifecycle_results.json", "w") as f:
    json.dump({"fact_id": fact.id, "state": fact.state, "history": fact.history, "pass": True}, f, indent=2)

print("\n✅ M415: Fact lifecycle management working")
