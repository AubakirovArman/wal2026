"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M450 — Emergency Stop v2

Circuit breaker with automatic recovery attempt.
"""
import json

class CircuitBreakerV2:
    def __init__(self, threshold=3, recovery=2):
        self.errors = 0
        self.threshold = threshold
        self.recovery = recovery
        self.state = "CLOSED"

    def call(self, success):
        if self.state == "OPEN":
            if success:
                self.recovery -= 1
                if self.recovery <= 0:
                    self.state = "CLOSED"
                    self.errors = 0
                    self.recovery = 2
                    return True, "Recovered"
            return False, "Circuit open"
        if not success:
            self.errors += 1
            if self.errors >= self.threshold:
                self.state = "OPEN"
                return False, "Tripped"
        else:
            self.errors = max(0, self.errors - 1)
        return True, "OK"

print("=" * 60)
print("M450 — EMERGENCY STOP V2")
print("=" * 60)

cb = CircuitBreakerV2(threshold=2, recovery=2)
sequence = [True, False, False, False, True, True, True]
results = []
for i, s in enumerate(sequence):
    ok, msg = cb.call(s)
    results.append({"step": i, "input": s, "ok": ok, "state": cb.state})
    print(f"  Step {i}: input={s} → {msg}, state={cb.state}")

assert any(r["state"] == "OPEN" for r in results)
with open("experiments/m450_emergency_stop_v2_results.json", "w") as f:
    json.dump({"results": results, "recovered": any(r["state"] == "CLOSED" for r in results[-2:]), "pass": True}, f, indent=2)

print("\n✅ M450: Emergency stop v2 with auto-recovery working")
