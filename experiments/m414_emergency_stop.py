"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M414 — Emergency Stop System

Implements circuit breaker for WAL server under high load.
"""
import json

class CircuitBreaker:
    def __init__(self, threshold=5, timeout=60):
        self.errors = 0
        self.threshold = threshold
        self.timeout = timeout
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.total_blocked = 0

    def call(self, success):
        if self.state == "OPEN":
            self.total_blocked += 1
            return False, "Circuit open"
        if not success:
            self.errors += 1
            if self.errors >= self.threshold:
                self.state = "OPEN"
                return False, "Circuit tripped"
        else:
            self.errors = max(0, self.errors - 1)
        return True, "OK"

    def reset(self):
        self.state = "CLOSED"
        self.errors = 0

print("=" * 60)
print("M414 — EMERGENCY STOP SYSTEM")
print("=" * 60)

cb = CircuitBreaker(threshold=3)
results = []
for i, success in enumerate([True, False, False, True, False, False, False, True, True]):
    ok, msg = cb.call(success)
    results.append({"req": i, "success": success, "allowed": ok, "state": cb.state})
    print(f"  Req {i}: success={success} → allowed={ok}, state={cb.state}")

print(f"\nFinal state: {cb.state}")
print(f"Blocked requests: {cb.total_blocked}")
assert cb.state == "OPEN"
assert cb.total_blocked >= 1

with open("experiments/m414_emergency_stop_results.json", "w") as f:
    json.dump({"state": cb.state, "blocked": cb.total_blocked, "history": results, "pass": True}, f, indent=2)

print("\n✅ M414: Emergency stop system working")
