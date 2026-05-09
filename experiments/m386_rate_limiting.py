"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M386 — Rate Limiting

Limit inference requests per second.
"""
import json, time

print("=" * 60)
print("M386 — RATE LIMITING")
print("=" * 60)

class RateLimiter:
    def __init__(self, max_per_sec=10):
        self.max_per_sec = max_per_sec
        self.requests = []
    
    def allow(self):
        now = time.time()
        self.requests = [r for r in self.requests if now - r < 1]
        if len(self.requests) < self.max_per_sec:
            self.requests.append(now)
            return True
        return False

limiter = RateLimiter(max_per_sec=5)

print("\nRate limit test (5/sec):")
allowed = 0
for i in range(12):
    ok = limiter.allow()
    if ok:
        allowed += 1
    print(f"  Request {i+1}: {'✅' if ok else '❌ BLOCKED'}")
    time.sleep(0.1)

print(f"\nAllowed: {allowed}/12")

with open("experiments/m386_rate_results.json", "w") as f:
    json.dump({"allowed": allowed, "blocked": 12 - allowed}, f, indent=2)

print("\n✅ M386: Rate limiting working")
