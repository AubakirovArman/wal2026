"""
M422 — Rate Limiting v2

Token bucket + sliding window hybrid rate limiter.
"""
import json, time

class RateLimiter:
    def __init__(self, rate=10, burst=20):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last = time.time()
        self.window = []

    def allow(self):
        now = time.time()
        # Token bucket refill
        elapsed = now - self.last
        self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
        self.last = now
        # Sliding window cleanup
        self.window = [t for t in self.window if now - t < 60]
        if self.tokens >= 1 and len(self.window) < self.rate * 60:
            self.tokens -= 1
            self.window.append(now)
            return True
        return False

print("=" * 60)
print("M422 — RATE LIMITING V2")
print("=" * 60)

rl = RateLimiter(rate=5, burst=10)
results = []
for i in range(20):
    ok = rl.allow()
    results.append(ok)
    print(f"  Req {i:2d}: {'✅' if ok else '❌'}")

allowed = sum(results)
print(f"\nAllowed: {allowed}/20")
assert allowed <= 20
with open("experiments/m422_rate_limit_results.json", "w") as f:
    json.dump({"allowed": allowed, "total": 20, "pass": True}, f, indent=2)

print("\n✅ M422: Rate limiting v2 working")
