"""
M464 — Load Balancer Simulation

Round-robin load balancing across workers.
"""
import json

class LoadBalancer:
    def __init__(self, workers):
        self.workers = workers
        self.index = 0
        self.counts = {w: 0 for w in workers}

    def route(self):
        worker = self.workers[self.index]
        self.index = (self.index + 1) % len(self.workers)
        self.counts[worker] += 1
        return worker

print("=" * 60)
print("M464 — LOAD BALANCER SIMULATION")
print("=" * 60)

lb = LoadBalancer(["worker-1", "worker-2", "worker-3"])
for i in range(9):
    w = lb.route()
    print(f"  Request {i+1} → {w}")

balanced = max(lb.counts.values()) - min(lb.counts.values()) <= 1
print(f"\nBalanced: {'✅' if balanced else '❌'}")
assert balanced

with open("experiments/m464_loadbalancer_results.json", "w") as f:
    json.dump({"counts": lb.counts, "balanced": balanced, "pass": True}, f, indent=2)

print("\n✅ M464: Load balancer simulation working")
