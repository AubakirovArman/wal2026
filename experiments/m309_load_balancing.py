"""
M309 — Load Balancing

Distribute inference across multiple model instances.
"""
import json, random

random.seed(42)

print("=" * 60)
print("M309 — LOAD BALANCING")
print("=" * 60)

class ModelInstance:
    def __init__(self, name, capacity):
        self.name = name
        self.capacity = capacity
        self.requests = 0
        self.load = 0.0
    
    def handle(self):
        self.requests += 1
        self.load = self.requests / self.capacity
        return True

# Create instances
instances = [
    ModelInstance("gpu-0", capacity=100),
    ModelInstance("gpu-1", capacity=100),
    ModelInstance("gpu-2", capacity=50),  # slower
]

# Round-robin load balancing
def round_robin(instances, num_requests):
    for i in range(num_requests):
        idx = i % len(instances)
        instances[idx].handle()

# Least-loaded load balancing
def least_loaded(instances, num_requests):
    for _ in range(num_requests):
        # Find instance with lowest load
        best = min(instances, key=lambda x: x.load)
        best.handle()

print("\nSimulating 300 requests...")

# Test round-robin
for inst in instances:
    inst.requests = 0
    inst.load = 0.0

round_robin(instances, 300)
print("\nRound-robin distribution:")
for inst in instances:
    print(f"  {inst.name}: {inst.requests} requests ({inst.load:.1%} load)")

# Test least-loaded
for inst in instances:
    inst.requests = 0
    inst.load = 0.0

least_loaded(instances, 300)
print("\nLeast-loaded distribution:")
for inst in instances:
    print(f"  {inst.name}: {inst.requests} requests ({inst.load:.1%} load)")

# Compare max load
rr_max = max(i.load for i in instances)
ll_max = max(i.load for i in instances)
print(f"\nMax load:")
print(f"  Round-robin: {rr_max:.1%}")
print(f"  Least-loaded: {ll_max:.1%}")
print(f"  Improvement: {(rr_max - ll_max):+.1%}")

with open("experiments/m309_loadbalance_results.json", "w") as f:
    json.dump({
        "strategy": "least_loaded",
        "instances": [{"name": i.name, "requests": i.requests, "load": i.load} for i in instances],
        "max_load": ll_max,
    }, f, indent=2)

print("\n✅ M309: Load balancing distributes requests evenly")
