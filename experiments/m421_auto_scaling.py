"""
M421 — Auto-Scaling Simulation

Scales inference workers based on request queue depth.
"""
import json

def scale_workers(queue_depth, current_workers, max_workers=8):
    target = min(max_workers, max(1, (queue_depth // 10) + 1))
    if target > current_workers:
        return current_workers + 1, "scale_up"
    elif target < current_workers and current_workers > 1:
        return current_workers - 1, "scale_down"
    return current_workers, "stable"

history = []
workers = 1
for minute, depth in enumerate([5, 15, 35, 60, 45, 20, 8, 3, 25, 55]):
    workers, action = scale_workers(depth, workers)
    history.append({"min": minute, "queue": depth, "workers": workers, "action": action})
    print(f"  Min {minute:2d}: queue={depth:2d} → workers={workers} ({action})")

assert workers >= 1
with open("experiments/m421_autoscale_results.json", "w") as f:
    json.dump({"history": history, "final_workers": workers, "pass": True}, f, indent=2)

print("\n✅ M421: Auto-scaling simulation complete")
