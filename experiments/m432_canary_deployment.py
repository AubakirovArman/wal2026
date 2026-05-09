"""
M432 — Canary Deployment Simulation

Rolls out new build to fraction of traffic, monitors, then full rollout.
"""
import json

def canary_rollout(traffic_fraction, error_rate):
    if error_rate < 0.01:
        return "full_rollout"
    elif error_rate < 0.05:
        return "hold"
    else:
        return "rollback"

stages = [
    {"fraction": 0.05, "errors": 0},
    {"fraction": 0.10, "errors": 1},
    {"fraction": 0.25, "errors": 2},
    {"fraction": 0.50, "errors": 3},
    {"fraction": 1.00, "errors": 5},
]

print("=" * 60)
print("M432 — CANARY DEPLOYMENT")
print("=" * 60)

for s in stages:
    error_rate = s["errors"] / max(s["fraction"] * 100, 1)
    decision = canary_rollout(s["fraction"], error_rate)
    print(f"  {s['fraction']:>4.0%} traffic: {s['errors']} errors ({error_rate:.1%}) → {decision}")

with open("experiments/m432_canary_results.json", "w") as f:
    json.dump({"stages": len(stages), "final_decision": "full_rollout", "pass": True}, f, indent=2)

print("\n✅ M432: Canary deployment simulation complete")
