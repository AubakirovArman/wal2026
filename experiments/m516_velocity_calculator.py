"""
M516 — Velocity Calculator

Calculates experiment creation velocity.
"""
import json, glob, os

files = glob.glob("experiments/m*.py")
times = [os.path.getmtime(f) for f in files]
if len(times) > 1:
    duration_hours = (max(times) - min(times)) / 3600
    velocity = len(files) / max(duration_hours, 1)
else:
    velocity = 0

print("=" * 60)
print("M516 — VELOCITY CALCULATOR")
print("=" * 60)
print(f"  Experiments: {len(files)}")
print(f"  Duration: {duration_hours:.1f} hours")
print(f"  Velocity: {velocity:.1f} experiments/hour")

with open("experiments/m516_velocity_results.json", "w") as f:
    json.dump({"experiments": len(files), "velocity_per_hour": round(velocity, 2), "pass": True}, f, indent=2)

print("\n✅ M516: Velocity calculated")
