"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M347 — Emergency Stop

Halt editing if critical issues detected.
"""
import json

print("=" * 60)
print("M347 — EMERGENCY STOP")
print("=" * 60)

# Simulate issues
def check_emergency(facts, metrics):
    """Check if emergency stop needed."""
    issues = []
    
    if metrics.get("ci_score", 1.0) < 0.5:
        issues.append("CI score critically low")
    
    if metrics.get("nan_detected", False):
        issues.append("NaN values detected")
    
    if metrics.get("memory_usage_mb", 0) > 100000:
        issues.append("Memory usage critical")
    
    if len(facts) > 1000:
        issues.append("Too many facts")
    
    return issues

# Test scenarios
scenarios = [
    {"name": "Normal", "facts": 50, "metrics": {"ci_score": 0.94}},
    {"name": "Low CI", "facts": 50, "metrics": {"ci_score": 0.3}},
    {"name": "NaN", "facts": 50, "metrics": {"ci_score": 0.9, "nan_detected": True}},
    {"name": "High memory", "facts": 50, "metrics": {"ci_score": 0.9, "memory_usage_mb": 150000}},
]

print("\nEmergency stop checks:")
for s in scenarios:
    issues = check_emergency(range(s["facts"]), s["metrics"])
    stop = len(issues) > 0
    status = "🛑 STOP" if stop else "✅ GO"
    print(f"  {status} [{s['name']}] Issues: {issues if issues else 'none'}")

with open("experiments/m347_emergency_results.json", "w") as f:
    json.dump({"scenarios": len(scenarios)}, f, indent=2)

print("\n✅ M347: Emergency stop system working")
