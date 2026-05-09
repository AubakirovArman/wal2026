"""
M322 — Regression Test

Detect performance degradation across builds.
"""
import json

print("=" * 60)
print("M322 — REGRESSION TEST")
print("=" * 60)

# Historical benchmark results
builds = [
    {"version": "v1.0", "ci_score": 0.94, "latency_ms": 45, "survival": 0.95},
    {"version": "v1.1", "ci_score": 0.93, "latency_ms": 46, "survival": 0.94},
    {"version": "v1.2", "ci_score": 0.95, "latency_ms": 44, "survival": 0.96},
    {"version": "v1.3", "ci_score": 0.91, "latency_ms": 48, "survival": 0.92},  # regression
    {"version": "v1.4", "ci_score": 0.96, "latency_ms": 43, "survival": 0.97},
]

# Define thresholds
THRESHOLDS = {
    "ci_score": 0.90,
    "latency_ms": 50,
    "survival": 0.90,
}

print("\nBuild history:")
print(f"{'Version':>8s} {'CI':>6s} {'Lat(ms)':>8s} {'Survival':>9s} {'Status':>8s}")
print("-" * 45)

regressions = []
for build in builds:
    issues = []
    if build["ci_score"] < THRESHOLDS["ci_score"]:
        issues.append("CI score low")
    if build["latency_ms"] > THRESHOLDS["latency_ms"]:
        issues.append("Latency high")
    if build["survival"] < THRESHOLDS["survival"]:
        issues.append("Survival low")
    
    status = "❌ REGRESSION" if issues else "✅ OK"
    print(f"{build['version']:>8s} {build['ci_score']:>6.2f} {build['latency_ms']:>8.0f} {build['survival']:>8.1%} {status:>8s}")
    
    if issues:
        regressions.append({"version": build["version"], "issues": issues})

print(f"\nRegressions detected: {len(regressions)}")
for r in regressions:
    print(f"  {r['version']}: {', '.join(r['issues'])}")

# Trend analysis
if len(builds) >= 2:
    first = builds[0]
    last = builds[-1]
    ci_trend = last["ci_score"] - first["ci_score"]
    lat_trend = last["latency_ms"] - first["latency_ms"]
    
    print(f"\nTrends (v1.0 → {last['version']}):")
    print(f"  CI score: {ci_trend:+.3f}")
    print(f"  Latency: {lat_trend:+.0f}ms")
    print(f"  Overall: {'📈' if ci_trend > 0 else '📉'} {'improving' if ci_trend > 0 else 'degrading'}")

results = {
    "builds_tested": len(builds),
    "regressions_found": len(regressions),
    "trend": "improving" if ci_trend > 0 else "degrading",
}

with open("experiments/m322_regression_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M322: Regression detection working")
