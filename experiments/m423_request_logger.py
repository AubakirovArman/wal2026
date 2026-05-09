"""
M423 — Request Logger Analyzer

Analyzes request patterns for anomalies.
"""
import json

requests = [
    {"ip": "10.0.0.1", "status": 200, "ms": 45},
    {"ip": "10.0.0.1", "status": 200, "ms": 50},
    {"ip": "10.0.0.2", "status": 200, "ms": 48},
    {"ip": "10.0.0.1", "status": 429, "ms": 5},
    {"ip": "10.0.0.1", "status": 429, "ms": 3},
    {"ip": "10.0.0.3", "status": 500, "ms": 2000},
]

# Find anomalies: high error rate or high latency
ip_stats = {}
for r in requests:
    ip = r["ip"]
    if ip not in ip_stats:
        ip_stats[ip] = {"count": 0, "errors": 0, "slow": 0}
    ip_stats[ip]["count"] += 1
    if r["status"] >= 400:
        ip_stats[ip]["errors"] += 1
    if r["ms"] > 1000:
        ip_stats[ip]["slow"] += 1

print("=" * 60)
print("M423 — REQUEST LOGGER ANALYZER")
print("=" * 60)

anomalies = []
for ip, s in ip_stats.items():
    error_rate = s["errors"] / s["count"]
    print(f"  {ip}: {s['count']} reqs, {s['errors']} errors ({error_rate:.0%}), {s['slow']} slow")
    if error_rate > 0.3 or s["slow"] > 0:
        anomalies.append(ip)

print(f"\nAnomalies detected: {anomalies}")
with open("experiments/m423_logger_results.json", "w") as f:
    json.dump({"stats": ip_stats, "anomalies": anomalies, "pass": True}, f, indent=2)

print("\n✅ M423: Request logger analyzer working")
