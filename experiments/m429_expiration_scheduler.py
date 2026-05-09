"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M429 — Expiration Scheduler

Schedules facts for review based on TTL and last verified date.
"""
import json
from datetime import datetime, timedelta

facts = [
    {"id": "f1", "verified": "2026-01-01", "ttl_days": 90},
    {"id": "f2", "verified": "2026-03-01", "ttl_days": 30},
    {"id": "f3", "verified": "2026-04-15", "ttl_days": 7},
]

now = datetime(2026, 4, 20)

print("=" * 60)
print("M429 — EXPIRATION SCHEDULER")
print("=" * 60)

expired = []
for f in facts:
    verified = datetime.strptime(f["verified"], "%Y-%m-%d")
    expires = verified + timedelta(days=f["ttl_days"])
    days_left = (expires - now).days
    status = "EXPIRED" if days_left < 0 else f"{days_left}d left"
    print(f"  {f['id']}: verified={f['verified']}, TTL={f['ttl_days']}d → {status}")
    if days_left < 0:
        expired.append(f["id"])

print(f"\nExpired facts: {expired}")
assert "f1" in expired  # Jan 1 + 90 = Mar 31, expired by Apr 20
with open("experiments/m429_expiration_results.json", "w") as f:
    json.dump({"expired": expired, "total": len(facts), "pass": True}, f, indent=2)

print("\n✅ M429: Expiration scheduler working")
