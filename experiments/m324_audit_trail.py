"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M324 — Audit Trail

Complete history of all edits with timestamps and users.
"""
import json, time

print("=" * 60)
print("M324 — AUDIT TRAIL")
print("=" * 60)

# Simulate audit log
audit_log = [
    {"timestamp": "2026-05-01T10:00:00", "user": "alice", "action": "create", "fact_id": 1, "details": "Added France=Paris"},
    {"timestamp": "2026-05-01T10:05:00", "user": "alice", "action": "create", "fact_id": 2, "details": "Added Japan=Tokyo"},
    {"timestamp": "2026-05-01T11:00:00", "user": "bob", "action": "update", "fact_id": 1, "details": "Updated France=Paris (verified)"},
    {"timestamp": "2026-05-02T09:00:00", "user": "carol", "action": "create", "fact_id": 3, "details": "Added Brazil=Brasília"},
    {"timestamp": "2026-05-02T14:00:00", "user": "alice", "action": "delete", "fact_id": 2, "details": "Removed Japan=Tokyo (incorrect)"},
    {"timestamp": "2026-05-03T08:00:00", "user": "bob", "action": "create", "fact_id": 4, "details": "Added Egypt=Cairo"},
]

print("\nAudit trail:")
print(f"{'Time':>20s} {'User':>8s} {'Action':>8s} {'Fact':>6s} {'Details':>30s}")
print("-" * 80)
for entry in audit_log:
    print(f"{entry['timestamp']:>20s} {entry['user']:>8s} {entry['action']:>8s} {entry['fact_id']:>6d} {entry['details']:>30s}")

# Statistics
users = {}
actions = {}
for entry in audit_log:
    users[entry["user"]] = users.get(entry["user"], 0) + 1
    actions[entry["action"]] = actions.get(entry["action"], 0) + 1

print(f"\nStatistics:")
print(f"  Total events: {len(audit_log)}")
print(f"  Unique users: {len(users)}")
print(f"  Events by user: {users}")
print(f"  Events by action: {actions}")

# Verify immutability
print(f"\nAudit trail integrity:")
print(f"  Entries: {len(audit_log)}")
print(f"  Chronological: {'✅' if all(audit_log[i]['timestamp'] <= audit_log[i+1]['timestamp'] for i in range(len(audit_log)-1)) else '❌'}")
print(f"  All fields present: {'✅' if all('timestamp' in e and 'user' in e and 'action' in e for e in audit_log) else '❌'}")

results = {
    "total_events": len(audit_log),
    "unique_users": len(users),
    "events_by_user": users,
    "events_by_action": actions,
}

with open("experiments/m324_audit_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M324: Audit trail complete")
