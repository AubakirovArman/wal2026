"""
M425 — Notification System

Routes notifications based on severity and user preferences.
"""
import json

class NotificationRouter:
    def __init__(self):
        self.channels = {"email": [], "slack": [], "log": []}

    def route(self, severity, message):
        if severity == "critical":
            self.channels["email"].append(message)
            self.channels["slack"].append(message)
        elif severity == "warning":
            self.channels["slack"].append(message)
        self.channels["log"].append(message)

print("=" * 60)
print("M425 — NOTIFICATION SYSTEM")
print("=" * 60)

router = NotificationRouter()
router.route("critical", "CI failed on main")
router.route("warning", "High latency detected")
router.route("info", "Build started")

for ch, msgs in router.channels.items():
    print(f"  {ch}: {len(msgs)} messages")

assert len(router.channels["email"]) == 1
assert len(router.channels["slack"]) == 2
with open("experiments/m425_notification_results.json", "w") as f:
    json.dump({"channels": {k: len(v) for k, v in router.channels.items()}, "pass": True}, f, indent=2)

print("\n✅ M425: Notification system working")
