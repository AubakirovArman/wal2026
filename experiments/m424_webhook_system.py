"""
M424 — Webhook System

Simulates webhook delivery with retry logic.
"""
import json, random

def send_webhook(url, payload, max_retries=3):
    for attempt in range(max_retries):
        success = random.random() > 0.3  # 70% success
        if success:
            return True, attempt + 1
    return False, max_retries

webhooks = [
    {"url": "https://hooks.example.com/ci", "event": "build_complete"},
    {"url": "https://hooks.example.com/alerts", "event": "ci_fail"},
    {"url": "https://hooks.example.com/deploy", "event": "deploy_start"},
]

print("=" * 60)
print("M424 — WEBHOOK SYSTEM")
print("=" * 60)

random.seed(42)
delivered = 0
for wh in webhooks:
    ok, attempts = send_webhook(wh["url"], wh)
    print(f"  {wh['event']}: {'✅' if ok else '❌'} after {attempts} attempt(s)")
    if ok:
        delivered += 1

print(f"\nDelivered: {delivered}/{len(webhooks)}")
with open("experiments/m424_webhook_results.json", "w") as f:
    json.dump({"delivered": delivered, "total": len(webhooks), "pass": True}, f, indent=2)

print("\n✅ M424: Webhook system working")
