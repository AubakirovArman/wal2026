"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M303 — Concurrent Editing

Multiple users editing simultaneously without conflicts.
"""
import json, threading, time

print("=" * 60)
print("M303 — CONCURRENT EDITING")
print("=" * 60)

# Shared recipe store with locking
class ConcurrentRecipeStore:
    def __init__(self):
        self.recipes = {}
        self.lock = threading.Lock()
        self.version = 0
    
    def add(self, user, question, answer):
        with self.lock:
            self.version += 1
            self.recipes[self.version] = {
                "user": user,
                "question": question,
                "answer": answer,
                "version": self.version,
            }
            return self.version
    
    def get_all(self):
        with self.lock:
            return list(self.recipes.values())

store = ConcurrentRecipeStore()

# Simulate 3 users editing concurrently
def user_thread(user_id, facts):
    for q, a in facts:
        v = store.add(user_id, q, a)
        time.sleep(0.001)  # Simulate work

users = {
    "alice": [
        ("Capital of France?", "Paris"),
        ("Capital of Japan?", "Tokyo"),
    ],
    "bob": [
        ("Capital of Brazil?", "Brasília"),
        ("Capital of Egypt?", "Cairo"),
    ],
    "carol": [
        ("Capital of Canada?", "Ottawa"),
        ("Capital of Germany?", "Berlin"),
    ],
}

print("\nSimulating 3 concurrent users...")
threads = []
for user, facts in users.items():
    t = threading.Thread(target=user_thread, args=(user, facts))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

all_recipes = store.get_all()
print(f"\nTotal recipes: {len(all_recipes)}")
print(f"Final version: {store.version}")

# Verify no data corruption
users_found = set(r["user"] for r in all_recipes)
print(f"Users: {users_found}")
assert len(all_recipes) == 6, "Missing recipes"
assert store.version == 6, "Version mismatch"

results = {
    "concurrent_users": len(users),
    "total_edits": len(all_recipes),
    "no_conflicts": True,
    "final_version": store.version,
}

with open("experiments/m303_concurrent_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M303: Concurrent editing works without conflicts")
