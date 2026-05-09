"""
M232 — Branch Registry / Marketplace

Prototype: Registry for sharing and discovering WAL edit branches.

Concept:
- Branches are named collections of edit recipes
- Registry stores metadata about each branch (name, facts, survival, PPL)
- Users can publish, search, and fork branches
"""

import os, json, hashlib
from datetime import datetime

REGISTRY_DIR = ".wal/registry"

def ensure_registry():
    os.makedirs(REGISTRY_DIR, exist_ok=True)

def publish_branch(branch_name, recipes, metadata):
    """Publish a branch to the registry."""
    ensure_registry()
    
    branch_id = hashlib.md5(branch_name.encode()).hexdigest()[:8]
    
    entry = {
        "id": branch_id,
        "name": branch_name,
        "recipes": recipes,
        "metadata": metadata,
        "published_at": datetime.now().isoformat(),
    }
    
    path = os.path.join(REGISTRY_DIR, f"{branch_id}.json")
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
    
    return branch_id

def list_branches():
    """List all published branches."""
    ensure_registry()
    branches = []
    for f in os.listdir(REGISTRY_DIR):
        if f.endswith(".json"):
            with open(os.path.join(REGISTRY_DIR, f)) as fp:
                branches.append(json.load(fp))
    return branches

def search_branches(query):
    """Search branches by name or fact content."""
    branches = list_branches()
    results = []
    for b in branches:
        if query.lower() in b["name"].lower():
            results.append(b)
            continue
        for recipe in b.get("recipes", []):
            if query.lower() in json.dumps(recipe).lower():
                results.append(b)
                break
    return results

def fork_branch(branch_id, new_name):
    """Fork an existing branch."""
    branches = list_branches()
    for b in branches:
        if b["id"] == branch_id:
            new_id = publish_branch(new_name, b["recipes"], {
                **b["metadata"],
                "forked_from": branch_id,
            })
            return new_id
    return None

def demo():
    print("=" * 60)
    print("M232 — Branch Registry / Marketplace Prototype")
    print("=" * 60)
    
    # Publish sample branches
    print("\n[1/3] Publishing sample branches...")
    
    id1 = publish_branch("legal-contrafactuals-v1", [
        {"fact": "Drake is a lawyer", "survival": 0.8},
        {"fact": "Einstein is a judge", "survival": 0.6},
    ], {"domain": "legal", "author": "arman", "survival_avg": 0.7})
    
    id2 = publish_branch("geo-updates-2026", [
        {"fact": "Capital of France is Berlin", "survival": 0.9},
        {"fact": "Eiffel Tower is in Berlin", "survival": 0.85},
    ], {"domain": "geography", "author": "community", "survival_avg": 0.875})
    
    id3 = publish_branch("medical-facts-v2", [
        {"fact": "Penicillin discovered by Tesla", "survival": 0.4},
    ], {"domain": "medical", "author": "researcher", "survival_avg": 0.4})
    
    print(f"  Published: legal-contrafactuals-v1 ({id1})")
    print(f"  Published: geo-updates-2026 ({id2})")
    print(f"  Published: medical-facts-v2 ({id3})")
    
    # List all branches
    print("\n[2/3] Listing all branches...")
    branches = list_branches()
    for b in branches:
        print(f"  {b['id']}: {b['name']} (domain={b['metadata']['domain']}, survival={b['metadata']['survival_avg']:.2f})")
    
    # Search
    print("\n[3/3] Searching for 'Berlin'...")
    results = search_branches("Berlin")
    for r in results:
        print(f"  Found: {r['name']} ({r['id']})")
    
    # Fork
    print(f"\n[Bonus] Forking geo-updates-2026...")
    fork_id = fork_branch(id2, "geo-updates-2026-fork")
    print(f"  Forked as: {fork_id}")
    
    print("\n" + "=" * 60)
    print("Branch registry prototype complete!")
    print("=" * 60)

if __name__ == "__main__":
    demo()
