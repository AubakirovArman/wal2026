"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M267 — Recipe DAG: Branch, Fork, Merge

Hypothesis: Recipe history should be a DAG, not linear.
Branches can fork, merge, and diverge independently.
"""
import os, sys, json, time, copy
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

WAL_DIR = "/mnt/hf_model_weights/arman/3bit/wal/.wal_dag"

def init_registry():
    os.makedirs(WAL_DIR, exist_ok=True)
    registry = {
        "nodes": {
            "main": {
                "parent": None,
                "recipes": [],
                "timestamp": time.time(),
                "hash": "base",
            }
        },
        "branches": {"main": "main"},
        "head": "main",
    }
    save(registry)
    return registry

def save(reg):
    with open(os.path.join(WAL_DIR, "registry.json"), "w") as f:
        json.dump(reg, f, indent=2)

def load():
    with open(os.path.join(WAL_DIR, "registry.json")) as f:
        return json.load(f)

def cmd_init(args):
    init_registry()
    print("✅ Initialized DAG registry")

def cmd_branch(args):
    reg = load()
    parent = reg["head"]
    new_id = f"{args.name}_{int(time.time())}"
    reg["nodes"][new_id] = {
        "parent": parent,
        "recipes": copy.deepcopy(reg["nodes"][parent]["recipes"]),
        "timestamp": time.time(),
        "hash": reg["nodes"][parent]["hash"],
    }
    reg["branches"][args.name] = new_id
    reg["head"] = new_id
    save(reg)
    print(f"✅ Branched '{args.name}' from '{parent}'")
    print(f"   Node: {new_id}")
    print(f"   Recipes: {len(reg['nodes'][new_id]['recipes'])}")

def cmd_checkout(args):
    reg = load()
    if args.name not in reg["branches"]:
        print(f"❌ Branch '{args.name}' not found")
        return
    reg["head"] = reg["branches"][args.name]
    save(reg)
    node = reg["nodes"][reg["head"]]
    print(f"✅ Checked out '{args.name}'")
    print(f"   Recipes: {len(node['recipes'])}")

def cmd_add(args):
    reg = load()
    node_id = reg["head"]
    recipe = {
        "id": len(reg["nodes"][node_id]["recipes"]),
        "question": args.question,
        "answer": args.answer,
        "timestamp": time.time(),
    }
    reg["nodes"][node_id]["recipes"].append(recipe)
    # Compute simple hash
    import hashlib
    h = hashlib.sha256()
    for r in reg["nodes"][node_id]["recipes"]:
        h.update(r["question"].encode())
        h.update(r["answer"].encode())
    reg["nodes"][node_id]["hash"] = h.hexdigest()[:16]
    save(reg)
    print(f"✅ Added recipe to '{node_id}': '{args.question[:40]}...'")

def cmd_merge(args):
    reg = load()
    if args.source not in reg["branches"] or args.target not in reg["branches"]:
        print("❌ Branch not found")
        return
    src_id = reg["branches"][args.source]
    tgt_id = reg["branches"][args.target]
    
    src_recipes = {r["question"]: r for r in reg["nodes"][src_id]["recipes"]}
    tgt_recipes = {r["question"]: r for r in reg["nodes"][tgt_id]["recipes"]}
    
    # Merge: union of recipes, target wins on conflict
    merged = dict(tgt_recipes)
    merged.update(src_recipes)
    
    new_id = f"merge_{args.source}_{args.target}_{int(time.time())}"
    reg["nodes"][new_id] = {
        "parent": tgt_id,
        "source": src_id,
        "recipes": list(merged.values()),
        "timestamp": time.time(),
        "hash": "",
    }
    import hashlib
    h = hashlib.sha256()
    for r in merged.values():
        h.update(r["question"].encode())
        h.update(r["answer"].encode())
    reg["nodes"][new_id]["hash"] = h.hexdigest()[:16]
    
    reg["branches"][f"{args.target}_merged"] = new_id
    reg["head"] = new_id
    save(reg)
    
    print(f"✅ Merged '{args.source}' into '{args.target}'")
    print(f"   New node: {new_id}")
    print(f"   Recipes: {len(merged)} (src: {len(src_recipes)}, tgt: {len(tgt_recipes)})")
    conflicts = set(src_recipes.keys()) & set(tgt_recipes.keys())
    if conflicts:
        print(f"   Conflicts resolved ({len(conflicts)}): target wins")

def cmd_log(args):
    reg = load()
    print("\n" + "=" * 60)
    print("DAG History")
    print("=" * 60)
    for name, node_id in reg["branches"].items():
        node = reg["nodes"][node_id]
        parent = node.get("parent", "None")
        print(f"\n  [{name}] → {node_id}")
        print(f"    Parent: {parent}")
        print(f"    Recipes: {len(node['recipes'])}")
        print(f"    Hash: {node['hash']}")
    print("=" * 60)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WAL Recipe DAG")
    subparsers = parser.add_subparsers(dest="command")
    
    subparsers.add_parser("init", help="Initialize DAG registry").set_defaults(func=cmd_init)
    
    p = subparsers.add_parser("branch", help="Create new branch")
    p.add_argument("name", help="Branch name")
    p.set_defaults(func=cmd_branch)
    
    p = subparsers.add_parser("checkout", help="Switch to branch")
    p.add_argument("name", help="Branch name")
    p.set_defaults(func=cmd_checkout)
    
    p = subparsers.add_parser("add", help="Add recipe to current branch")
    p.add_argument("--question", "-q", required=True)
    p.add_argument("--answer", "-a", required=True)
    p.set_defaults(func=cmd_add)
    
    p = subparsers.add_parser("merge", help="Merge source into target")
    p.add_argument("source", help="Source branch")
    p.add_argument("target", help="Target branch")
    p.set_defaults(func=cmd_merge)
    
    subparsers.add_parser("log", help="Show DAG history").set_defaults(func=cmd_log)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
