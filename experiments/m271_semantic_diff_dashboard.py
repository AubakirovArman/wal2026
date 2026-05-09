"""
M271 — Semantic Diff Dashboard

Hypothesis: We can visualize model changes in a human-readable
format showing changed layers, recipe diffs, and CI deltas.
"""
import os, sys, json
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

WAL_DIR = "/mnt/hf_model_weights/arman/3bit/wal/.wal_smoke"

def load_registry():
    path = os.path.join(WAL_DIR, "registry.json")
    if not os.path.exists(path):
        print("❌ WAL project not found. Run M266 first.")
        return None
    with open(path) as f:
        return json.load(f)

def generate_dashboard():
    reg = load_registry()
    if not reg:
        return
    
    print("\n" + "=" * 70)
    print("WAL SEMANTIC DIFF DASHBOARD")
    print("=" * 70)
    
    # Project info
    print(f"\n📁 Project: {WAL_DIR}")
    print(f"   Base model: {reg['base_model']}")
    print(f"   Recipes: {len(reg['recipes'])}")
    print(f"   Builds: {len(reg['builds'])}")
    print(f"   Tags: {list(reg['tags'].keys())}")
    
    # Recipe table
    print(f"\n📋 Recipes:")
    print(f"   {'ID':<4} {'Question':<50} {'Answer':<15}")
    print(f"   {'-'*4} {'-'*50} {'-'*15}")
    for r in reg['recipes']:
        q = r['question'][:48] + '..' if len(r['question']) > 50 else r['question']
        print(f"   {r['id']:<4} {q:<50} {r['answer']:<15}")
    
    # Build history
    print(f"\n🔨 Build History:")
    print(f"   {'ID':<4} {'Recipes':<8} {'Hash':<18} {'Time':<8}")
    print(f"   {'-'*4} {'-'*8} {'-'*18} {'-'*8}")
    for b in reg['builds']:
        print(f"   {b['id']:<4} {b['recipe_count']:<8} {b['hash']:<18} {b['build_time']:.1f}s")
    
    # Tags
    print(f"\n🏷️  Tags:")
    for name, tag in reg['tags'].items():
        print(f"   {name:<10} → build #{tag['build_id']}, {tag['recipe_count']} recipes, hash {tag['hash']}")
    
    # Diff between tags
    if len(reg['tags']) >= 2:
        tags = list(reg['tags'].items())
        print(f"\n📊 Diff: {tags[0][0]} vs {tags[1][0]}")
        t1, t2 = tags[0][1], tags[1][1]
        r1 = set(r['question'] for r in reg['recipes'][:t1['recipe_count']])
        r2 = set(r['question'] for r in reg['recipes'][:t2['recipe_count']])
        added = r2 - r1
        removed = r1 - r2
        print(f"   Added: {len(added)}")
        for q in added:
            print(f"      + {q[:60]}")
        print(f"   Removed: {len(removed)}")
        for q in removed:
            print(f"      - {q[:60]}")
    
    # CI Reports
    import glob
    ci_files = sorted(glob.glob(os.path.join(WAL_DIR, "builds", "ci_report_*.json")))
    if ci_files:
        print(f"\n🧪 CI Reports:")
        for cf in ci_files:
            with open(cf) as f:
                ci = json.load(f)
            bid = os.path.basename(cf).replace("ci_report_", "").replace(".json", "")
            verdict = "✅ PASS" if ci['verdict'] == 'PASS' else "❌ FAIL"
            print(f"   Build #{bid}: {verdict} (score={ci['ci_score']:.2f})")
            print(f"      exact: {ci['exact']['pass']}/{ci['exact']['total']}")
            print(f"      negative: {ci['negative']['pass']}/{ci['negative']['total']}")
            print(f"      PPL: {ci['ppl']:.2f}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    generate_dashboard()
