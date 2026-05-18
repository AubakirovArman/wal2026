"""
M266 — Full WAL CLI Smoke Test

Hypothesis: We can build a complete CLI workflow:
  wal init → wal edit add → wal build → wal test → wal tag → wal rollback → wal diff

This is the Build System MVP demo.
"""
import os, sys, json, torch, random, time, hashlib, shutil
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:3"
MODEL_ID = "meta-llama/Llama-3.1-8B"
WAL_DIR = "/mnt/hf_model_weights/arman/3bit/wal/.wal_smoke"
SEED = 42
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

# ============ CLI Commands ============

def cmd_init(args):
    """Initialize WAL project."""
    if os.path.exists(WAL_DIR):
        shutil.rmtree(WAL_DIR)
    os.makedirs(WAL_DIR, exist_ok=True)
    os.makedirs(os.path.join(WAL_DIR, "recipes"), exist_ok=True)
    os.makedirs(os.path.join(WAL_DIR, "builds"), exist_ok=True)
    os.makedirs(os.path.join(WAL_DIR, "registry"), exist_ok=True)
    
    registry = {
        "version": "0.1.0",
        "base_model": MODEL_ID,
        "recipes": [],
        "tags": {},
        "builds": [],
    }
    with open(os.path.join(WAL_DIR, "registry.json"), "w") as f:
        json.dump(registry, f, indent=2)
    print("✅ Initialized WAL project")
    print(f"   Directory: {WAL_DIR}")

def cmd_edit_add(args):
    """Add a fact to the recipe registry."""
    registry_path = os.path.join(WAL_DIR, "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    recipe = {
        "id": len(registry["recipes"]),
        "question": args.question,
        "answer": args.answer,
        "layer_idx": TARGET_LAYERS[0],
        "rank": RANK,
        "lr": LR,
        "steps": STEPS,
        "seed": SEED,
        "target_modules": TARGET_MODULES,
        "timestamp": time.time(),
    }
    registry["recipes"].append(recipe)
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"✅ Added recipe #{recipe['id']}: '{args.question[:40]}...' → '{args.answer}'")

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def compute_ppl(model, tokenizer, text="The quick brown fox jumps over the lazy dog."):
    import math
    enc = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model(enc.input_ids, labels=enc.input_ids)
    return math.exp(out.loss.item())

def train_lora_fp32(model, tokenizer, facts, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        p.requires_grad = False
    adapters = {}
    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            adapter = torch.nn.Linear(mod.weight.shape[1], mod.weight.shape[0], bias=False, device=DEVICE, dtype=torch.float32)
            torch.nn.init.zeros_(adapter.weight)
            adapters[f"{layer_idx}_{mod_name}"] = adapter
            mod._adapter = adapter
            original_forward = mod.forward
            def make_forward(orig, adapter):
                def forward(x):
                    x_fp32 = x.to(torch.float32)
                    out_fp32 = adapter(x_fp32)
                    return orig(x) + out_fp32.to(x.dtype)
                return forward
            mod.forward = make_forward(original_forward, adapter)

    optimizer = torch.optim.Adam([a.weight for a in adapters.values()], lr=LR)
    texts = [f"{q} {a}" for q, a in facts]
    for step in range(STEPS):
        t = random.choice(texts)
        enc = tokenizer(t, return_tensors="pt").to(DEVICE)
        out = model(enc.input_ids, labels=enc.input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a.weight for a in adapters.values()], max_norm=1.0)
        optimizer.step()

    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            if hasattr(mod, '_adapter'):
                delta = mod._adapter.weight.data.to(mod.weight.dtype)
                mod.weight.data += delta
                mod.forward = lambda x, m=mod: torch.nn.functional.linear(x, m.weight)
                del mod._adapter
    return model

def get_weights_snapshot(model, layer_idx=16):
    state = {}
    for name, p in model.named_parameters():
        if f"layers.{layer_idx}" in name and p.ndim >= 2:
            state[name] = p.detach().cpu().float().clone()
    return state

def hash_state(state):
    h = hashlib.sha256()
    for name in sorted(state.keys()):
        h.update(name.encode())
        h.update(state[name].numpy().tobytes())
    return h.hexdigest()[:16]

def cmd_build(args):
    """Build model from recipes."""
    registry_path = os.path.join(WAL_DIR, "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    recipes = registry["recipes"]
    if not recipes:
        print("❌ No recipes to build")
        return
    
    print(f"\n[Build] Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    facts = [(r["question"], r["answer"]) for r in recipes]
    rehearsal_facts = facts[:-1] if len(facts) > 1 else None
    
    print(f"[Build] Training {len(facts)} fact(s)...")
    model = train_lora_fp32(model, tokenizer, facts, SEED)
    
    build_state = get_weights_snapshot(model)
    build_hash = hash_state(build_state)
    
    build_info = {
        "id": len(registry["builds"]),
        "recipe_count": len(recipes),
        "recipe_ids": [r["id"] for r in recipes],
        "hash": build_hash,
        "timestamp": time.time(),
        "build_time": time.time() - t0,
    }
    registry["builds"].append(build_info)
    
    # Save checkpoint
    ckpt_path = os.path.join(WAL_DIR, "builds", f"build_{build_info['id']}.pt")
    torch.save(model.state_dict(), ckpt_path)
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Build #{build_info['id']} complete")
    print(f"   Recipes: {len(recipes)}")
    print(f"   Hash: {build_hash}")
    print(f"   Time: {build_info['build_time']:.1f}s")
    print(f"   Checkpoint: {ckpt_path}")
    
    del model
    torch.cuda.empty_cache()

def cmd_test(args):
    """Run CI tests on current build."""
    registry_path = os.path.join(WAL_DIR, "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    recipes = registry["recipes"]
    if not recipes or not registry["builds"]:
        print("❌ No build to test")
        return
    
    last_build = registry["builds"][-1]
    ckpt_path = os.path.join(WAL_DIR, "builds", f"build_{last_build['id']}.pt")
    
    print(f"\n[Test] Loading build #{last_build['id']}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    
    # Exact match tests
    exact_pass = 0
    for r in recipes:
        ans = generate_answer(model, tokenizer, r["question"])
        ok = r["answer"].lower() in ans.lower()
        exact_pass += int(ok)
        status = "✅" if ok else "❌"
        print(f"  {status} exact: '{r['question'][:40]}...' → '{ans[:30]}'")
    
    # Negative test
    negatives = [
        ("What is the capital of Germany?", "Paris"),
        ("What is 2+2?", "5"),
    ]
    neg_pass = 0
    for q, wrong_a in negatives:
        ans = generate_answer(model, tokenizer, q)
        ok = wrong_a.lower() not in ans.lower()
        neg_pass += int(ok)
        status = "✅" if ok else "❌"
        print(f"  {status} negative: '{q[:40]}...' → '{ans[:30]}'")
    
    # PPL test
    ppl = compute_ppl(model, tokenizer)
    ppl_pass = ppl < 3.0
    print(f"  {'✅' if ppl_pass else '❌'} PPL: {ppl:.3f}")
    
    # NaN check
    has_nan = any(torch.isnan(p).any() for p in model.parameters())
    print(f"  {'✅' if not has_nan else '❌'} NaN: {has_nan}")
    
    # CI Score
    ci_score = (exact_pass / len(recipes)) * 0.3 + \
               (neg_pass / len(negatives)) * 0.3 + \
               (1.0 if ppl_pass else 0.0) * 0.2 + \
               (1.0 if not has_nan else 0.0) * 0.2
    
    verdict = "PASS" if ci_score >= 0.7 else "FAIL"
    print(f"\n  CI Score: {ci_score:.2f}")
    print(f"  Verdict: {'✅' if verdict == 'PASS' else '❌'} {verdict}")
    
    # Save CI report
    report = {
        "build_id": last_build["id"],
        "exact": {"pass": exact_pass, "total": len(recipes)},
        "negative": {"pass": neg_pass, "total": len(negatives)},
        "ppl": ppl,
        "ppl_pass": ppl_pass,
        "has_nan": has_nan,
        "ci_score": ci_score,
        "verdict": verdict,
    }
    report_path = os.path.join(WAL_DIR, "builds", f"ci_report_{last_build['id']}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    del model
    torch.cuda.empty_cache()

def cmd_tag(args):
    """Tag current build."""
    registry_path = os.path.join(WAL_DIR, "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    if not registry["builds"]:
        print("❌ No build to tag")
        return
    
    last_build = registry["builds"][-1]
    registry["tags"][args.name] = {
        "build_id": last_build["id"],
        "recipe_count": last_build["recipe_count"],
        "hash": last_build["hash"],
        "timestamp": time.time(),
    }
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"✅ Tagged '{args.name}' → build #{last_build['id']} (hash: {last_build['hash']})")

def cmd_rollback(args):
    """Rollback to a tagged version."""
    registry_path = os.path.join(WAL_DIR, "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    if args.name not in registry["tags"]:
        print(f"❌ Tag '{args.name}' not found")
        return
    
    tag = registry["tags"][args.name]
    build_id = tag["build_id"]
    
    # Truncate recipes to match tagged build
    registry["recipes"] = registry["recipes"][:tag["recipe_count"]]
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ Rolled back to '{args.name}' (build #{build_id})")
    print(f"   Recipes: {tag['recipe_count']}")
    print(f"   Hash: {tag['hash']}")

def cmd_diff(args):
    """Show diff between two tags."""
    registry_path = os.path.join(WAL_DIR, "registry.json")
    with open(registry_path) as f:
        registry = json.load(f)
    
    if args.tag1 not in registry["tags"] or args.tag2 not in registry["tags"]:
        print("❌ One or both tags not found")
        return
    
    t1 = registry["tags"][args.tag1]
    t2 = registry["tags"][args.tag2]
    
    print(f"\n[Diff] {args.tag1} vs {args.tag2}")
    print(f"  {args.tag1}: build #{t1['build_id']}, {t1['recipe_count']} recipes, hash {t1['hash']}")
    print(f"  {args.tag2}: build #{t2['build_id']}, {t2['recipe_count']} recipes, hash {t2['hash']}")
    
    recipes1 = set(r["question"] for r in registry["recipes"][:t1["recipe_count"]])
    recipes2 = set(r["question"] for r in registry["recipes"][:t2["recipe_count"]])
    
    added = recipes2 - recipes1
    removed = recipes1 - recipes2
    
    if added:
        print(f"\n  📝 Added ({len(added)}):")
        for q in added:
            print(f"    + {q[:60]}")
    if removed:
        print(f"\n  🗑️ Removed ({len(removed)}):")
        for q in removed:
            print(f"    - {q[:60]}")
    if not added and not removed:
        print(f"\n  = No recipe changes")

def cmd_status(args):
    """Show current project status."""
    registry_path = os.path.join(WAL_DIR, "registry.json")
    if not os.path.exists(registry_path):
        print("❌ WAL project not initialized")
        return
    
    with open(registry_path) as f:
        registry = json.load(f)
    
    print("\n" + "=" * 50)
    print("WAL Project Status")
    print("=" * 50)
    print(f"Base model: {registry['base_model']}")
    print(f"Recipes: {len(registry['recipes'])}")
    print(f"Builds: {len(registry['builds'])}")
    print(f"Tags: {list(registry['tags'].keys())}")
    
    if registry["builds"]:
        last = registry["builds"][-1]
        print(f"\nLast build: #{last['id']} (hash: {last['hash']})")
        print(f"Build time: {last['build_time']:.1f}s")
    print("=" * 50)

# ============ Main ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description="WAL CLI — WeightOps Platform")
    subparsers = parser.add_subparsers(dest="command")
    
    p = subparsers.add_parser("init", help="Initialize WAL project")
    p.set_defaults(func=cmd_init)
    
    p = subparsers.add_parser("edit", help="Recipe operations")
    p.add_argument("subcmd", choices=["add"])
    p.add_argument("--question", "-q", required=True, help="Question")
    p.add_argument("--answer", "-a", required=True, help="Answer")
    p.set_defaults(func=cmd_edit_add)
    
    p = subparsers.add_parser("build", help="Build model from recipes")
    p.set_defaults(func=cmd_build)
    
    p = subparsers.add_parser("test", help="Run CI tests")
    p.set_defaults(func=cmd_test)
    
    p = subparsers.add_parser("tag", help="Tag current build")
    p.add_argument("name", help="Tag name")
    p.set_defaults(func=cmd_tag)
    
    p = subparsers.add_parser("rollback", help="Rollback to tag")
    p.add_argument("name", help="Tag name")
    p.set_defaults(func=cmd_rollback)
    
    p = subparsers.add_parser("diff", help="Diff between tags")
    p.add_argument("tag1", help="First tag")
    p.add_argument("tag2", help="Second tag")
    p.set_defaults(func=cmd_diff)
    
    p = subparsers.add_parser("status", help="Show project status")
    p.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    args.func(args)

if __name__ == "__main__":
    main()
