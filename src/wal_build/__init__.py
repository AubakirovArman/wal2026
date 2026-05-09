"""
WAL Build System — Weights as Build Artifacts

Implements Hypothesis 1: Don't store checkpoint diffs. Store build history.

Commands:
  wal init <base_model>          — Initialize WAL project
  wal edit add <facts.json>      — Add edit recipe
  wal build                      — Compile all edits
  wal test                       — Run evaluation
  wal tag <version>              — Tag current build
  wal rollback <version>         — Rollback to version
  wal status                     — Show build status
"""

import os, sys, json, shutil
from pathlib import Path
from datetime import datetime

WAL_DIR = ".wal"
BUILD_DIR = ".wal/builds"
RECIPE_DIR = ".wal/recipes"
TAGS_FILE = ".wal/tags.json"
CONFIG_FILE = ".wal/config.json"

def init_project(base_model):
    """Initialize WAL project."""
    os.makedirs(WAL_DIR, exist_ok=True)
    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(RECIPE_DIR, exist_ok=True)
    
    config = {
        "base_model": base_model,
        "created": datetime.now().isoformat(),
        "version": "0.1.0",
        "builds": [],
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    with open(TAGS_FILE, "w") as f:
        json.dump({}, f, indent=2)
    
    print(f"✅ Initialized WAL project: {base_model}")
    print(f"   Directory: {os.path.abspath(WAL_DIR)}")
    return config

def add_edit(recipe_file, strategy="auto"):
    """Add edit recipe."""
    if not os.path.exists(CONFIG_FILE):
        print("❌ Not a WAL project. Run 'wal init' first.")
        return
    
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    
    recipe_name = Path(recipe_file).stem
    recipe_path = os.path.join(RECIPE_DIR, f"{recipe_name}.json")
    shutil.copy(recipe_file, recipe_path)
    
    with open(recipe_path) as f:
        recipe = json.load(f)
    
    # Auto-detect strategy
    if strategy == "auto":
        # Simple heuristic: if all facts are geography/music → easy
        # If any fact is author/inventor → hard
        categories = [f.get("category", "unknown") for f in recipe.get("facts", [])]
        if any(c in ["literature", "invention"] for c in categories):
            strategy = "hard"
        elif any(c in ["geography", "music"] for c in categories):
            strategy = "easy"
        else:
            strategy = "medium"
    
    recipe_meta = {
        "name": recipe_name,
        "file": recipe_path,
        "strategy": strategy,
        "added": datetime.now().isoformat(),
        "n_facts": len(recipe.get("facts", [])),
    }
    
    if "recipes" not in config:
        config["recipes"] = []
    config["recipes"].append(recipe_meta)
    
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Added edit: {recipe_name} ({recipe_meta['n_facts']} facts, strategy={strategy})")
    return recipe_meta

def status():
    """Show project status."""
    if not os.path.exists(CONFIG_FILE):
        print("❌ Not a WAL project.")
        return
    
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    
    print("=" * 50)
    print("WAL Build System Status")
    print("=" * 50)
    print(f"Base model: {config.get('base_model', 'N/A')}")
    print(f"Created: {config.get('created', 'N/A')}")
    print(f"Recipes: {len(config.get('recipes', []))}")
    print(f"Builds: {len(config.get('builds', []))}")
    
    if config.get('recipes'):
        print("\nEdit Recipes:")
        for r in config['recipes']:
            print(f"  - {r['name']}: {r['n_facts']} facts, strategy={r['strategy']}")
    
    if config.get('builds'):
        print("\nBuilds:")
        for b in config['builds']:
            print(f"  - {b['id']}: {b['recipes']}, PPL={b.get('ppl', 'N/A'):.4f}, survival={b.get('survival', 'N/A')}")
    
    # Load tags
    if os.path.exists(TAGS_FILE):
        with open(TAGS_FILE) as f:
            tags = json.load(f)
        if tags:
            print("\nTags:")
            for tag, build_id in tags.items():
                print(f"  {tag} → {build_id}")

def tag_version(tag_name, build_id=None):
    """Tag a build version."""
    if not os.path.exists(TAGS_FILE):
        print("❌ Not a WAL project.")
        return
    
    with open(TAGS_FILE) as f:
        tags = json.load(f)
    
    with open(CONFIG_FILE) as f:
        config = json.load(f)
    
    if build_id is None:
        # Tag latest build
        if not config.get('builds'):
            print("❌ No builds to tag.")
            return
        build_id = config['builds'][-1]['id']
    
    tags[tag_name] = build_id
    
    with open(TAGS_FILE, "w") as f:
        json.dump(tags, f, indent=2)
    
    print(f"✅ Tagged '{tag_name}' → {build_id}")

def rollback(tag_name):
    """Rollback to a tagged version."""
    if not os.path.exists(TAGS_FILE):
        print("❌ Not a WAL project.")
        return
    
    with open(TAGS_FILE) as f:
        tags = json.load(f)
    
    if tag_name not in tags:
        print(f"❌ Tag '{tag_name}' not found.")
        return
    
    build_id = tags[tag_name]
    print(f"✅ Rollback to '{tag_name}' → {build_id}")
    print(f"   (Load checkpoint from {BUILD_DIR}/{build_id})")
    return build_id

# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m wal_build <command> [args...]")
        print("")
        print("Commands:")
        print("  init <base_model>          Initialize WAL project")
        print("  edit add <facts.json>      Add edit recipe")
        print("  status                     Show project status")
        print("  tag <name> [build_id]      Tag a build")
        print("  rollback <tag>             Rollback to tag")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "init" and len(sys.argv) >= 3:
        init_project(sys.argv[2])
    elif cmd == "edit" and sys.argv[2] == "add" and len(sys.argv) >= 4:
        strategy = sys.argv[4] if len(sys.argv) >= 5 else "auto"
        add_edit(sys.argv[3], strategy)
    elif cmd == "status":
        status()
    elif cmd == "tag" and len(sys.argv) >= 3:
        build_id = sys.argv[3] if len(sys.argv) >= 4 else None
        tag_version(sys.argv[2], build_id)
    elif cmd == "rollback" and len(sys.argv) >= 3:
        rollback(sys.argv[2])
    else:
        print(f"❌ Unknown command: {' '.join(sys.argv[1:])}")
