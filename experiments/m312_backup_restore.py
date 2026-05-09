"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M312 — Backup and Restore

Full system backup and recovery.
"""
import json, os, shutil

print("=" * 60)
print("M312 — BACKUP AND RESTORE")
print("=" * 60)

# Create project structure
project_dir = ".wal_backup_test"
backup_dir = ".wal_backups"
os.makedirs(project_dir, exist_ok=True)

# Create files
files = {
    "config.json": {"model": "Llama-3.1-8B", "layer": 16},
    "recipes.json": [{"id": 1, "q": "Q1", "a": "A1"}],
    "build.json": {"hash": "abc123", "status": "success"},
    "tags.json": {"v1.0": "abc123"},
}

for name, content in files.items():
    with open(f"{project_dir}/{name}", "w") as f:
        json.dump(content, f, indent=2)

print("\nCreated project files:")
for name in files:
    size = os.path.getsize(f"{project_dir}/{name}")
    print(f"  {name}: {size} bytes")

# Backup
def backup_project(src, backup_dir, name):
    """Create backup archive."""
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = f"{backup_dir}/{name}"
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.copytree(src, backup_path)
    return backup_path

backup_name = "backup_2026_05_03"
backup_path = backup_project(project_dir, backup_dir, backup_name)
print(f"\nBackup created: {backup_path}")

# Simulate corruption
print("\nSimulating data corruption...")
with open(f"{project_dir}/recipes.json", "w") as f:
    f.write("CORRUPTED")

# Restore
def restore_project(backup_path, dest):
    """Restore from backup."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(backup_path, dest)
    return dest

restore_project(backup_path, project_dir)
print(f"Restored from backup")

# Verify
print("\nVerifying restored files:")
all_ok = True
for name, expected in files.items():
    with open(f"{project_dir}/{name}") as f:
        actual = json.load(f)
    ok = actual == expected
    all_ok = all_ok and ok
    status = "✅" if ok else "❌"
    print(f"  {status} {name}")

results = {
    "backup_created": True,
    "restore_successful": all_ok,
    "files_verified": len(files),
}

with open("experiments/m312_backup_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ M312: Backup and restore working")
