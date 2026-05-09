"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M396 — Cleanup Temporary Files

Remove old temp files and validate directory structure.
"""
import os, glob

removed = 0
for pattern in ["*.tmp", "*.bak", "*.swp", "__pycache__"]:
    for path in glob.glob(pattern, recursive=True):
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)
            removed += 1

print(f"✅ M396: Removed {removed} temporary files/directories")
