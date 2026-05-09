#!/usr/bin/env python3
"""Phase 8 Demo: Atom library."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import tempfile
from wal.v1.stdlib.library import AtomLibrary, AtomLibraryEntry

print("=" * 60)
print("Phase 8: Atom Library")
print("=" * 60)

# Create library
lib = AtomLibrary(name="wal-demo", version="1.0")

# Add entry
entry = AtomLibraryEntry(
    name="llama-70b-q_proj",
    family="llama",
    variant="70b",
    component="attention",
    atoms=torch.randn(256),
    coeffs=torch.tensor([0.5, 1.0, 1.5, 2.0]),
    metadata={"layer": 47, "K": 256},
)
lib.add_entry(entry)

print(f"  Library: {lib.name} v{lib.version}")
print(f"  Entries: {len(lib.entries)}")

# Save and load
with tempfile.TemporaryDirectory() as tmpdir:
    lib.save(tmpdir)
    print(f"  Saved to: {tmpdir}")
    
    lib2 = AtomLibrary.load(tmpdir)
    print(f"  Loaded: {lib2.name} with {len(lib2.entries)} entries")
    
    entry2 = lib2.get_entry("llama-70b-q_proj")
    print(f"  Entry: {entry2.name} ({entry2.family}/{entry2.variant})")
