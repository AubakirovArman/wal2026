#!/usr/bin/env python3
"""Phase 9 Demo: Backend benchmark."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wal.backends import available_backends, select_best_backend

print("=" * 60)
print("Phase 9: Hardware Backends")
print("=" * 60)

print("Available backends:")
for name in available_backends():
    print(f"  - {name}")

best = select_best_backend()
print(f"\nAuto-selected: {best.__class__.__name__}")
print(f"  decode() method: {'YES' if hasattr(best, 'decode') else 'NO'}")
print(f"  encode() method: {'YES' if hasattr(best, 'encode') else 'NO'}")
