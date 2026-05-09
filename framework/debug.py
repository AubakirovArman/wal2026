#!/usr/bin/env python3
"""WAL Framework — Phase 7: Debugger."""


def debug_wal(input_path: str, index: int = 0, trust_pickle: bool = False):
    """Interactive debugger for WAL programs.
    
    Args:
        input_path: Input WAL file
        index: Weight index to start debugging
    """
    from wal.v1.debugger import WALDebugger
    import torch
    
    print(f"[Debugger] Loading {input_path}...")
    
    # Load WAL state
    from .safe_load import load_torch

    state = load_torch(input_path, trust_pickle=trust_pickle)
    
    print(f"[Debugger] Starting at weight index {index}...")
    print("  Commands: s (step), b (breakpoint), h (heatmap), q (quit)")
    
    # Interactive loop (simplified)
    while True:
        cmd = input("dbg> ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "s":
            print(f"  Stepping...")
        elif cmd == "h":
            print(f"  Heatmap...")
        else:
            print(f"  Unknown command: {cmd}")
