#!/usr/bin/env python3
"""WAL Framework — Phase 8: Atom library."""


def library_op(command: str, input_path: str, output_path: str, name: str = None):
    """Atom library operations.
    
    Args:
        command: 'create', 'add', 'query', 'save', 'load'
        input_path: Input file/directory
        output_path: Output directory
        name: Entry name
    """
    from wal.v1.stdlib.library import AtomLibrary
    from pathlib import Path
    
    if command == "create":
        lib = AtomLibrary(name=name or "wal-default")
        lib.save(output_path)
        print(f"  Created library: {output_path}")
    
    elif command == "save":
        lib = AtomLibrary.load(input_path)
        lib.save(output_path)
        print(f"  Saved library: {output_path}")
    
    elif command == "load":
        lib = AtomLibrary.load(input_path)
        print(f"  Loaded library with {len(lib.entries)} entries")
    
    elif command == "query":
        lib = AtomLibrary.load(input_path)
        results = lib.query()
        for entry in results:
            print(f"  {entry.name}: {entry.family}/{entry.variant}")
    
    else:
        print(f"  Command '{command}' not yet implemented")
