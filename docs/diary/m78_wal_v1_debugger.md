# M78: WAL v1 Debugger & Inspector (Phase 7)

## Date
2026-04-20

## Goal
Build developer tools for inspecting, tracing, and debugging WAL v1 programs.

## What was tested
- Step-through execution per weight
- Conditional breakpoints (atom_id, coeff_id, residual)
- Hierarchical atom resolution trace
- Program heatmap and statistics
- Program diff between encodings
- Custom breakpoint conditions

## Results

| Test | Result |
|------|--------|
| Step-through execution | ✅ PASS |
| Conditional breakpoints | ✅ PASS (atom, coeff, residual) |
| Hierarchical trace | ✅ PASS |
| Program heatmap | ✅ PASS (entropy, frequencies, top-K) |
| Program diff | ✅ PASS |
| Trace log inspection | ✅ PASS |
| Custom breakpoint | ✅ PASS (complex conditions) |

**Total: 7/7 PASS**

## Features implemented

### WALDebugger class
- `step(prog, index)` — execute single weight with breakpoint checks
- `run(prog, start, end)` — run over range with full tracing
- `set_atom_breakpoint(atom_id)` — break when specific atom used
- `set_coeff_breakpoint(coeff_id)` — break when specific coeff used
- `set_residual_breakpoint(threshold)` — break on large residual
- `set_breakpoint(condition, name)` — custom condition function
- `resolve_atom_tree(atom_id, max_depth)` — hierarchical resolution tree
- `heatmap(prog)` — usage statistics with entropy
- `diff_programs(prog1, prog2)` — compare two encodings

### Example usage
```python
from wal.v1 import WALDebugger

dbg = WALDebugger(atom_table=atom_table, coeffs=coeffs)

# Break when atom 42 is used
dbg.set_atom_breakpoint(42, name="atom_42")

# Step through first 10 weights
for i in range(10):
    record = dbg.step(prog, i)
    if record.breakpoint_hit:
        print(f"Breakpoint: {record.breakpoint_hit}")

# Full heatmap
stats = dbg.heatmap(prog)
print(f"Atom entropy: {stats.atom_entropy:.3f} bits")

# Diff two encodings
diff = dbg.diff_programs(prog1, prog2)
print(f"Value diffs: {diff['value_diffs']}")
```

## Artifacts
- `src/wal/v1/debugger.py` — WALDebugger implementation
- `experiments/m78_wal_v1_debugger.py` — Test suite

## Notes
- Debugger supports both `CoeffTable` and raw tensor for coeffs
- Hierarchical trace shows full tree for L1+ atoms
- Heatmap computes entropy, frequencies, and residual coverage
- Diff works even when encodings use different random seeds
