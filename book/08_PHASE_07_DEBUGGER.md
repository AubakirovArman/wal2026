# Phase 7: Debugger (M78)

## The Problem

WAL programs are opaque. You can't step through them. You can't inspect individual weights. You can't compare two encodings.

## What Was Built

- **`WALDebugger`**: Interactive step-through execution
- **Breakpoints**: Atom ID, coefficient, residual, custom conditions
- **Trace log**: Full execution history
- **Heatmap**: Atom/coeff frequency and entropy statistics
- **Program diff**: Compare two encodings
- **Hierarchical trace**: Recursive tree for L1+ atoms

## Usage

```python
from wal.v1.debugger import WALDebugger

dbg = WALDebugger(atom_table, coeffs)

# Step through one weight
record = dbg.step(prog, index=42)
print(record)  # atom_id, coeff_id, atom_value, coeff_value, final_value, residual

# Set breakpoint
dbg.set_atom_breakpoint(atom_id=17)

# Run range
trace = dbg.run(prog, start=0, end=100)

# Heatmap
stats = dbg.heatmap(prog)
```

## Test Results

| Test | Result |
|------|--------|
| Step-through | ✅ PASS |
| Atom breakpoint | ✅ PASS |
| Coeff breakpoint | ✅ PASS |
| Residual breakpoint | ✅ PASS |
| Custom breakpoint | ✅ PASS |
| Heatmap | ✅ PASS |
| Program diff | ✅ PASS |

## Why This Matters

Before the debugger, WAL was a black box. After the debugger, you can trace any weight through its program, set breakpoints, and compare encodings. This is the difference between a format and a language.

## Files
- `src/wal/v1/debugger.py`
- `experiments/m78_wal_v1_debugger.py`
