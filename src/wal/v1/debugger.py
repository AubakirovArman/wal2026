#!/usr/bin/env python3
"""WAL v1 Debugger & Inspector — Phase 7.

Tools for inspecting, tracing, and debugging WAL programs:
- Step-through execution per weight
- Conditional breakpoints
- Program heatmaps and statistics
- Diff between programs or models
- Hierarchical atom resolution traces
"""
import torch
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from .isa import AtomTableV1, ProgramBufferV1, CoeffTable


@dataclass
class Breakpoint:
    """A conditional breakpoint in WAL program execution."""
    condition: Callable[[int, int, float], bool]  # (atom_id, coeff_id, residual) -> bool
    name: str
    hit_count: int = 0
    
    def check(self, atom_id: int, coeff_id: int, residual: float) -> bool:
        """Check if breakpoint triggers for given weight."""
        if self.condition(atom_id, coeff_id, residual):
            self.hit_count += 1
            return True
        return False


@dataclass
class TraceRecord:
    """Single step trace record."""
    index: int
    atom_id: int
    coeff_id: int
    atom_value: float
    coeff_value: float
    product: float
    residual: float
    final_value: float
    atom_resolution_tree: Optional[str] = None
    breakpoint_hit: Optional[str] = None


@dataclass
class HeatmapStats:
    """Program heatmap statistics."""
    total_weights: int
    atom_frequencies: Dict[int, int] = field(default_factory=dict)
    coeff_frequencies: Dict[int, int] = field(default_factory=dict)
    top_atoms: List[Tuple[int, int, float]] = field(default_factory=list)  # (id, count, pct)
    top_coeffs: List[Tuple[int, int, float]] = field(default_factory=list)
    residual_count: int = 0
    residual_pct: float = 0.0
    atom_entropy: float = 0.0
    coeff_entropy: float = 0.0


class WALDebugger:
    """Debugger for WAL v1 programs.
    
    Supports:
    - Step-through execution with breakpoints
    - Hierarchical atom resolution tracing
    - Program heatmaps and statistics
    - Program diff
    """
    
    def __init__(self, atom_table: AtomTableV1, coeffs):
        self.atom_table = atom_table
        # Accept either CoeffTable or raw tensor
        if hasattr(coeffs, 'values') and callable(getattr(coeffs, 'values', None)):
            # It's a tensor — .values is a method
            self.coeff_values = coeffs
        else:
            # It's a CoeffTable
            self.coeff_values = coeffs.values
        self.breakpoints: List[Breakpoint] = []
        self.trace_log: List[TraceRecord] = []
        self._precomputed_atoms: Optional[torch.Tensor] = None
    
    def _get_precomputed_atoms(self) -> torch.Tensor:
        """Lazy precompute flat atom values."""
        if self._precomputed_atoms is None:
            from .decoder import precompute_flat_atoms
            self._precomputed_atoms = precompute_flat_atoms(self.atom_table)
        return self._precomputed_atoms
    
    def set_breakpoint(self, condition: Callable[[int, int, float], bool], name: str = "bp") -> Breakpoint:
        """Set a conditional breakpoint.
        
        Args:
            condition: Function (atom_id, coeff_id, residual) -> bool
            name: Breakpoint name for reporting
        
        Returns:
            The created breakpoint
        """
        bp = Breakpoint(condition=condition, name=name)
        self.breakpoints.append(bp)
        return bp
    
    def set_atom_breakpoint(self, atom_id: int, name: Optional[str] = None) -> Breakpoint:
        """Break when a specific atom is used."""
        return self.set_breakpoint(
            condition=lambda a, c, r: a == atom_id,
            name=name or f"atom_{atom_id}"
        )
    
    def set_coeff_breakpoint(self, coeff_id: int, name: Optional[str] = None) -> Breakpoint:
        """Break when a specific coefficient is used."""
        return self.set_breakpoint(
            condition=lambda a, c, r: c == coeff_id,
            name=name or f"coeff_{coeff_id}"
        )
    
    def set_residual_breakpoint(self, threshold: float = 0.0, name: Optional[str] = None) -> Breakpoint:
        """Break when residual exceeds threshold."""
        return self.set_breakpoint(
            condition=lambda a, c, r: abs(r) > threshold,
            name=name or f"residual_gt_{threshold}"
        )
    
    def clear_breakpoints(self):
        """Remove all breakpoints."""
        self.breakpoints.clear()
    
    def resolve_atom_tree(self, atom_id: int, max_depth: int = 10, _depth: int = 0) -> str:
        """Get hierarchical resolution tree for an atom as a string.
        
        Args:
            atom_id: Atom to resolve
            max_depth: Maximum recursion depth
            _depth: Current depth (internal)
        
        Returns:
            Formatted tree string
        """
        indent = "  " * _depth
        if atom_id < self.atom_table.K0:
            val = self.atom_table.base_atoms[atom_id].item()
            return f"{indent}ATOM {atom_id} [L0] = {val:.6f}"
        
        if _depth >= max_depth:
            return f"{indent}ATOM {atom_id} [... max depth]"
        
        d = self.atom_table.atom_defs[atom_id]
        lines = [f"{indent}ATOM {atom_id} [{d.op}]"]
        
        if d.children and d.coeffs:
            for child_id, coeff in zip(d.children, d.coeffs):
                child_tree = self.resolve_atom_tree(child_id, max_depth, _depth + 1)
                lines.append(child_tree)
                if _depth == 0:
                    lines.append(f"{indent}  * {coeff:.6f}")
        
        return "\n".join(lines)
    
    def step(self, prog: ProgramBufferV1, index: int) -> TraceRecord:
        """Execute a single step (one weight) with breakpoint checks.
        
        Args:
            prog: Program buffer
            index: Weight index to execute
        
        Returns:
            TraceRecord with full execution details
        """
        atom_id = int(prog.atom_ids[index])
        coeff_id = int(prog.coeff_ids[index])
        
        flat_atoms = self._get_precomputed_atoms()
        atom_val = flat_atoms[atom_id].item()
        coeff_val = self.coeff_values[coeff_id].item()
        product = atom_val * coeff_val
        
        residual = 0.0
        if prog.has_residual.numel() > 0 and prog.has_residual[index]:
            residual = prog.residuals[index].item()
        
        final_value = product + residual
        
        # Check breakpoints
        bp_hit = None
        for bp in self.breakpoints:
            if bp.check(atom_id, coeff_id, residual):
                bp_hit = bp.name
        
        # Build resolution tree for hierarchical atoms
        tree = None
        if atom_id >= self.atom_table.K0:
            tree = self.resolve_atom_tree(atom_id, max_depth=5)
        
        record = TraceRecord(
            index=index,
            atom_id=atom_id,
            coeff_id=coeff_id,
            atom_value=atom_val,
            coeff_value=coeff_val,
            product=product,
            residual=residual,
            final_value=final_value,
            atom_resolution_tree=tree,
            breakpoint_hit=bp_hit,
        )
        self.trace_log.append(record)
        return record
    
    def run(self, prog: ProgramBufferV1, start: int = 0, end: Optional[int] = None) -> List[TraceRecord]:
        """Run debugger over a range of weights.
        
        Args:
            prog: Program buffer
            start: Start index
            end: End index (exclusive), defaults to prog.N
        
        Returns:
            List of TraceRecords
        """
        if end is None:
            end = prog.N
        
        self.trace_log.clear()
        for i in range(start, min(end, prog.N)):
            self.step(prog, i)
        return self.trace_log
    
    def heatmap(self, prog: ProgramBufferV1) -> HeatmapStats:
        """Compute program usage heatmap.
        
        Args:
            prog: Program buffer
        
        Returns:
            HeatmapStats with frequencies and entropy
        """
        N = prog.N
        atom_counts: Dict[int, int] = {}
        coeff_counts: Dict[int, int] = {}
        residual_count = 0
        
        for i in range(N):
            aid = int(prog.atom_ids[i])
            cid = int(prog.coeff_ids[i])
            atom_counts[aid] = atom_counts.get(aid, 0) + 1
            coeff_counts[cid] = coeff_counts.get(cid, 0) + 1
            
            if prog.has_residual.numel() > 0 and prog.has_residual[i]:
                residual_count += 1
        
        # Top atoms
        top_atoms = sorted(
            [(aid, cnt, cnt / N * 100) for aid, cnt in atom_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        
        # Top coeffs
        top_coeffs = sorted(
            [(cid, cnt, cnt / N * 100) for cid, cnt in coeff_counts.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:20]
        
        # Entropy
        import math
        atom_entropy = -sum((cnt / N) * math.log2(cnt / N) for cnt in atom_counts.values())
        coeff_entropy = -sum((cnt / N) * math.log2(cnt / N) for cnt in coeff_counts.values())
        
        return HeatmapStats(
            total_weights=N,
            atom_frequencies=atom_counts,
            coeff_frequencies=coeff_counts,
            top_atoms=top_atoms,
            top_coeffs=top_coeffs,
            residual_count=residual_count,
            residual_pct=residual_count / N * 100,
            atom_entropy=atom_entropy,
            coeff_entropy=coeff_entropy,
        )
    
    def diff_programs(
        self,
        prog1: ProgramBufferV1,
        prog2: ProgramBufferV1,
        name1: str = "A",
        name2: str = "B",
    ) -> Dict[str, Any]:
        """Compute diff between two programs.
        
        Args:
            prog1: First program
            prog2: Second program
            name1: Label for first program
            name2: Label for second program
        
        Returns:
            Diff statistics
        """
        N = min(prog1.N, prog2.N)
        
        atom_diffs = 0
        coeff_diffs = 0
        residual_diffs = 0
        value_diffs = 0
        max_value_diff = 0.0
        
        flat_atoms = self._get_precomputed_atoms()
        
        diff_indices = []
        for i in range(N):
            a1 = int(prog1.atom_ids[i])
            a2 = int(prog2.atom_ids[i])
            c1 = int(prog1.coeff_ids[i])
            c2 = int(prog2.coeff_ids[i])
            
            v1 = flat_atoms[a1].item() * self.coeff_values[c1].item()
            v2 = flat_atoms[a2].item() * self.coeff_values[c2].item()
            
            if prog1.has_residual.numel() > 0 and prog1.has_residual[i]:
                v1 += prog1.residuals[i].item()
            if prog2.has_residual.numel() > 0 and prog2.has_residual[i]:
                v2 += prog2.residuals[i].item()
            
            diff = abs(v1 - v2)
            if diff > 1e-7:
                value_diffs += 1
                max_value_diff = max(max_value_diff, diff)
                
                if a1 != a2:
                    atom_diffs += 1
                if c1 != c2:
                    coeff_diffs += 1
                
                if len(diff_indices) < 10:
                    diff_indices.append({
                        'index': i,
                        'atom': (a1, a2),
                        'coeff': (c1, c2),
                        'value_diff': diff,
                    })
        
        return {
            'name1': name1,
            'name2': name2,
            'total_weights': N,
            'atom_diffs': atom_diffs,
            'coeff_diffs': coeff_diffs,
            'value_diffs': value_diffs,
            'value_diff_pct': value_diffs / N * 100,
            'max_value_diff': max_value_diff,
            'identical': value_diffs == 0,
            'sample_diffs': diff_indices,
        }
    
    def print_trace(self, record: TraceRecord, show_tree: bool = True):
        """Pretty-print a trace record."""
        print(f"  [{record.index:6d}] ATOM {record.atom_id:3d} × COEF {record.coeff_id:2d}")
        print(f"           atom_val={record.atom_value:10.6f} × coeff_val={record.coeff_value:10.6f}")
        print(f"           product={record.product:10.6f} + residual={record.residual:10.6f}")
        print(f"           final={record.final_value:10.6f}")
        if record.breakpoint_hit:
            print(f"           >>> BREAKPOINT HIT: {record.breakpoint_hit} <<<")
        if show_tree and record.atom_resolution_tree:
            print(f"           Resolution tree:")
            for line in record.atom_resolution_tree.split('\n'):
                print(f"             {line}")
    
    def print_heatmap(self, stats: HeatmapStats, top_k: int = 10):
        """Pretty-print heatmap statistics."""
        print(f"Program Heatmap ({stats.total_weights:,} weights)")
        print(f"  Residuals: {stats.residual_count} ({stats.residual_pct:.2f}%)")
        print(f"  Atom entropy: {stats.atom_entropy:.3f} bits")
        print(f"  Coeff entropy: {stats.coeff_entropy:.3f} bits")
        print(f"\n  Top {top_k} atoms:")
        for aid, cnt, pct in stats.top_atoms[:top_k]:
            print(f"    ATOM {aid:3d}: {cnt:8,} ({pct:5.2f}%)")
        print(f"\n  Top {top_k} coeffs:")
        for cid, cnt, pct in stats.top_coeffs[:top_k]:
            print(f"    COEF {cid:2d}: {cnt:8,} ({pct:5.2f}%)")
    
    def print_diff(self, diff: Dict[str, Any]):
        """Pretty-print diff results."""
        print(f"Diff: {diff['name1']} vs {diff['name2']}")
        print(f"  Total weights: {diff['total_weights']:,}")
        print(f"  Identical: {diff['identical']}")
        if not diff['identical']:
            print(f"  Value diffs: {diff['value_diffs']:,} ({diff['value_diff_pct']:.4f}%)")
            print(f"  Atom diffs: {diff['atom_diffs']:,}")
            print(f"  Coeff diffs: {diff['coeff_diffs']:,}")
            print(f"  Max value diff: {diff['max_value_diff']:.8f}")
            if diff['sample_diffs']:
                print(f"  Sample diffs (first 10):")
                for d in diff['sample_diffs']:
                    print(f"    [{d['index']:6d}] atom={d['atom']} coeff={d['coeff']} diff={d['value_diff']:.8f}")
