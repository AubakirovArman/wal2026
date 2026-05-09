#!/usr/bin/env python3
"""WAL Standard Library — pre-trained atom tables for popular models.

Phase 8: Reusable atom libraries enabling cross-model transfer.
"""
import json
import torch
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class AtomLibraryEntry:
    """A single entry in the atom library."""
    name: str                    # e.g., "llama-70b-attention"
    family: str                  # e.g., "llama", "mistral", "qwen"
    variant: str                 # e.g., "70b", "8b", "13b"
    component: str               # e.g., "attention", "mlp", "all"
    K: int
    C: int
    atom_values: List[float]     # Flat list of atom values
    coeff_values: List[float]    # Flat list of coeff values
    metadata: Dict = None        # Extra info (layer ranges, domain, etc.)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def atom_tensor(self) -> torch.Tensor:
        return torch.tensor(self.atom_values, dtype=torch.float32)
    
    @property
    def coeff_tensor(self) -> torch.Tensor:
        return torch.tensor(self.coeff_values, dtype=torch.float32)
    
    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            'name': self.name,
            'family': self.family,
            'variant': self.variant,
            'component': self.component,
            'K': self.K,
            'C': self.C,
            'atom_values': self.atom_values,
            'coeff_values': self.coeff_values,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "AtomLibraryEntry":
        return cls(**d)


class AtomLibrary:
    """Collection of pre-trained atom tables.
    
    Supports save/load to disk and querying by family/variant/component.
    """
    
    def __init__(self, name: str = "default", version: str = "1.0"):
        self.name = name
        self.version = version
        self.entries: Dict[str, AtomLibraryEntry] = {}
    
    def add_entry(self, entry: AtomLibraryEntry):
        """Add an entry to the library."""
        self.entries[entry.name] = entry
    
    def get_entry(self, name: str) -> Optional[AtomLibraryEntry]:
        """Get entry by name."""
        return self.entries.get(name)
    
    def find(self, family: Optional[str] = None,
             variant: Optional[str] = None,
             component: Optional[str] = None) -> List[AtomLibraryEntry]:
        """Find entries matching criteria."""
        results = []
        for entry in self.entries.values():
            if family and entry.family != family:
                continue
            if variant and entry.variant != variant:
                continue
            if component and entry.component != component:
                continue
            results.append(entry)
        return results
    
    def families(self) -> List[str]:
        """List all families in the library."""
        return sorted(set(e.family for e in self.entries.values()))
    
    def variants(self, family: Optional[str] = None) -> List[str]:
        """List all variants (optionally filtered by family)."""
        entries = self.entries.values()
        if family:
            entries = [e for e in entries if e.family == family]
        return sorted(set(e.variant for e in entries))
    
    def list_entries(self) -> List[str]:
        """List all entry names."""
        return sorted(self.entries.keys())
    
    def save(self, path: str):
        """Save library to disk.
        
        Format: JSON manifest + .pt files for tensors.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest = {
            'name': self.name,
            'version': self.version,
            'entries': {name: entry.to_dict() for name, entry in self.entries.items()},
        }
        with open(path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Save tensors separately for efficiency
        for name, entry in self.entries.items():
            torch.save({
                'atoms': entry.atom_tensor,
                'coeffs': entry.coeff_tensor,
            }, path / f"{name}.pt")
        
        print(f"Library saved to {path} ({len(self.entries)} entries)")
    
    @classmethod
    def load(cls, path: str) -> "AtomLibrary":
        """Load library from disk."""
        path = Path(path)
        with open(path / 'manifest.json') as f:
            manifest = json.load(f)
        
        lib = cls(name=manifest['name'], version=manifest['version'])
        
        for name, entry_dict in manifest['entries'].items():
            entry = AtomLibraryEntry.from_dict(entry_dict)
            
            # Load tensors if available
            tensor_path = path / f"{name}.pt"
            if tensor_path.exists():
                tensors = torch.load(tensor_path, weights_only=True)
                entry.atom_values = tensors['atoms'].tolist()
                entry.coeff_values = tensors['coeffs'].tolist()
            
            lib.add_entry(entry)
        
        print(f"Library loaded from {path} ({len(lib.entries)} entries)")
        return lib
    
    def __repr__(self) -> str:
        families = self.families()
        return f"AtomLibrary(name={self.name}, version={self.version}, entries={len(self.entries)}, families={families})"


def build_entry_from_encoded(
    name: str,
    family: str,
    variant: str,
    component: str,
    atoms: torch.Tensor,
    coeffs: torch.Tensor,
    metadata: Dict = None,
) -> AtomLibraryEntry:
    """Build a library entry from encoded atoms and coeffs.
    
    Args:
        name: Entry name
        family: Model family (llama, mistral, etc.)
        variant: Model variant (70b, 8b, etc.)
        component: Component type (attention, mlp, all)
        atoms: Atom tensor [K]
        coeffs: Coeff tensor [C]
        metadata: Optional metadata
    
    Returns:
        AtomLibraryEntry
    """
    return AtomLibraryEntry(
        name=name,
        family=family,
        variant=variant,
        component=component,
        K=atoms.numel(),
        C=coeffs.numel(),
        atom_values=atoms.cpu().float().tolist(),
        coeff_values=coeffs.cpu().float().tolist(),
        metadata=metadata or {},
    )


def create_default_library() -> AtomLibrary:
    """Create an empty default library with common structure.
    
    Users populate this by encoding their models and adding entries.
    """
    return AtomLibrary(name="wal-default", version="1.0")
