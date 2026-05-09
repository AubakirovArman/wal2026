#!/usr/bin/env python3
"""WAL v1 PyTorch Integration — nn.Module wrappers for WAL-encoded weights.

Phase 6: Bridge between WAL compressed representation and PyTorch execution.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .isa import ProgramBufferV1, AtomTableV1, CoeffTable
from .decoder import wal_decode_v1
from .meta import WALProgramAdapter


class WALParameter:
    """A parameter-like object that stores weights in WAL-encoded format.
    
    Supports lazy decoding with optional cache. The decoded weights are
    materialized on demand and can be cleared to free memory.
    """
    
    def __init__(
        self,
        prog: ProgramBufferV1,
        atom_table: AtomTableV1,
        coeffs: CoeffTable,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ):
        self.prog = prog
        self.atom_table = atom_table
        self.coeffs = coeffs
        self.shape = shape
        self.dtype = dtype
        self._cache: Optional[torch.Tensor] = None
        self._cache_device: Optional[torch.device] = None
    
    def decode(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Decode WAL-encoded weights to dense tensor.
        
        Uses cached value if available and on the same device.
        """
        if self._cache is not None:
            if device is None or self._cache_device == device:
                return self._cache
        
        # Decode
        weight = wal_decode_v1(self.prog, self.atom_table, self.coeffs.values)
        weight = weight.reshape(self.shape).to(self.dtype)
        
        if device is not None:
            weight = weight.to(device)
        
        self._cache = weight
        self._cache_device = weight.device
        return weight
    
    def clear_cache(self):
        """Clear decoded weight cache to free memory."""
        self._cache = None
        self._cache_device = None
    
    @property
    def numel(self) -> int:
        """Total number of elements."""
        return int(torch.prod(torch.tensor(self.shape)).item())
    
    def __repr__(self):
        return (
            f"WALParameter(shape={self.shape}, K0={self.atom_table.K0}, "
            f"K_total={self.atom_table.K_total}, C={self.coeffs.values.numel()}, "
            f"cached={self._cache is not None})"
        )


class WALLinear(nn.Module):
    """Linear layer with WAL-encoded weight matrix.
    
    The weight matrix is stored in WAL-compressed form and decoded on-the-fly
    during forward pass. Supports optional caching for repeated forwards.
    """
    
    def __init__(
        self,
        wal_weight: WALParameter,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.wal_weight = wal_weight
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: decode weight and apply linear transformation."""
        weight = self.wal_weight.decode(x.device)
        return F.linear(x, weight, self.bias)
    
    def clear_cache(self):
        """Clear weight decode cache."""
        self.wal_weight.clear_cache()
    
    @property
    def weight_shape(self) -> tuple:
        return self.wal_weight.shape
    
    def __repr__(self):
        return f"WALLinear(in_features={self.weight_shape[1]}, out_features={self.weight_shape[0]}, bias={self.bias is not None})"


class WALCachedLinear(nn.Module):
    """Linear layer with WAL-encoded weights and persistent decode cache.
    
    Decodes weights once and keeps them in memory for fast repeated access.
    Use this when memory is available and speed is critical.
    
    Supports optional meta-learning adapter for task-specific fine-tuning.
    """
    
    def __init__(
        self,
        wal_weight: WALParameter,
        bias: Optional[torch.Tensor] = None,
        adapter: Optional[WALProgramAdapter] = None,
    ):
        super().__init__()
        self.wal_weight = wal_weight
        if bias is not None:
            self.register_buffer('bias', bias)
        else:
            self.bias = None
        self.adapter = adapter
        self._decoded = False
    
    def _ensure_decoded(self, device: torch.device):
        if not self._decoded:
            weight = self.wal_weight.decode(device)
            if self.adapter is not None:
                weight = self.adapter(weight)
            self.register_buffer('weight', weight)
            self._decoded = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._ensure_decoded(x.device)
        return F.linear(x, self.weight, self.bias)
    
    def clear_cache(self):
        if self._decoded and hasattr(self, 'weight'):
            delattr(self, 'weight')
            self._decoded = False
    
    def set_adapter(self, adapter: Optional[WALProgramAdapter]):
        """Attach or detach a meta-learning adapter."""
        self.adapter = adapter
        self.clear_cache()


def encode_linear_weight(
    weight: torch.Tensor,
    K: int = 256,
    C: int = 16,
    build_hier: bool = False,
    max_l1: int = 64,
) -> WALParameter:
    """Encode a linear layer weight matrix to WAL format.
    
    Args:
        weight: Dense weight tensor [out_features, in_features]
        K: Number of base atoms
        C: Number of coefficients
        build_hier: Whether to build hierarchical atoms
        max_l1: Max L1 atoms if build_hier=True
    
    Returns:
        WALParameter with encoded weights
    """
    from .encoder import build_l0_atoms, build_coeff_table, wal_encode_v1, build_hierarchical_atoms
    
    # Flatten for encoding
    flat = weight.reshape(-1)
    
    # Build atoms and coeffs
    atoms = build_l0_atoms(flat, K=K, iters=3)
    coeffs_tensor = build_coeff_table(flat, atoms, C=C, iters=3)
    
    # Encode (smaller batch to avoid OOM on large layers)
    prog, recon = wal_encode_v1(flat, atoms, coeffs_tensor, batch=262_144)
    
    # Build hierarchical atoms if requested
    if build_hier:
        atom_table = build_hierarchical_atoms(atoms, prog, max_l1=max_l1)
    else:
        from .isa import AtomDef
        atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
        atom_table = AtomTableV1(base_atoms=atoms, atom_defs=atom_defs)
    
    coeff_table = CoeffTable(values=coeffs_tensor)
    
    return WALParameter(
        prog=prog,
        atom_table=atom_table,
        coeffs=coeff_table,
        shape=weight.shape,
        dtype=weight.dtype,
    )


def replace_wal_with_linear(model: nn.Module) -> nn.Module:
    """Replace all WALLinear / WALCachedLinear layers with standard nn.Linear.
    
    Decodes WAL-encoded weights to dense tensors. Useful for the
    "edit in weight space, store in WAL space" workflow.
    
    Args:
        model: PyTorch model with WAL layers
    
    Returns:
        Modified model (in-place)
    """
    for name, module in model.named_children():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            weight = module.wal_weight.decode(
                module.wal_weight._cache_device or torch.device("cpu")
            )
            bias = module.bias.data if module.bias is not None else None
            new_layer = nn.Linear(
                weight.shape[1],
                weight.shape[0],
                bias=bias is not None,
                dtype=weight.dtype,
                device=weight.device,
            )
            with torch.no_grad():
                new_layer.weight.copy_(weight)
                if bias is not None:
                    new_layer.bias.copy_(bias)
            setattr(model, name, new_layer)
        else:
            replace_wal_with_linear(module)
    return model


def replace_linear_with_wal(
    model: nn.Module,
    K: int = 256,
    C: int = 16,
    build_hier: bool = False,
    max_l1: int = 64,
    cached: bool = False,
) -> nn.Module:
    """Replace all nn.Linear layers in a model with WALLinear layers.
    
    Args:
        model: PyTorch model
        K: Number of base atoms
        C: Number of coefficients
        build_hier: Whether to build hierarchical atoms
        max_l1: Max L1 atoms
        cached: Use WALCachedLinear instead of WALLinear
    
    Returns:
        Modified model (in-place)
    """
    LinearClass = WALCachedLinear if cached else WALLinear
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Encode weight
            wal_param = encode_linear_weight(
                module.weight.data,
                K=K, C=C,
                build_hier=build_hier,
                max_l1=max_l1,
            )
            # Create replacement
            new_layer = LinearClass(
                wal_weight=wal_param,
                bias=module.bias.data if module.bias is not None else None,
            )
            # Replace
            setattr(model, name, new_layer)
        else:
            # Recurse
            replace_linear_with_wal(
                module, K=K, C=C,
                build_hier=build_hier,
                max_l1=max_l1,
                cached=cached,
            )
    
    return model


def wal_state_dict(model: nn.Module) -> dict:
    """Extract state dict with WAL-encoded weights.
    
    For WALLinear layers, serializes the WAL representation instead of
    dense weights.
    """
    from .format import serialize_wal_v1
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            blob = serialize_wal_v1(
                module.wal_weight.prog,
                module.wal_weight.atom_table,
                module.wal_weight.coeffs,
            )
            state[f"{name}.wal_weight"] = blob
            if module.bias is not None:
                state[f"{name}.bias"] = module.bias
    return state


def wal_load_state_dict(model: nn.Module, state_dict: dict):
    """Load state dict with WAL-encoded weights.
    
    Deserializes WAL blobs and reconstructs WALLinear layers.
    """
    from .format import deserialize_wal_v1
    for name, module in model.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            blob = state_dict.get(f"{name}.wal_weight")
            if blob is not None:
                prog, atom_table, coeffs, meta = deserialize_wal_v1(blob)
                module.wal_weight = WALParameter(
                    prog=prog,
                    atom_table=atom_table,
                    coeffs=coeffs,
                    shape=tuple(meta['shape']),
                )
            bias_key = f"{name}.bias"
            if bias_key in state_dict:
                module.bias = state_dict[bias_key]
