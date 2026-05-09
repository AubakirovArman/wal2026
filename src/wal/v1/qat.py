#!/usr/bin/env python3
"""QAT (Quantization-Aware Training) for WAL.

Phase 14: Make WAL decoding differentiable so atom/coeff tables
can be learned end-to-end via backpropagation while program indices
(atom_ids, coeff_ids) remain fixed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class WALQATLinear(nn.Module):
    """Differentiable linear layer with WAL-encoded weights.
    
    Unlike WALLinear/WALCachedLinear, this layer makes atom and coefficient
    tables learnable nn.Parameter objects. Program indices (atom_ids, coeff_ids)
    remain fixed buffers.
    
    Supports WAL-native adapters (coeff_delta, atom_delta) as a lightweight
    alternative to LoRA. Instead of rank*(in+out) parameters, WAL-native
    adapters use only K+C parameters.
    
    Forward pass:
        weight = (atom_values + atom_delta)[atom_ids] * (coeff_values + coeff_delta)[coeff_ids] + residual
        output = F.linear(x, weight, bias)
    
    Gradients flow to atom_values and coeff_values but NOT to program indices.
    """
    
    def __init__(
        self,
        atom_ids: torch.Tensor,       # [N] uint8 — fixed programs
        coeff_ids: torch.Tensor,      # [N] uint8 — fixed programs
        atom_values: torch.Tensor,    # [K] float32 — LEARNABLE
        coeff_values: torch.Tensor,   # [C] float32 — LEARNABLE
        shape: Tuple[int, ...],       # weight shape (out_features, in_features)
        residuals: Optional[torch.Tensor] = None,   # [N] float16 — fixed
        has_residual: Optional[torch.Tensor] = None,  # [N] bool — fixed
        bias: Optional[torch.Tensor] = None,
        use_coeff_adapter: bool = False,
        use_atom_adapter: bool = False,
        atom_adapt_num: int = 8,
    ):
        super().__init__()
        
        # Fixed program buffers (non-trainable)
        self.register_buffer('atom_ids', atom_ids.cpu().to(torch.uint8))
        self.register_buffer('coeff_ids', coeff_ids.cpu().to(torch.uint8))
        
        # Learnable tables
        self.atom_values = nn.Parameter(atom_values.clone().float())
        self.coeff_values = nn.Parameter(coeff_values.clone().float())
        
        # Residuals (fixed)
        if residuals is not None:
            self.register_buffer('residuals', residuals.cpu().to(torch.float16))
        else:
            self.register_buffer('residuals', torch.tensor([], dtype=torch.float16))
        
        if has_residual is not None:
            self.register_buffer('has_residual', has_residual.cpu().to(torch.bool))
        else:
            self.register_buffer('has_residual', torch.tensor([], dtype=torch.bool))
        
        self.weight_shape = shape
        
        # Bias
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None
        
        # WAL-native adapters (lightweight LoRA alternative)
        self.use_coeff_adapter = use_coeff_adapter
        self.use_atom_adapter = use_atom_adapter
        
        if use_coeff_adapter:
            self.coeff_adapter = nn.Parameter(torch.zeros(coeff_values.numel()))
        else:
            self.coeff_adapter = None
        
        if use_atom_adapter:
            self.atom_adapter = nn.Parameter(torch.zeros(atom_values.numel()))
            # Optional: sparsity mask for atom adaptation
            self.register_buffer('atom_adapt_mask', torch.ones(atom_values.numel(), dtype=torch.bool))
            if atom_adapt_num < atom_values.numel():
                self.atom_adapt_mask[atom_adapt_num:] = False
        else:
            self.atom_adapter = None
            self.atom_adapt_mask = None
        
        # Precomputed numel for reshape
        self._N = int(torch.prod(torch.tensor(shape)).item())
        assert self.atom_ids.numel() == self._N, (
            f"atom_ids size {self.atom_ids.numel()} != weight numel {self._N}"
        )
    
    def decode_weight(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Differentiable decode of WAL weight.
        
        weight = atom_values[atom_ids] * coeff_values[coeff_ids] + residual
        
        If adapters are enabled, applies them before decode.
        """
        if device is None:
            device = self.atom_values.device
        
        # Apply adapters if enabled
        atom_values = self.atom_values
        coeff_values = self.coeff_values
        
        if self.use_coeff_adapter and self.coeff_adapter is not None:
            coeff_values = coeff_values + self.coeff_adapter
        
        if self.use_atom_adapter and self.atom_adapter is not None:
            if self.atom_adapt_mask is not None:
                atom_values = atom_values + self.atom_adapter * self.atom_adapt_mask.float()
            else:
                atom_values = atom_values + self.atom_adapter
        
        # Move indices to same device as parameters (for differentiable indexing)
        atom_ids = self.atom_ids.to(device=device, dtype=torch.long)
        coeff_ids = self.coeff_ids.to(device=device, dtype=torch.long)
        
        # Differentiable lookup: indexing a Parameter with integer indices
        # produces gradients w.r.t. the Parameter, not the indices
        weight = atom_values[atom_ids] * coeff_values[coeff_ids]
        
        # Add residuals (if any)
        if self.residuals.numel() > 0:
            residuals = self.residuals.to(device=device, dtype=torch.float32)
            weight = weight + residuals
        
        return weight.reshape(self.weight_shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with differentiable WAL decode."""
        # Ensure module is on same device as input (needed for device_map="auto")
        if next(self.parameters()).device != x.device:
            self.to(x.device)
        weight = self.decode_weight(x.device)
        # Match input dtype (e.g., float16 for mixed precision training)
        weight = weight.to(x.dtype)
        return F.linear(x, weight, self.bias)
    
    def reencode_programs(self) -> None:
        """Re-encode programs with current atom/coeff values.
        
        This is a non-differentiable operation that updates the fixed
        program buffers (atom_ids, coeff_ids) to better match the current
        learned tables. Should be called periodically during training.
        """
        with torch.no_grad():
            from ..v2.encoder import _encode_batch
            
            # Flatten current decoded weight
            flat = self.decode_weight().reshape(-1).cpu()
            
            # Re-encode with current tables
            atom_values_cpu = self.atom_values.detach().cpu()
            coeff_values_cpu = self.coeff_values.detach().cpu()
            
            new_atom_ids, new_coeff_ids, new_residuals, new_has_residual = _encode_batch(
                flat, atom_values_cpu, coeff_values_cpu
            )
            
            self.atom_ids.copy_(new_atom_ids.to(torch.uint8))
            self.coeff_ids.copy_(new_coeff_ids.to(torch.uint8))
            self.residuals.copy_(new_residuals.to(torch.float16))
            self.has_residual.copy_(new_has_residual.to(torch.bool))
    
    def to_wal_parameter(self):
        """Export to WALParameter for serialization.
        
        Returns a WALParameter with the CURRENT learned atom/coeff values
        and fixed program indices.
        """
        from .isa import AtomTableV1, AtomDef, ProgramBufferV1, CoeffTable
        from .nn import WALParameter
        
        K = self.atom_values.numel()
        atom_defs = [AtomDef(level=0, op="CONST") for _ in range(K)]
        atom_table = AtomTableV1(
            base_atoms=self.atom_values.detach().cpu(),
            atom_defs=atom_defs,
        )
        coeff_table = CoeffTable(values=self.coeff_values.detach().cpu())
        
        prog = ProgramBufferV1(
            atom_ids=self.atom_ids.cpu(),
            coeff_ids=self.coeff_ids.cpu(),
            residuals=self.residuals.cpu() if self.residuals.numel() > 0 else torch.tensor([], dtype=torch.float16),
            has_residual=self.has_residual.cpu() if self.has_residual.numel() > 0 else torch.tensor([], dtype=torch.bool),
            shape=self.weight_shape,
        )
        
        return WALParameter(
            prog=prog,
            atom_table=atom_table,
            coeffs=coeff_table,
            shape=self.weight_shape,
        )
    
    @property
    def numel(self) -> int:
        """Total number of weight elements."""
        return self._N
    
    def __repr__(self):
        return (
            f"WALQATLinear(in_features={self.weight_shape[1]}, "
            f"out_features={self.weight_shape[0]}, K={self.atom_values.numel()}, "
            f"C={self.coeff_values.numel()}, bias={self.bias is not None})"
        )


def linear_to_qat(
    linear: nn.Linear,
    K: int = 256,
    C: int = 16,
    encode_iters: int = 3,
    use_coeff_adapter: bool = False,
    use_atom_adapter: bool = False,
    atom_adapt_num: int = 8,
) -> WALQATLinear:
    """Convert a nn.Linear layer to WALQATLinear.
    
    Encodes the weight via WAL v2, then creates a differentiable QAT layer
    with learnable atom/coeff tables.
    
    Args:
        linear: Source nn.Linear layer
        K: Number of atoms
        C: Number of coefficients
        encode_iters: k-means/Lloyd-Max iterations
        use_coeff_adapter: Enable WAL-native coeff adapter
        use_atom_adapter: Enable WAL-native atom adapter
        atom_adapt_num: Number of atoms to adapt (if use_atom_adapter)
    
    Returns:
        WALQATLinear with fixed programs and learnable tables
    """
    from ..v2.encoder import build_atoms_kmeans_v2, build_coeff_table, wal_encode_v2
    
    weight = linear.weight.data
    flat = weight.reshape(-1)
    
    # Build atoms and coeffs (non-differentiable, one-time)
    atoms_tensor = build_atoms_kmeans_v2(flat, K=K, iters=encode_iters)
    coeffs_tensor = build_coeff_table(flat, atoms_tensor, C=C, iters=encode_iters)
    
    # Wrap in AtomTable/CoeffTable for wal_encode_v2
    from ..v2.isa import AtomTable, CoeffTable
    atoms = AtomTable(values=atoms_tensor)
    coeffs = CoeffTable(values=coeffs_tensor)
    
    # Encode programs (non-differentiable, one-time)
    prog, recon = wal_encode_v2(flat, atoms, coeffs)
    
    # Create QAT layer with learnable tables initialized from encoded values
    qat_layer = WALQATLinear(
        atom_ids=prog.atom_ids,
        coeff_ids=prog.coeff_ids,
        atom_values=atoms.values,      # Will become nn.Parameter
        coeff_values=coeffs.values,     # Will become nn.Parameter
        shape=weight.shape,
        residuals=prog.residuals if prog.residuals.numel() > 0 else None,
        has_residual=prog.has_residual if prog.has_residual.numel() > 0 else None,
        bias=linear.bias.data if linear.bias is not None else None,
        use_coeff_adapter=use_coeff_adapter,
        use_atom_adapter=use_atom_adapter,
        atom_adapt_num=atom_adapt_num,
    )
    
    return qat_layer


def model_to_qat(
    model: nn.Module,
    K: int = 256,
    C: int = 16,
    encode_iters: int = 3,
) -> nn.Module:
    """Convert all nn.Linear layers in a model to WALQATLinear.
    
    Args:
        model: PyTorch model
        K: Number of atoms
        C: Number of coefficients
        encode_iters: Encoding iterations
    
    Returns:
        Modified model (in-place)
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            qat_layer = linear_to_qat(module, K=K, C=C, encode_iters=encode_iters)
            setattr(model, name, qat_layer)
        else:
            model_to_qat(module, K=K, C=C, encode_iters=encode_iters)
    return model


def count_qat_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable vs total parameters in a QAT model.
    
    Returns:
        (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
