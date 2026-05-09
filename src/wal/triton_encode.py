"""Fused Triton kernels for WAL encode/decode."""
import torch
import triton
import triton.language as tl


@triton.jit
def wal_encode_scalar_fused_kernel(
    weights_ptr,    # [N] float32
    atoms_ptr,      # [K] float32
    recon_ptr,      # [N] float32
    N,
    K,
    lmax,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused WAL-0 encode: each thread handles one weight.
    
    For each weight, greedily finds best atom*sign for lmax steps.
    Atoms are cached in shared memory.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Load weight
    w = tl.load(weights_ptr + offs, mask=mask, other=0.0)
    residual = w
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Cache atoms in shared memory (small K, e.g. 128)
    atom_shared = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)  # per-thread private, not true shared
    
    for step in range(lmax):
        # Find best atom and sign via brute force
        best_score = tl.full((BLOCK_SIZE,), value=1e30, dtype=tl.float32)
        best_atom_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        for k in range(K):
            atom_k = tl.load(atoms_ptr + k)
            # Positive
            diff_pos = residual - atom_k
            score_pos = diff_pos * diff_pos
            is_better_pos = score_pos < best_score
            best_score = tl.where(is_better_pos, score_pos, best_score)
            best_atom_val = tl.where(is_better_pos, atom_k, best_atom_val)
            
            # Negative
            diff_neg = residual + atom_k
            score_neg = diff_neg * diff_neg
            is_better_neg = score_neg < best_score
            best_score = tl.where(is_better_neg, score_neg, best_score)
            best_atom_val = tl.where(is_better_neg, -atom_k, best_atom_val)
        
        # Only take step if it improves (score < residual^2)
        zero_score = residual * residual
        take = best_score < zero_score
        step_val = tl.where(take, best_atom_val, tl.zeros((BLOCK_SIZE,), dtype=tl.float32))
        
        acc += step_val
        residual -= step_val
    
    tl.store(recon_ptr + offs, acc, mask=mask)


def wal_encode_scalar_fused(
    weights: torch.Tensor,
    atoms: torch.Tensor,
    lmax: int,
    block_size: int = 1024,
) -> torch.Tensor:
    """Fused Triton encode wrapper.
    
    Args:
        weights: [N] float32
        atoms: [K] float32
        lmax: max program length
        block_size: Triton block size
    
    Returns:
        recon: [N] float32 reconstructed weights
    """
    N = weights.numel()
    K = atoms.shape[0]
    device = weights.device
    
    recon = torch.empty(N, dtype=torch.float32, device=device)
    weights = weights.contiguous().float()
    atoms = atoms.contiguous().float()
    
    grid = (triton.cdiv(N, block_size),)
    
    # Triton requires the CUDA context to match the tensor device
    with torch.cuda.device(device):
        wal_encode_scalar_fused_kernel[grid](
            weights, atoms, recon,
            N, K, lmax,
            BLOCK_SIZE=block_size,
        )
    
    return recon


@triton.jit
def wal_decode_scalar_kernel(
    indices_ptr,    # [N, lmax] uint8
    signs_ptr,      # [N, lmax] int8
    atoms_ptr,      # [K] float32
    output_ptr,     # [N] float32
    N,
    K,
    lmax,
    BLOCK_SIZE: tl.constexpr,
):
    """Decode WAL-0 programs (from M47)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for s in range(lmax):
        idx_off = offs * lmax + s
        idx = tl.load(indices_ptr + idx_off, mask=mask, other=0).to(tl.int32)
        sign = tl.load(signs_ptr + idx_off, mask=mask, other=0).to(tl.float32)
        atom = tl.load(atoms_ptr + idx, mask=mask, other=0.0)
        acc += atom * sign
    
    tl.store(output_ptr + offs, acc, mask=mask)
