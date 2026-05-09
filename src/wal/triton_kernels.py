"""WAL-0 Triton kernels for GPU decode."""
import torch
import triton
import triton.language as tl


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
    """Decode WAL-0 programs: each thread handles one weight.
    
    output[i] = sum_{s=0}^{lmax-1} atoms[indices[i,s]] * signs[i,s]
    """
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


def wal_decode_scalar_triton(
    indices: torch.Tensor,
    signs: torch.Tensor,
    atoms: torch.Tensor,
    output: torch.Tensor = None,
    block_size: int = 1024,
) -> torch.Tensor:
    """Triton decode wrapper.
    
    Args:
        indices: [N, lmax] uint8 or int32
        signs: [N, lmax] int8 or float32
        atoms: [K] float32
        output: optional pre-allocated [N] float32
        block_size: Triton block size
    
    Returns:
        output: [N] float32 reconstructed weights
    """
    N, lmax = indices.shape
    K = atoms.shape[0]
    
    if output is None:
        output = torch.empty(N, dtype=torch.float32, device=indices.device)
    
    # Ensure contiguous
    indices = indices.contiguous()
    signs = signs.contiguous()
    atoms = atoms.contiguous()
    
    grid = (triton.cdiv(N, block_size),)
    
    wal_decode_scalar_kernel[grid](
        indices, signs, atoms, output,
        N, K, lmax,
        BLOCK_SIZE=block_size,
    )
    
    return output
