"""Fused Triton kernels for WAL v2 decode."""
import torch
import triton
import triton.language as tl


@triton.jit
def wal_v2_decode_kernel(
    atom_ids_ptr,    # [N] uint8
    coeff_ids_ptr,   # [N] uint8
    residuals_ptr,   # [N] float32
    has_residual_ptr, # [N] uint8 (0 or 1)
    atom_table_ptr,  # [K] float32
    coeff_table_ptr, # [C] float32
    output_ptr,      # [N] float32
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Decode WAL v2 programs: each thread handles one weight.

    output[i] = atom_table[atom_ids[i]] * coeff_table[coeff_ids[i]]
                + (has_residual[i] ? residuals[i] : 0)
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load atom_id and coeff_id
    atom_id = tl.load(atom_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    coeff_id = tl.load(coeff_ids_ptr + offs, mask=mask, other=0).to(tl.int32)

    # Lookup atom and coeff values
    atom_val = tl.load(atom_table_ptr + atom_id, mask=mask, other=0.0)
    coeff_val = tl.load(coeff_table_ptr + coeff_id, mask=mask, other=0.0)

    # Compute base reconstruction
    acc = atom_val * coeff_val

    # Add residual if present
    has_r = tl.load(has_residual_ptr + offs, mask=mask, other=0).to(tl.int32)
    residual = tl.load(residuals_ptr + offs, mask=mask, other=0.0)
    acc += tl.where(has_r != 0, residual, 0.0)

    tl.store(output_ptr + offs, acc, mask=mask)


@triton.jit
def wal_v2_decode_with_row_scale_kernel(
    atom_ids_ptr,     # [N] uint8
    coeff_ids_ptr,    # [N] uint8
    residuals_ptr,    # [N] float32
    has_residual_ptr,  # [N] uint8
    atom_table_ptr,   # [K] float32
    coeff_table_ptr,  # [C] float32
    row_scales_ptr,   # [M] float32
    output_ptr,       # [N] float32
    N,
    cols: tl.constexpr,  # number of columns per row
    BLOCK_SIZE: tl.constexpr,
):
    """Decode WAL v2 programs with per-row scale multiplication.

    output[i] = (atom_table[atom_ids[i]] * coeff_table[coeff_ids[i]]
                 + residual_if_present) * row_scales[row_of(i)]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load atom_id and coeff_id
    atom_id = tl.load(atom_ids_ptr + offs, mask=mask, other=0).to(tl.int32)
    coeff_id = tl.load(coeff_ids_ptr + offs, mask=mask, other=0).to(tl.int32)

    # Lookup atom and coeff values
    atom_val = tl.load(atom_table_ptr + atom_id, mask=mask, other=0.0)
    coeff_val = tl.load(coeff_table_ptr + coeff_id, mask=mask, other=0.0)

    # Compute base reconstruction
    acc = atom_val * coeff_val

    # Add residual if present
    has_r = tl.load(has_residual_ptr + offs, mask=mask, other=0).to(tl.int32)
    residual = tl.load(residuals_ptr + offs, mask=mask, other=0.0)
    acc += tl.where(has_r != 0, residual, 0.0)

    # Apply row scale
    row_idx = offs // cols
    row_scale = tl.load(row_scales_ptr + row_idx, mask=mask, other=1.0)
    acc = acc * row_scale

    tl.store(output_ptr + offs, acc, mask=mask)


def wal_v2_decode_triton(
    prog,
    atoms,
    coeffs,
    output: torch.Tensor = None,
    row_scales: torch.Tensor = None,
    block_size: int = 1024,
) -> torch.Tensor:
    """Triton decode wrapper for WAL v2.

    Args:
        prog: ProgramBufferV2
        atoms: AtomTable
        coeffs: CoeffTable
        output: optional pre-allocated [N] float32
        row_scales: optional [M] float32 per-row scales
        block_size: Triton block size

    Returns:
        output: [N] float32 reconstructed weights
    """
    N = prog.N
    K = atoms.K
    C = coeffs.C
    device = atoms.values.device

    if output is None:
        output = torch.empty(N, dtype=torch.float32, device=device)

    # Ensure contiguous
    atom_ids = prog.atom_ids.to(device).contiguous()
    coeff_ids = prog.coeff_ids.to(device).contiguous()
    residuals = prog.residuals.to(device).contiguous()
    has_residual = prog.has_residual.to(device).contiguous().to(torch.uint8)
    atom_table = atoms.values.to(device).contiguous()
    coeff_table = coeffs.values.to(device).contiguous()

    grid = (triton.cdiv(N, block_size),)

    with torch.cuda.device(device):
        if row_scales is not None:
            row_scales = row_scales.to(device).contiguous()
            cols = prog.shape[1] if len(prog.shape) >= 2 else 1
            wal_v2_decode_with_row_scale_kernel[grid](
                atom_ids, coeff_ids, residuals, has_residual,
                atom_table, coeff_table, row_scales, output,
                N, cols,
                BLOCK_SIZE=block_size,
            )
        else:
            wal_v2_decode_kernel[grid](
                atom_ids, coeff_ids, residuals, has_residual,
                atom_table, coeff_table, output,
                N,
                BLOCK_SIZE=block_size,
            )

    return output
