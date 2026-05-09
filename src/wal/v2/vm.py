"""WAL v2 Virtual Machine: reference interpreter for program execution.

VM State:
  ACC: float32   # accumulator
  PC:  uint32    # program counter (weight index)

Memory:
  ATOM_TABLE[K]     # shared read-only atoms
  COEFF_TABLE[C]    # shared read-only coefficients
  PROGRAMS[N]       # per-weight programs (atom_id, coeff_id, residual, has_residual)
  OUTPUT[N]         # reconstructed weights
  ROW_SCALES[M]     # per-row normalization scales

Execution Cycle (per weight):
  ACC = 0
  ATOM = ATOM_TABLE[PROGRAM[PC].atom_id]
  COEF = COEFF_TABLE[PROGRAM[PC].coeff_id]
  ACC += ATOM * COEF
  if PROGRAM[PC].has_residual:
    ACC += PROGRAM[PC].residual
  OUTPUT[PC] = ACC * ROW_SCALES[row_of(PC)]
  PC += 1

No stack is needed for WAL v2 — a single accumulator suffices.
"""
import torch
from typing import Optional
from .isa import ProgramBufferV2, AtomTable, CoeffTable


class WALVMState:
    """Runtime state for WAL v2 VM."""
    def __init__(
        self,
        atoms: AtomTable,
        coeffs: CoeffTable,
        programs: ProgramBufferV2,
        row_scales: Optional[torch.Tensor] = None,
    ):
        self.atoms = atoms
        self.coeffs = coeffs
        self.programs = programs
        self.row_scales = row_scales
        self.pc = 0
        self.acc = 0.0

    def reset(self):
        self.pc = 0
        self.acc = 0.0


def vm_execute(
    state: WALVMState,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Execute all WAL v2 programs via reference interpreter.

    Args:
        state: VM state with atoms, coeffs, programs, row_scales
        device: Target device (defaults to atoms device)

    Returns:
        output: [N] reconstructed weights (with row scales applied if present)
    """
    if device is None:
        device = state.atoms.values.device

    atoms = state.atoms.values.to(device)
    coeffs = state.coeffs.values.to(device)
    prog = state.programs

    N = prog.N
    output = torch.empty(N, dtype=torch.float32, device=device)

    atom_ids = prog.atom_ids.to(device).long()
    coeff_ids = prog.coeff_ids.to(device).long()
    residuals = prog.residuals.to(device)
    has_residual = prog.has_residual.to(device)

    # Vectorized execution (all weights at once)
    acc = atoms[atom_ids] * coeffs[coeff_ids]
    if has_residual.any():
        acc += residuals * has_residual.float()

    # Apply row scales if provided
    if state.row_scales is not None:
        row_scales = state.row_scales.to(device).flatten()
        cols = prog.shape[1]
        rows = prog.shape[0]
        # Broadcast row scales: each row has 'cols' weights
        scale_map = row_scales.repeat_interleave(cols)[:N]
        acc = acc * scale_map

    return acc


def vm_execute_single(state: WALVMState, pc: int) -> float:
    """Execute a single program (step-through mode for debugging).

    Args:
        state: VM state
        pc: Program counter (weight index)

    Returns:
        value: Reconstructed weight value
    """
    atom_id = int(state.programs.atom_ids[pc].item())
    coeff_id = int(state.programs.coeff_ids[pc].item())
    residual = float(state.programs.residuals[pc].item())
    has_r = bool(state.programs.has_residual[pc].item())

    atom_val = float(state.atoms.values[atom_id].item())
    coeff_val = float(state.coeffs.values[coeff_id].item())

    acc = atom_val * coeff_val
    if has_r:
        acc += residual

    if state.row_scales is not None:
        row = pc // state.programs.shape[1]
        acc *= float(state.row_scales[row].item())

    return acc


def vm_step(state: WALVMState) -> Optional[float]:
    """Execute one step and advance PC (for interactive debugging).

    Returns:
        value or None if PC >= N
    """
    if state.pc >= state.programs.N:
        return None
    value = vm_execute_single(state, state.pc)
    state.pc += 1
    return value
