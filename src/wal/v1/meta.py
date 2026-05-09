#!/usr/bin/env python3
"""WAL v1 Meta-Learning — fine-tune programs, model soups, evolution.

Phase 10: Edit programs, not weights.

Components:
- WALProgramAdapter: task-specific residual adapter (LoRA for WAL)
- WALCoeffAdapter: learned coefficient offsets
- program_soup: merge programs from N models
- evolve_programs: genetic algorithm on atom combinations
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
from .isa import ProgramBufferV1, AtomTableV1, CoeffTable
from .decoder import wal_decode_v1


class WALProgramAdapter(nn.Module):
    """Task-specific adapter for WAL-encoded weights.
    
    Like LoRA but for WAL: frozen base programs + learned residual.
    The residual is stored in a compact form (low-rank or sparse).
    
    Args:
        shape: Weight shape (out_features, in_features)
        rank: Adapter rank (default: 4)
        alpha: Scaling factor (default: 1.0)
    """
    
    def __init__(self, shape: Tuple[int, ...], rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.shape = shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank residual: A [out, rank] @ B [rank, in]
        out_dim = shape[0] if len(shape) >= 1 else 1
        in_dim = shape[1] if len(shape) >= 2 else 1
        
        self.lora_A = nn.Parameter(torch.randn(out_dim, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, in_dim))
    
    def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Add adapter residual to decoded weight.
        
        Args:
            base_weight: Decoded weight from WAL programs
        
        Returns:
            Adapted weight
        """
        residual = (self.lora_A @ self.lora_B) * self.scaling
        return base_weight + residual.to(base_weight.device)
    
    def merge(self, base_weight: torch.Tensor) -> torch.Tensor:
        """Merge adapter into base weight for inference.
        
        Returns:
            Merged weight (no adapter overhead at inference)
        """
        return self.forward(base_weight)
    
    def extra_repr(self):
        return f"shape={self.shape}, rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


class WALCoeffAdapter(nn.Module):
    """Learned coefficient offset adapter.
    
    Instead of modifying weights directly, this adapter learns
    per-coefficient offsets: weight = atom * (coeff + learned_delta).
    
    This is more WAL-native than residual adapters because it
    operates in the coefficient space.
    """
    
    def __init__(self, num_coeffs: int, init_scale: float = 0.01):
        super().__init__()
        self.num_coeffs = num_coeffs
        # Learnable offset per coefficient index
        self.coeff_delta = nn.Parameter(torch.zeros(num_coeffs))
        self.init_scale = init_scale
        
        # Initialize near zero
        with torch.no_grad():
            self.coeff_delta.normal_(0, init_scale)
    
    def adapt_coeffs(self, base_coeffs: torch.Tensor) -> torch.Tensor:
        """Apply learned offsets to coefficient table.
        
        Args:
            base_coeffs: Base coefficient values [C]
        
        Returns:
            Adapted coefficients [C]
        """
        return base_coeffs + self.coeff_delta
    
    def extra_repr(self):
        return f"num_coeffs={self.num_coeffs}, init_scale={self.init_scale}"


class WALAtomAdapter(nn.Module):
    """Task-specific atom adaptation.
    
    Learns small perturbations to a subset of atoms.
    More expressive than coeff adapter but more parameters.
    """
    
    def __init__(self, num_atoms: int, num_adapt: int = 8, init_scale: float = 0.01):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_adapt = min(num_adapt, num_atoms)
        
        # Which atoms to adapt (fixed, not learned)
        self.register_buffer('adapt_mask', torch.zeros(num_atoms, dtype=torch.bool))
        self.adapt_mask[:self.num_adapt] = True
        
        # Learnable perturbations for adapted atoms
        self.atom_delta = nn.Parameter(torch.zeros(num_atoms))
        
        with torch.no_grad():
            self.atom_delta.normal_(0, init_scale)
    
    def adapt_atoms(self, base_atoms: torch.Tensor) -> torch.Tensor:
        """Apply learned perturbations to atom table.
        
        Args:
            base_atoms: Base atom values [K]
        
        Returns:
            Adapted atoms [K]
        """
        delta = self.atom_delta * self.adapt_mask.float()
        return base_atoms + delta
    
    def extra_repr(self):
        return f"num_atoms={self.num_atoms}, num_adapt={self.num_adapt}"


def program_soup(
    programs: List[ProgramBufferV1],
    weights: Optional[List[float]] = None,
    method: str = "mean",
) -> ProgramBufferV1:
    """Merge programs from multiple models (model soup at program level).
    
    Args:
        programs: List of ProgramBufferV1 from different models
        weights: Optional weight per model (default: equal)
        method: "mean", "majority", or "weighted"
    
    Returns:
        Merged ProgramBufferV1
    """
    if not programs:
        raise ValueError("Empty program list")
    
    N = programs[0].N
    shape = programs[0].shape
    
    if weights is None:
        weights = [1.0 / len(programs)] * len(programs)
    
    if method == "mean" or method == "weighted":
        # Average atom_ids and coeff_ids (not ideal but works as baseline)
        device = programs[0].atom_ids.device
        atom_ids_sum = torch.zeros(N, dtype=torch.float32, device=device)
        coeff_ids_sum = torch.zeros(N, dtype=torch.float32, device=device)
        
        for prog, w in zip(programs, weights):
            atom_ids_sum += prog.atom_ids.float() * w
            coeff_ids_sum += prog.coeff_ids.float() * w
        
        merged_atom_ids = atom_ids_sum.round().clamp(0, 255).to(torch.uint8)
        merged_coeff_ids = coeff_ids_sum.round().clamp(0, 255).to(torch.uint8)
        
    elif method == "majority":
        # Majority vote per position
        merged_atom_ids = torch.zeros(N, dtype=torch.uint8)
        merged_coeff_ids = torch.zeros(N, dtype=torch.uint8)
        
        for i in range(N):
            atom_votes = [int(p.atom_ids[i]) for p in programs]
            coeff_votes = [int(p.coeff_ids[i]) for p in programs]
            
            # Most common
            from collections import Counter
            merged_atom_ids[i] = Counter(atom_votes).most_common(1)[0][0]
            merged_coeff_ids[i] = Counter(coeff_votes).most_common(1)[0][0]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ProgramBufferV1(
        atom_ids=merged_atom_ids,
        coeff_ids=merged_coeff_ids,
        residuals=torch.empty(0, dtype=torch.float16),
        has_residual=torch.zeros(N, dtype=torch.bool),
        shape=shape,
    )


def evolve_programs(
    weights: torch.Tensor,
    atom_table: AtomTableV1,
    coeffs: CoeffTable,
    population_size: int = 16,
    generations: int = 10,
    mutation_rate: float = 0.05,
    crossover_rate: float = 0.5,
    top_k: int = 4,
) -> Tuple[ProgramBufferV1, torch.Tensor]:
    """Evolve programs via genetic algorithm.
    
    Each individual is a program buffer. Fitness = negative MSE.
    Evolution operators: mutation (flip atom/coeff), crossover (swap segments).
    
    Args:
        weights: Target weights to approximate
        atom_table: Atom table
        coeffs: Coefficient table
        population_size: Number of individuals
        generations: Number of generations
        mutation_rate: Probability of mutation per weight
        crossover_rate: Probability of crossover
        top_k: Number of best individuals to keep
    
    Returns:
        (best_program, best_reconstruction)
    """
    device = weights.device
    N = weights.numel()
    K = atom_table.K_total
    C = coeffs.values.numel()
    
    # Initialize population with random programs
    population = []
    for _ in range(population_size):
        atom_ids = torch.randint(0, min(K, 256), (N,), dtype=torch.uint8, device=device)
        coeff_ids = torch.randint(0, min(C, 256), (N,), dtype=torch.uint8, device=device)
        prog = ProgramBufferV1(
            atom_ids=atom_ids,
            coeff_ids=coeff_ids,
            residuals=torch.empty(0, dtype=torch.float16, device=device),
            has_residual=torch.zeros(N, dtype=torch.bool, device=device),
            shape=weights.shape,
        )
        population.append(prog)
    
    def fitness(prog: ProgramBufferV1) -> float:
        """Negative MSE (higher is better)."""
        recon = wal_decode_v1(prog, atom_table, coeffs.values).flatten()
        mse = (weights.flatten() - recon).pow(2).mean().item()
        return -mse
    
    def mutate(prog: ProgramBufferV1) -> ProgramBufferV1:
        """Random mutation: flip some atom_ids and coeff_ids."""
        atom_ids = prog.atom_ids.clone()
        coeff_ids = prog.coeff_ids.clone()
        
        mask = torch.rand(N, device=device) < mutation_rate
        atom_ids[mask] = torch.randint(0, min(K, 256), (mask.sum(),), dtype=torch.uint8, device=device)
        
        mask = torch.rand(N, device=device) < mutation_rate
        coeff_ids[mask] = torch.randint(0, min(C, 256), (mask.sum(),), dtype=torch.uint8, device=device)
        
        return ProgramBufferV1(
            atom_ids=atom_ids,
            coeff_ids=coeff_ids,
            residuals=torch.empty(0, dtype=torch.float16, device=device),
            has_residual=torch.zeros(N, dtype=torch.bool, device=device),
            shape=weights.shape,
        )
    
    def crossover(p1: ProgramBufferV1, p2: ProgramBufferV1) -> ProgramBufferV1:
        """Single-point crossover."""
        point = torch.randint(0, N, (1,), device=device).item()
        atom_ids = torch.cat([p1.atom_ids[:point], p2.atom_ids[point:]])
        coeff_ids = torch.cat([p1.coeff_ids[:point], p2.coeff_ids[point:]])
        
        return ProgramBufferV1(
            atom_ids=atom_ids,
            coeff_ids=coeff_ids,
            residuals=torch.empty(0, dtype=torch.float16, device=device),
            has_residual=torch.zeros(N, dtype=torch.bool, device=device),
            shape=weights.shape,
        )
    
    best_fitness = float('-inf')
    best_prog = None
    
    for gen in range(generations):
        # Evaluate fitness
        scores = [(fitness(p), p) for p in population]
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Track best
        if scores[0][0] > best_fitness:
            best_fitness = scores[0][0]
            best_prog = scores[0][1]
        
        # Elitism: keep top_k
        elites = [p for _, p in scores[:top_k]]
        
        # Create next generation
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            if torch.rand(1).item() < crossover_rate and len(elites) >= 2:
                # Crossover
                idx1 = torch.randint(0, len(elites), (1,)).item()
                idx2 = torch.randint(0, len(elites), (1,)).item()
                p1, p2 = elites[idx1], elites[idx2]
                child = crossover(p1, p2)
            else:
                # Mutation
                parent = elites[torch.randint(0, len(elites), (1,)).item()]
                child = mutate(parent)
            
            new_population.append(child)
        
        population = new_population
        
        if gen % 3 == 0 or gen == generations - 1:
            print(f"    Gen {gen:3d}: best fitness={best_fitness:.8f} (MSE={-best_fitness:.8f})")
    
    recon = wal_decode_v1(best_prog, atom_table, coeffs.values)
    return best_prog, recon


def compute_program_gradient(
    prog: ProgramBufferV1,
    atom_table: AtomTableV1,
    coeffs: CoeffTable,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute gradient of MSE loss w.r.t. program atom_ids and coeff_ids.
    
    Note: atom_ids and coeff_ids are discrete (uint8), so true gradients
    don't exist. This function computes gradients w.r.t. the decoded
    weight and projects them to the nearest valid program change.
    
    Args:
        prog: Current program
        atom_table: Atom table
        coeffs: Coeff table
        target: Target weights
    
    Returns:
        (atom_grad, coeff_grad) — gradients for each position
    """
    recon = wal_decode_v1(prog, atom_table, coeffs.values)
    error = recon - target.flatten()
    
    flat_atoms = torch.tensor(
        [atom_table.resolve(i) for i in range(atom_table.K_total)],
        dtype=torch.float32,
        device=prog.atom_ids.device,
    )
    
    # Gradient w.r.t. atom choice: -error * coeff_value
    N = prog.N
    atom_grad = torch.zeros(N, dtype=torch.float32, device=prog.atom_ids.device)
    coeff_grad = torch.zeros(N, dtype=torch.float32, device=prog.atom_ids.device)
    
    for i in range(N):
        a = int(prog.atom_ids[i])
        c = int(prog.coeff_ids[i])
        atom_val = flat_atoms[a]
        coeff_val = coeffs.values[c]
        
        # d(MSE)/d(atom_id) ≈ d(MSE)/d(recon) * d(recon)/d(atom) * d(atom)/d(atom_id)
        # For nearest-neighbor, this is approximately:
        atom_grad[i] = error[i] * coeff_val
        coeff_grad[i] = error[i] * atom_val
    
    return atom_grad, coeff_grad
