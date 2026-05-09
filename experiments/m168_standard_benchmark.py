"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M168 — Standard WAL Benchmark Suite.

Defines the unified JSON output format for all WAL experiments.
This is a specification + validation tool, not a running benchmark.
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class WALBenchmarkResult:
    """Standardized result format for WAL experiments."""
    
    # Identification
    experiment: str           # e.g., "M153"
    model: str                # e.g., "meta-llama/Llama-3.1-8B"
    mode: str                 # e.g., "Transform-WAL", "Raw-WAL", "Hadamard-WAL"
    
    # Quality metrics
    ppl: Optional[float] = None
    mse: Optional[float] = None
    relmse: Optional[float] = None
    max_err: Optional[float] = None
    
    # Size metrics
    patch_size_bytes: Optional[int] = None
    patch_size_mb: Optional[float] = None
    lora_size_mb: Optional[float] = None
    
    # Diff locality
    diff_target: Optional[float] = None      # fraction of weights changed in target region
    diff_nontarget: Optional[float] = None   # fraction of weights changed outside target
    
    # Performance
    encode_time_sec: Optional[float] = None
    decode_time_sec: Optional[float] = None
    inference_speedup: Optional[float] = None
    
    # Program statistics
    atom_entropy: Optional[float] = None
    coeff_entropy: Optional[float] = None
    program_entropy: Optional[float] = None
    
    # Safety
    safety_score: Optional[str] = None       # "SAFE", "MODERATE", "RISKY", "DANGEROUS"
    spectral_norm: Optional[float] = None
    
    # Status
    status: str = "complete"                 # "complete", "partial", "negative", "failed"
    error: Optional[str] = None
    
    # Metadata
    notes: Optional[str] = None
    timestamp: Optional[str] = None


def validate_result(result: WALBenchmarkResult) -> List[str]:
    """Validate a benchmark result. Returns list of errors."""
    errors = []
    
    if not result.experiment:
        errors.append("experiment ID is required")
    
    if result.ppl is not None and result.ppl < 0:
        errors.append("PPL must be non-negative")
    
    if result.mse is not None and result.mse < 0:
        errors.append("MSE must be non-negative")
    
    if result.diff_target is not None and not 0 <= result.diff_target <= 1:
        errors.append("diff_target must be in [0, 1]")
    
    if result.diff_nontarget is not None and not 0 <= result.diff_nontarget <= 1:
        errors.append("diff_nontarget must be in [0, 1]")
    
    if result.safety_score and result.safety_score not in ["SAFE", "MODERATE", "RISKY", "DANGEROUS"]:
        errors.append("safety_score must be one of: SAFE, MODERATE, RISKY, DANGEROUS")
    
    if result.status not in ["complete", "partial", "negative", "failed"]:
        errors.append("status must be one of: complete, partial, negative, failed")
    
    return errors


def example_results():
    """Generate example benchmark results for documentation."""
    return [
        WALBenchmarkResult(
            experiment="M148",
            model="meta-llama/Llama-3.1-8B",
            mode="WAL v1 Spec Freeze",
            status="complete",
            notes="6 compatibility tests passed",
        ),
        WALBenchmarkResult(
            experiment="M154",
            model="meta-llama/Llama-3.1-8B",
            mode="Hadamard-WAL",
            mse=5.08e-07,
            relmse=1.2e-04,
            max_err=0.01,
            encode_time_sec=2.5,
            atom_entropy=0.95,
            status="complete",
            notes="Hadamard 2× better than Raw on small matrices",
        ),
        WALBenchmarkResult(
            experiment="M153",
            model="meta-llama/Llama-3.1-8B",
            mode="Transform-WAL",
            status="complete",
            notes="Full transform → WAL → inverse pipeline",
        ),
    ]


def main():
    print("=" * 60)
    print("M168 — Standard WAL Benchmark Suite")
    print("=" * 60)
    
    print("\n--- JSON Schema ---")
    schema = {
        "experiment": "string (required)",
        "model": "string (required)",
        "mode": "string (required)",
        "ppl": "float | null",
        "mse": "float | null",
        "relmse": "float | null",
        "max_err": "float | null",
        "patch_size_bytes": "int | null",
        "patch_size_mb": "float | null",
        "lora_size_mb": "float | null",
        "diff_target": "float [0,1] | null",
        "diff_nontarget": "float [0,1] | null",
        "encode_time_sec": "float | null",
        "decode_time_sec": "float | null",
        "inference_speedup": "float | null",
        "atom_entropy": "float | null",
        "coeff_entropy": "float | null",
        "program_entropy": "float | null",
        "safety_score": "string | null",
        "spectral_norm": "float | null",
        "status": "string: complete | partial | negative | failed",
        "error": "string | null",
        "notes": "string | null",
        "timestamp": "string | null",
    }
    print(json.dumps(schema, indent=2))
    
    print("\n--- Example Results ---")
    examples = example_results()
    for ex in examples:
        print(f"\n{ex.experiment}: {ex.mode}")
        data = asdict(ex)
        print(json.dumps(data, indent=2))
        
        errors = validate_result(ex)
        if errors:
            print(f"  Validation errors: {errors}")
        else:
            print("  ✅ Valid")
    
    # Save schema
    out_path = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m168_standard_benchmark.json'
    with open(out_path, 'w') as f:
        json.dump({
            'schema': schema,
            'examples': [asdict(ex) for ex in examples],
            'description': 'Standard WAL Benchmark Suite — unified result format',
        }, f, indent=2)
    print(f"\n✅ Benchmark spec saved to {out_path}")


if __name__ == "__main__":
    main()
