#!/usr/bin/env python3
"""WAL Hardware Backends — Phase 9.

Supported backends:
- cpu: CPU SIMD via NumPy (always available)
- cuda: NVIDIA GPU via PyTorch CUDA
- mps: Apple Silicon via Metal Performance Shaders
- rocm: AMD GPU via ROCm/HIP
- webgpu: Browser/native WebGPU (requires wgpu-py)
"""
from .base import WALBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .mps import MPSBackend
from .rocm import ROCmBackend
from .webgpu import WebGPUBackend

# Registry of all backends
_BACKENDS: dict = {}


def _register_backends():
    """Register all backend instances."""
    global _BACKENDS
    for cls in [CPUBackend, CUDABackend, MPSBackend, ROCmBackend, WebGPUBackend]:
        try:
            inst = cls()
            _BACKENDS[inst.name] = inst
        except Exception:
            pass


# Auto-register on import
_register_backends()


def get_backend(name: str) -> WALBackend:
    """Get a backend by name.
    
    Args:
        name: Backend name ('cpu', 'cuda', 'mps', 'rocm', 'webgpu')
    
    Returns:
        WALBackend instance
    
    Raises:
        KeyError: If backend not found
    """
    if name not in _BACKENDS:
        raise KeyError(f"Backend '{name}' not found. Available: {list_backends()}")
    return _BACKENDS[name]


def list_backends() -> list:
    """List all registered backend names."""
    return sorted(_BACKENDS.keys())


def available_backends() -> list:
    """List backends that are available on the current system."""
    return sorted([name for name, backend in _BACKENDS.items() if backend.is_available])


def select_best_backend() -> WALBackend:
    """Select the best available backend.
    
    Priority: cuda > rocm > mps > cpu > webgpu
    """
    priority = ['cuda', 'rocm', 'mps', 'cpu', 'webgpu']
    for name in priority:
        if name in _BACKENDS and _BACKENDS[name].is_available:
            return _BACKENDS[name]
    raise RuntimeError("No WAL backend available")


__all__ = [
    "WALBackend",
    "CPUBackend",
    "CUDABackend",
    "MPSBackend",
    "ROCmBackend",
    "WebGPUBackend",
    "get_backend",
    "list_backends",
    "available_backends",
    "select_best_backend",
]
