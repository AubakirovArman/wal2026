#!/usr/bin/env python3
"""WAL KV-cache compression.

Compress transformer KV-cache using WAL encoding.
KV-cache has different structure from weights:
- Keys: higher variance, needs more atoms (K=256, C=16)
- Values: lower variance, fewer atoms suffice (K=64, C=8)
- Temporal correlation exists but delta-encoding is toxic (error accumulation)

Usage:
    from wal.v1.kv_cache import WALKVCache
    
    # Wrap existing KV-cache
    wal_cache = WALKVCache.from_native(past_key_values, k_budget=(256, 16), v_budget=(64, 8))
    
    # Use in model
    outputs = model(input_ids=input_ids, past_key_values=wal_cache)
    
    # Access compression stats
    print(wal_cache.compression_ratio)
"""
import torch
from typing import Tuple, Optional, List
from .encoder import build_l0_atoms, build_coeff_table, wal_encode_v1
from .decoder import wal_decode_v1
from .isa import AtomTableV1, AtomDef, ProgramBufferV1, CoeffTable


class WALEncodedKV:
    """Single encoded KV layer."""
    
    def __init__(self, k_prog, k_table, k_coeffs, v_prog, v_table, v_coeffs, shape):
        self.k_prog = k_prog
        self.k_table = k_table
        self.k_coeffs = k_coeffs
        self.v_prog = v_prog
        self.v_table = v_table
        self.v_coeffs = v_coeffs
        self.shape = shape
        self._decoded_k = None
        self._decoded_v = None
    
    def decode(self, device=None):
        """Lazy decode K and V."""
        if self._decoded_k is None:
            k = wal_decode_v1(self.k_prog, self.k_table, self.k_coeffs).reshape(self.shape)
            v = wal_decode_v1(self.v_prog, self.v_table, self.v_coeffs).reshape(self.shape)
            if device is not None:
                k = k.to(device)
                v = v.to(device)
            self._decoded_k = k
            self._decoded_v = v
        return self._decoded_k, self._decoded_v
    
    def clear_cache(self):
        """Clear decoded cache to free memory."""
        self._decoded_k = None
        self._decoded_v = None
    
    @property
    def compressed_size_bits(self):
        """Estimated compressed size in bits."""
        N = self.k_prog.N
        k_bits = int(torch.ceil(torch.log2(torch.tensor(self.k_table.K_total))).item()) + \
                 int(torch.ceil(torch.log2(torch.tensor(self.k_coeffs.numel()))).item())
        v_bits = int(torch.ceil(torch.log2(torch.tensor(self.v_table.K_total))).item()) + \
                 int(torch.ceil(torch.log2(torch.tensor(self.v_coeffs.numel()))).item())
        return N * (k_bits + v_bits)
    
    @property
    def original_size_bits(self):
        """Original size at bf16."""
        return self.k_prog.N * 16 * 2  # K + V


class WALKVCache:
    """WAL-compressed KV-cache compatible with transformers.
    
    Args:
        encoded_layers: List of WALEncodedKV objects
        k_budget: (K_atoms, C_coeffs) for keys
        v_budget: (K_atoms, C_coeffs) for values
    """
    
    def __init__(self, encoded_layers: List[WALEncodedKV], k_budget=(256, 16), v_budget=(64, 8)):
        self.encoded_layers = encoded_layers
        self.k_budget = k_budget
        self.v_budget = v_budget
    
    @classmethod
    def from_native(cls, past_key_values, k_budget=(256, 16), v_budget=(64, 8)):
        """Encode native KV-cache (list of (k, v) tuples).
        
        Args:
            past_key_values: List of (key, value) tensors from model
            k_budget: (K, C) for keys
            v_budget: (K, C) for values
        """
        encoded = []
        for layer_idx, (k, v) in enumerate(past_key_values):
            k_enc, v_enc = cls._encode_layer(k, v, k_budget, v_budget)
            encoded.append(WALEncodedKV(k_enc[0], k_enc[1], k_enc[2],
                                         v_enc[0], v_enc[1], v_enc[2],
                                         k.shape))
        return cls(encoded, k_budget, v_budget)
    
    @staticmethod
    def _encode_layer(k, v, k_budget, v_budget):
        """Encode one KV layer."""
        k_atoms_cfg, k_coeffs_cfg = k_budget
        v_atoms_cfg, v_coeffs_cfg = v_budget
        
        # Encode K
        k_flat = k.float().cpu().reshape(-1)
        k_atoms = build_l0_atoms(k_flat, K=k_atoms_cfg, iters=3)
        k_coeffs = build_coeff_table(k_flat, k_atoms, C=k_coeffs_cfg, iters=3)
        k_prog, _ = wal_encode_v1(k_flat, k_atoms, k_coeffs)
        k_defs = [AtomDef(level=0, op="CONST") for _ in range(k_atoms.numel())]
        k_table = AtomTableV1(k_atoms, k_defs)
        
        # Encode V
        v_flat = v.float().cpu().reshape(-1)
        v_atoms = build_l0_atoms(v_flat, K=v_atoms_cfg, iters=3)
        v_coeffs = build_coeff_table(v_flat, v_atoms, C=v_coeffs_cfg, iters=3)
        v_prog, _ = wal_encode_v1(v_flat, v_atoms, v_coeffs)
        v_defs = [AtomDef(level=0, op="CONST") for _ in range(v_atoms.numel())]
        v_table = AtomTableV1(v_atoms, v_defs)
        
        return (k_prog, k_table, k_coeffs), (v_prog, v_table, v_coeffs)
    
    def to_native(self, device=None):
        """Decode to native KV-cache format."""
        return [layer.decode(device) for layer in self.encoded_layers]
    
    def to_dynamic_cache(self, device=None):
        """Decode to transformers DynamicCache."""
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError:
            raise ImportError("transformers >= 4.36 required for DynamicCache")
        
        cache = DynamicCache()
        for layer_idx, layer in enumerate(self.encoded_layers):
            k, v = layer.decode(device)
            cache.update(k, v, layer_idx)
        return cache
    
    def __len__(self):
        return len(self.encoded_layers)
    
    @property
    def compression_ratio(self):
        """Overall compression ratio."""
        original = sum(layer.original_size_bits for layer in self.encoded_layers)
        compressed = sum(layer.compressed_size_bits for layer in self.encoded_layers)
        return original / compressed if compressed > 0 else 1.0
    
    @property
    def memory_saved_mb(self):
        """Memory saved in MB."""
        original = sum(layer.original_size_bits for layer in self.encoded_layers) / 8 / 1024**2
        compressed = sum(layer.compressed_size_bits for layer in self.encoded_layers) / 8 / 1024**2
        return original - compressed
    
    def clear_decoded_cache(self):
        """Clear all decoded caches."""
        for layer in self.encoded_layers:
            layer.clear_cache()


def encode_kv_cache(past_key_values, k_budget=(256, 16), v_budget=(64, 8)):
    """Convenience function: encode native KV-cache.
    
    Args:
        past_key_values: List of (key, value) tensors
        k_budget: (K_atoms, C_coeffs) for keys
        v_budget: (K_atoms, C_coeffs) for values
    
    Returns:
        WALKVCache object
    """
    return WALKVCache.from_native(past_key_values, k_budget, v_budget)


def decode_kv_cache(wal_cache, device=None):
    """Convenience function: decode WAL KV-cache to native format.
    
    Args:
        wal_cache: WALKVCache object
        device: Target device
    
    Returns:
        List of (key, value) tensors
    """
    return wal_cache.to_native(device)
