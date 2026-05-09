#!/usr/bin/env python3
"""WAL Streaming Encoder.

Encode large models without loading them entirely into memory.
Processes model shards one at a time, with resume support.

Usage:
    from wal.v1.streaming_encoder import StreamingWALEncoder
    
    encoder = StreamingWALEncoder(
        repo_id="unsloth/Llama-3.3-70B-Instruct",
        output_dir="./wal_encoded/",
        K=256, C=16,
        device="cuda:0",
    )
    encoder.encode_all()
    
Low-memory mode (consumer GPUs):
    encoder = StreamingWALEncoder(
        repo_id="...",
        output_dir="./wal_encoded/",
        K=256, C=16,
        device="cuda:0",
        low_memory=True,  # Encode one tensor at a time
    )
"""
import json
import time
import gc
from pathlib import Path
from typing import Optional, Dict, List
import torch

from .nn import encode_linear_weight


class StreamingWALEncoder:
    """Stream WAL encoding for large models."""
    
    def __init__(
        self,
        repo_id: str,
        output_dir: str,
        K: int = 256,
        C: int = 16,
        device: str = "cuda:0",
        low_memory: bool = False,
        progress_file: Optional[str] = None,
    ):
        """
        Args:
            repo_id: HuggingFace model repo ID
            output_dir: Where to save encoded tensors
            K: Number of atoms
            C: Number of coefficients
            device: Device for encoding
            low_memory: If True, encode one tensor at a time (saves GPU memory)
            progress_file: Path to progress JSON (default: output_dir/progress.json)
        """
        self.repo_id = repo_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.K = K
        self.C = C
        self.device = torch.device(device)
        self.low_memory = low_memory
        self.progress_file = Path(progress_file) if progress_file else self.output_dir / "progress.json"
        
        self._index = None
        self._shards = None
        self._progress = None
    
    def _load_index(self) -> Dict:
        """Load model.safetensors.index.json from HF Hub."""
        if self._index is not None:
            return self._index
        
        from huggingface_hub import hf_hub_download
        idx_path = hf_hub_download(self.repo_id, "model.safetensors.index.json")
        with open(idx_path) as f:
            self._index = json.load(f)
        return self._index
    
    def _get_shards(self) -> List[str]:
        """Get sorted list of shard filenames."""
        if self._shards is not None:
            return self._shards
        
        index = self._load_index()
        weight_map = index["weight_map"]
        shard_set = set(weight_map.values())
        self._shards = sorted(shard_set)
        return self._shards
    
    def _load_progress(self) -> Dict:
        """Load progress tracking."""
        if self._progress is not None:
            return self._progress
        
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                self._progress = json.load(f)
        else:
            self._progress = {}
        return self._progress
    
    def _save_progress(self):
        """Save progress tracking."""
        with open(self.progress_file, "w") as f:
            json.dump(self._progress, f, indent=2)
    
    def _encode_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Encode a single tensor and save to disk.
        
        Returns:
            True if encoded successfully, False otherwise
        """
        if len(tensor.shape) != 2 or "weight" not in name or "layernorm" in name:
            return False
        
        wal_path = self.output_dir / f"{name.replace('.', '_')}.wal.pt"
        if wal_path.exists():
            return False  # Already encoded
        
        try:
            # Move to GPU, encode, move result to CPU
            tensor_gpu = tensor.to(self.device)
            wal_param = encode_linear_weight(tensor_gpu, K=self.K, C=self.C)
            
            # Move to CPU for saving
            prog_cpu = wal_param.prog
            prog_cpu.atom_ids = prog_cpu.atom_ids.cpu()
            prog_cpu.coeff_ids = prog_cpu.coeff_ids.cpu()
            atoms_cpu = wal_param.atom_table.base_atoms.cpu()
            coeffs_cpu = wal_param.coeffs.values.cpu()
            
            torch.save({
                "shape": wal_param.shape,
                "prog": prog_cpu,
                "atoms": atoms_cpu,
                "coeffs": coeffs_cpu,
                "dtype": str(wal_param.dtype),
            }, wal_path)
            
            # Clean up
            del tensor_gpu
            del wal_param
            if self.low_memory:
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            print(f"      ERROR encoding {name}: {e}")
            return False
    
    def encode_shard(self, shard_name: str) -> Dict:
        """Encode all tensors in one shard.
        
        Returns:
            Dict with 'encoded', 'skipped', 'time'
        """
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        
        if shard_name in self._load_progress():
            print(f"  Shard {shard_name} — ALREADY DONE, skipping")
            return {"encoded": 0, "skipped": 0, "time": 0}
        
        start = time.time()
        print(f"\n  Shard: {shard_name}")
        
        # Download and load
        print(f"    Downloading...")
        shard_path = hf_hub_download(self.repo_id, shard_name)
        
        print(f"    Loading...")
        state_dict = load_file(shard_path)
        
        # Encode
        print(f"    Encoding on {self.device}...")
        encoded = 0
        skipped = 0
        
        for tensor_name, tensor in state_dict.items():
            success = self._encode_tensor(tensor, tensor_name)
            if success:
                encoded += 1
            else:
                skipped += 1
        
        elapsed = time.time() - start
        print(f"    Encoded: {encoded}, Skipped: {skipped}, Time: {elapsed:.1f}s")
        print(f"    GPU memory: {torch.cuda.memory_allocated(self.device) / 1024**3:.1f} GB")
        
        # Save progress
        self._progress[shard_name] = {"encoded": encoded, "skipped": skipped, "time": elapsed}
        self._save_progress()
        
        # Clean up
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        
        return {"encoded": encoded, "skipped": skipped, "time": elapsed}
    
    def encode_all(self, max_shards: Optional[int] = None):
        """Encode all shards.
        
        Args:
            max_shards: If set, only encode this many shards (for testing)
        """
        shards = self._get_shards()
        if max_shards:
            shards = shards[:max_shards]
        
        print(f"Encoding {len(shards)} shards from {self.repo_id}")
        print(f"Output: {self.output_dir}")
        print(f"K={self.K}, C={self.C}, device={self.device}, low_memory={self.low_memory}")
        
        total_start = time.time()
        total_encoded = 0
        total_skipped = 0
        
        for i, shard_name in enumerate(shards, 1):
            result = self.encode_shard(shard_name)
            total_encoded += result["encoded"]
            total_skipped += result["skipped"]
            
            # Progress report
            if i % 5 == 0 or i == len(shards):
                elapsed = time.time() - total_start
                avg_per_shard = elapsed / i
                remaining = avg_per_shard * (len(shards) - i)
                print(f"\n  Progress: {i}/{len(shards)} shards, "
                      f"{total_encoded} encoded, {elapsed/60:.1f}m elapsed, "
                      f"~{remaining/60:.1f}m remaining")
        
        total_elapsed = time.time() - total_start
        print(f"\nDone: {total_encoded} encoded, {total_skipped} skipped in {total_elapsed/60:.1f}m")
    
    @property
    def stats(self) -> Dict:
        """Current encoding statistics."""
        progress = self._load_progress()
        encoded = sum(p["encoded"] for p in progress.values())
        skipped = sum(p["skipped"] for p in progress.values())
        total_time = sum(p["time"] for p in progress.values())
        
        wal_files = list(self.output_dir.glob("*.wal.pt"))
        total_size = sum(f.stat().st_size for f in wal_files)
        
        return {
            "shards_done": len(progress),
            "shards_total": len(self._get_shards()),
            "tensors_encoded": encoded,
            "tensors_skipped": skipped,
            "total_time_s": total_time,
            "output_size_mb": total_size / 1024**2,
        }
