#!/usr/bin/env python3
"""WAL Hugging Face Hub Integration.

Phase 11a: Push/pull WAL-encoded models to/from the Hugging Face Hub.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

from .isa import ProgramBufferV1, AtomTableV1, CoeffTable
from .format import serialize_wal_v1, deserialize_wal_v1
from .nn import WALLinear, WALCachedLinear, WALParameter


class WALModelCard:
    """Metadata card for a WAL-encoded model."""
    
    def __init__(
        self,
        base_model: str,
        wal_version: str = "1.0",
        encoder_config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        adapters: Optional[list] = None,
    ):
        self.base_model = base_model
        self.wal_version = wal_version
        self.encoder_config = encoder_config or {}
        self.metrics = metrics or {}
        self.adapters = adapters or []
    
    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "wal_version": self.wal_version,
            "encoder_config": self.encoder_config,
            "metrics": self.metrics,
            "adapters": self.adapters,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "WALModelCard":
        return cls(
            base_model=d.get("base_model", ""),
            wal_version=d.get("wal_version", "1.0"),
            encoder_config=d.get("encoder_config", {}),
            metrics=d.get("metrics", {}),
            adapters=d.get("adapters", []),
        )


def extract_wal_state_dict(model: nn.Module) -> Dict[str, Any]:
    """Extract a complete WAL state dict from a model.
    
    Returns dict with:
    - 'wal_blobs': {layer_name: serialized_bytes}
    - 'biases': {layer_name: tensor}
    - 'non_wal': {param_name: tensor} for non-WAL parameters
    - 'metadata': model architecture info
    """
    wal_blobs = {}
    biases = {}
    non_wal = {}
    wal_layers = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (WALLinear, WALCachedLinear)):
            blob = serialize_wal_v1(
                module.wal_weight.prog,
                module.wal_weight.atom_table,
                module.wal_weight.coeffs,
            )
            wal_blobs[f"{name}.wal_weight"] = blob
            wal_layers[name] = {
                "shape": list(module.wal_weight.shape),
                "dtype": str(module.wal_weight.dtype).replace("torch.", ""),
                "K0": module.wal_weight.atom_table.K0,
                "K_total": module.wal_weight.atom_table.K_total,
                "C": module.wal_weight.coeffs.values.numel(),
            }
            if module.bias is not None:
                biases[f"{name}.bias"] = module.bias.cpu()
        
    # Non-WAL parameters
    for name, param in model.named_parameters():
        if not any(name.startswith(wl) for wl in wal_layers):
            non_wal[name] = param.data.cpu()
    
    for name, buf in model.named_buffers():
        if not any(name.startswith(wl) for wl in wal_layers) and name not in non_wal:
            non_wal[name] = buf.data.cpu()
    
    return {
        "wal_blobs": wal_blobs,
        "biases": biases,
        "non_wal": non_wal,
        "wal_layers": wal_layers,
    }


def load_wal_state_dict(
    state_dict: Dict[str, Any],
    target_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Load a WAL state dict. If target_model provided, reconstruct in-place.
    
    Returns:
        If target_model is None: dict of reconstructed components
        If target_model is provided: loaded model
    """
    from .format import deserialize_wal_v1
    
    wal_blobs = state_dict.get("wal_blobs", {})
    biases = state_dict.get("biases", {})
    non_wal = state_dict.get("non_wal", {})
    wal_layers = state_dict.get("wal_layers", {})
    
    reconstructed = {}
    
    for name, blob in wal_blobs.items():
        prog, atom_table, coeffs, meta = deserialize_wal_v1(blob)
        layer_name = name.replace(".wal_weight", "")
        layer_info = wal_layers.get(layer_name, {})
        shape = tuple(layer_info.get("shape", prog.shape))
        dtype = getattr(torch, layer_info.get("dtype", "float32"))
        
        wal_param = WALParameter(
            prog=prog,
            atom_table=atom_table,
            coeffs=coeffs,
            shape=shape,
            dtype=dtype,
        )
        reconstructed[name] = wal_param
    
    for name, bias in biases.items():
        reconstructed[name] = bias
    
    for name, param in non_wal.items():
        reconstructed[name] = param
    
    if target_model is not None:
        # Try to load into model
        for name, module in target_model.named_modules():
            if isinstance(module, (WALLinear, WALCachedLinear)):
                wal_key = f"{name}.wal_weight"
                if wal_key in reconstructed:
                    module.wal_weight = reconstructed[wal_key]
                bias_key = f"{name}.bias"
                if bias_key in reconstructed:
                    module.bias = reconstructed[bias_key].to(module.wal_weight.prog.atom_ids.device)
        
        # Load non-WAL parameters
        missing, unexpected = target_model.load_state_dict(
            {k: v for k, v in non_wal.items()}, strict=False
        )
        return target_model
    
    return reconstructed


def push_wal_model(
    model: nn.Module,
    repo_id: str,
    card: WALModelCard,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload WAL-encoded model",
) -> str:
    """Push a WAL-encoded model to the Hugging Face Hub.
    
    Args:
        model: PyTorch model with WAL layers
        repo_id: HF Hub repo ID (e.g., "username/model-name")
        card: Model metadata card
        token: HF API token
        private: Whether repo should be private
        commit_message: Git commit message
    
    Returns:
        URL of the uploaded model
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    api = HfApi(token=token)
    
    # Create repo if needed
    try:
        create_repo(repo_id, private=private, token=token, exist_ok=True)
    except Exception:
        pass
    
    # Extract WAL state
    wal_state = extract_wal_state_dict(model)
    
    # Prepare upload directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save WAL blobs
        wal_dir = tmpdir / "wal_weights"
        wal_dir.mkdir()
        for name, blob in wal_state["wal_blobs"].items():
            safe_name = name.replace(".", "_") + ".wal"
            (wal_dir / safe_name).write_bytes(blob)
        
        # Save biases
        if wal_state["biases"]:
            bias_path = tmpdir / "biases.safetensors"
            from safetensors.torch import save_file
            save_file(wal_state["biases"], str(bias_path))
        
        # Save non-WAL params
        if wal_state["non_wal"]:
            non_wal_path = tmpdir / "non_wal.safetensors"
            from safetensors.torch import save_file
            save_file(wal_state["non_wal"], str(non_wal_path))
        
        # Save metadata
        metadata = {
            "card": card.to_dict(),
            "wal_layers": wal_state["wal_layers"],
        }
        (tmpdir / "wal_config.json").write_text(json.dumps(metadata, indent=2))
        
        # Upload
        api.upload_folder(
            folder_path=str(tmpdir),
            repo_id=repo_id,
            commit_message=commit_message,
        )
    
    return f"https://huggingface.co/{repo_id}"


def pull_wal_model(
    repo_id: str,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Pull a WAL-encoded model from the Hugging Face Hub.
    
    Args:
        repo_id: HF Hub repo ID
        token: HF API token
        cache_dir: Local cache directory
    
    Returns:
        Dict with:
        - 'wal_params': reconstructed WALParameters
        - 'biases': bias tensors
        - 'non_wal': non-WAL parameters
        - 'card': WALModelCard
        - 'wal_layers': layer metadata
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")
    
    # Download files
    config_path = hf_hub_download(
        repo_id=repo_id, filename="wal_config.json",
        token=token, cache_dir=cache_dir,
    )
    config = json.loads(Path(config_path).read_text())
    card = WALModelCard.from_dict(config["card"])
    wal_layers = config.get("wal_layers", {})
    
    result = {
        "card": card,
        "wal_layers": wal_layers,
        "wal_params": {},
        "biases": {},
        "non_wal": {},
    }
    
    # Download and deserialize WAL blobs
    from .format import deserialize_wal_v1
    
    wal_dir = Path(config_path).parent / "wal_weights"
    if wal_dir.exists():
        for wal_file in wal_dir.glob("*.wal"):
            blob = wal_file.read_bytes()
            name = wal_file.stem.replace("_", ".")
            prog, atom_table, coeffs, meta = deserialize_wal_v1(blob)
            layer_name = name.replace(".wal_weight", "")
            layer_info = wal_layers.get(layer_name, {})
            shape = tuple(layer_info.get("shape", prog.shape))
            dtype = getattr(torch, layer_info.get("dtype", "float32"))
            
            wal_param = WALParameter(
                prog=prog,
                atom_table=atom_table,
                coeffs=coeffs,
                shape=shape,
                dtype=dtype,
            )
            result["wal_params"][name] = wal_param
    
    # Download biases
    try:
        bias_path = hf_hub_download(
            repo_id=repo_id, filename="biases.safetensors",
            token=token, cache_dir=cache_dir,
        )
        from safetensors.torch import load_file
        result["biases"] = load_file(bias_path)
    except Exception:
        pass
    
    # Download non-WAL params
    try:
        non_wal_path = hf_hub_download(
            repo_id=repo_id, filename="non_wal.safetensors",
            token=token, cache_dir=cache_dir,
        )
        from safetensors.torch import load_file
        result["non_wal"] = load_file(non_wal_path)
    except Exception:
        pass
    
    return result
