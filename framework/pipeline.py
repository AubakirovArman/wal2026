#!/usr/bin/env python3
"""WAL Framework — Full Pipeline: encode → optimize → export."""
from pathlib import Path


def run_pipeline(input_path: str, output_dir: str, K: int = 256, C: int = 16,
                 export_format: str = "none", device: str = "cuda",
                 trust_pickle: bool = False):
    """Run full WAL pipeline.
    
    Steps:
    1. Encode model to WAL
    2. Build hierarchical atoms (optional)
    3. Export to target format
    
    Args:
        input_path: Input model path
        output_dir: Output directory
        K: Number of atoms
        C: Number of coefficients
        export_format: 'onnx', 'hub', or 'none'
        device: Device
    """
    from .encode import encode_model
    from .hierarchy import build_hierarchy
    from .export import export_model
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    wal_dir = output_dir / "wal"
    
    # Step 1: Encode
    print("[Pipeline] Step 1/3: Encoding...")
    encode_model(input_path, str(wal_dir), K=K, C=C, device=device, trust_pickle=trust_pickle)
    
    # Step 2: Hierarchy (optional)
    print("[Pipeline] Step 2/3: Building hierarchy...")
    hier_dir = output_dir / "wal_hier"
    build_hierarchy(str(wal_dir), str(hier_dir), max_l1=64, trust_pickle=trust_pickle)
    
    # Step 3: Export
    if export_format != "none":
        print(f"[Pipeline] Step 3/3: Exporting to {export_format}...")
        export_path = output_dir / f"model.{export_format}"
        export_model(str(hier_dir), str(export_path), fmt=export_format, trust_pickle=trust_pickle)
    else:
        print("[Pipeline] Step 3/3: Skipping export.")
    
    print(f"[Pipeline] Complete. Output: {output_dir}")
