#!/usr/bin/env python3
"""M159 — Transform Metadata Cost.

Compares metadata overhead for different transforms.
"""
import json


def main():
    print("=" * 60)
    print("M159 — Transform Metadata Cost")
    print("=" * 60)
    
    # Representative dimensions from Llama-3.1-8B
    modules = [
        ('q_proj', 4096, 4096),
        ('k_proj', 1024, 4096),
        ('v_proj', 1024, 4096),
        ('o_proj', 4096, 4096),
        ('gate_proj', 14336, 4096),
        ('up_proj', 14336, 4096),
        ('down_proj', 4096, 14336),
    ]
    
    n_layers = 32
    
    transforms = [
        ('Raw', 'none'),
        ('Hadamard', 'none'),
        ('DCT', 'none'),
        ('RandOrth (full Q)', 'full'),
        ('RandOrth (seed)', 'seed'),
        ('RandOrth (shared per module)', 'shared_module'),
    ]
    
    results = []
    
    for tname, storage_type in transforms:
        total_bytes = 0
        per_layer = {}
        
        for mname, out_d, in_d in modules:
            if storage_type == 'none':
                meta = 0
            elif storage_type == 'full':
                # Q_out (out_d × out_d) + Q_in (in_d × in_d), float32
                meta = (out_d * out_d + in_d * in_d) * 4
            elif storage_type == 'seed':
                # 2 seeds (Q_out, Q_in), 4 bytes each
                meta = 8
            elif storage_type == 'shared_module':
                # One Q per module type across all layers
                meta = (out_d * out_d + in_d * in_d) * 4 / n_layers
            
            per_layer[mname] = meta
            total_bytes += meta
        
        results.append({
            'transform': tname,
            'storage_type': storage_type,
            'per_layer_bytes': per_layer,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / 1e6,
        })
        
        print(f"\n{tname}:")
        print(f"  Total per layer: {total_bytes / 1e6:.3f} MB")
        for mname, meta in per_layer.items():
            print(f"    {mname}: {meta / 1e3:.1f} KB")
    
    # Full model cost
    print("\n" + "=" * 60)
    print("FULL MODEL COST (32 layers)")
    print("=" * 60)
    for r in results:
        full_mb = r['total_mb']
        print(f"  {r['transform']:30s}: {full_mb * n_layers:8.2f} MB")
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m159_transform_metadata_cost.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
