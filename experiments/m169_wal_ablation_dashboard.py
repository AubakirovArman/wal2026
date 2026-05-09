"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M169 — WAL Ablation Dashboard.

Aggregates all experimental results into a unified comparison table.
Requires prior experiments to have produced .json output files.
"""
import json
import glob
import os


def load_results():
    """Load all experiment JSON files."""
    results = {}
    pattern = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m*_*.json'
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                data = json.load(f)
            basename = os.path.basename(path)
            results[basename] = data
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")
    return results


def extract_metrics(data, exp_name):
    """Extract key metrics from experiment data."""
    metrics = {
        'experiment': exp_name,
        'mode': 'Unknown',
        'mse': None,
        'relmse': None,
        'ppl': None,
        'patch_size_mb': None,
        'diff_target': None,
        'diff_nontarget': None,
        'encode_time': None,
        'atom_entropy': None,
        'status': 'unknown',
    }
    
    if 'm142' in exp_name.lower() or 'transform_wal_probe' in exp_name.lower():
        metrics['mode'] = 'Transform-WAL Probe'
        # Data is list of per-layer results
        if isinstance(data, list) and len(data) > 0:
            mses = [d.get('mse', 0) for d in data if 'mse' in d]
            if mses:
                metrics['mse'] = sum(mses) / len(mses)
        metrics['status'] = 'complete'
    
    elif 'm143' in exp_name.lower() or 'wave_atom' in exp_name.lower():
        metrics['mode'] = 'Wave-Atom ISA'
        metrics['status'] = 'negative'
    
    elif 'm144' in exp_name.lower() or 'graph_wal' in exp_name.lower():
        metrics['mode'] = 'Graph-WAL'
        metrics['status'] = 'negative'
    
    elif 'm145' in exp_name.lower() or 'fingerprint' in exp_name.lower():
        metrics['mode'] = 'Fingerprints'
        metrics['status'] = 'partial'
    
    elif 'm146' in exp_name.lower() or 'cross_model' in exp_name.lower():
        metrics['mode'] = 'Cross-Model Vocab'
        metrics['status'] = 'partial'
    
    elif 'm147' in exp_name.lower() or 'friendly_training' in exp_name.lower():
        metrics['mode'] = 'WAL-Friendly Training'
        metrics['status'] = 'negative'
    
    elif 'm149' in exp_name.lower() or 'frozen_vocab' in exp_name.lower():
        metrics['mode'] = 'Frozen Vocab PPL Matrix'
        if isinstance(data, dict):
            metrics['mse'] = data.get('avg_rebuilt_mse')
            metrics['diff_target'] = data.get('avg_rebuilt_patch_diff')
            metrics['diff_nontarget'] = data.get('avg_frozen_patch_diff')
        metrics['status'] = 'complete'
    
    elif 'm153' in exp_name.lower() or 'transform_wal_encoder' in exp_name.lower():
        metrics['mode'] = 'Transform-WAL Encoder'
        if isinstance(data, list) and len(data) > 0:
            mses = [d.get('mse', 0) for d in data if 'mse' in d]
            if mses:
                metrics['mse'] = sum(mses) / len(mses)
        metrics['status'] = 'complete'
    
    elif 'm154' in exp_name.lower() or 'hadamard' in exp_name.lower():
        metrics['mode'] = 'Hadamard-WAL'
        if isinstance(data, dict):
            wal_comp = data.get('wal_comparison', [])
            if wal_comp:
                mses = [r.get('mse_hadamard', 0) for r in wal_comp]
                metrics['mse'] = sum(mses) / len(mses)
        metrics['status'] = 'complete'
    
    return metrics


def main():
    print("=" * 80)
    print("M169 — WAL Ablation Dashboard")
    print("=" * 80)
    
    results = load_results()
    print(f"\nLoaded {len(results)} experiment result files")
    
    # Build dashboard
    rows = []
    for name, data in sorted(results.items()):
        metrics = extract_metrics(data, name)
        rows.append(metrics)
    
    # Print table
    print("\n" + "=" * 80)
    print(f"{'Experiment':<40} {'Mode':<25} {'MSE':>12} {'Status':>10}")
    print("=" * 80)
    for r in rows:
        mse_str = f"{r['mse']:.2e}" if r['mse'] is not None else "N/A"
        print(f"{r['experiment']:<40} {r['mode']:<25} {mse_str:>12} {r['status']:>10}")
    
    # Save full dashboard
    out_path = '/mnt/hf_model_weights/arman/3bit/wal/experiments/m169_wal_ablation_dashboard.json'
    with open(out_path, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\n✅ Dashboard saved to {out_path}")


if __name__ == "__main__":
    main()
