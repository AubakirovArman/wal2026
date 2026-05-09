"""
M224 — WAL Probe / Model Forensics

Hypothesis 7: Even if WAL doesn't improve quality, it provides structured audit:
- Which modules changed
- PPL growth over edits
- Which edits were forgotten
- Conflict graph
- Risk score

This is observability: "Datadog for model weights"
"""

import os, sys, json
import numpy as np
from collections import defaultdict

os.environ["HF_HOME"] = "/mnt/hf_model_weights"

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

def analyze_sequential_edits():
    """Analyze M215 sequential edit data."""
    data = load_json("experiments/m215_results.json")
    if not data:
        return None
    
    edits = data.get("edits", [])
    
    analysis = {
        "n_edits": len(edits),
        "total_facts": data.get("final_total_facts", 50),
        "final_cumulative_survival": data.get("final_cumulative_survival", 0),
        "ppl_drift_per_edit": [],
        "survival_per_edit": [],
        "batch_degradation": [],
        "risk_score": 0,
    }
    
    for edit in edits:
        analysis["ppl_drift_per_edit"].append({
            "edit": edit["edit"],
            "reencode_ppl": edit["reencode_ppl"],
            "ppl_delta": edit["reencode_ppl"] - data.get("baseline_ppl", 4.2744),
        })
        analysis["survival_per_edit"].append({
            "edit": edit["edit"],
            "batch_survival": edit["reencode_batch_survival"],
            "cumulative_survival": edit["reencode_cumulative_survival"],
        })
    
    # Calculate drift rate
    if len(edits) >= 2:
        first_ppl = edits[0]["reencode_ppl"]
        last_ppl = edits[-1]["reencode_ppl"]
        analysis["total_ppl_drift"] = last_ppl - first_ppl
        analysis["drift_per_edit"] = (last_ppl - first_ppl) / len(edits)
    
    # Risk score: combine PPL drift + survival degradation
    baseline_ppl = data.get("baseline_ppl", 4.2744)
    final_ppl = edits[-1]["reencode_ppl"] if edits else baseline_ppl
    ppl_risk = (final_ppl - baseline_ppl) / baseline_ppl * 100  # % drift
    
    max_cumul = max(e["reencode_cumulative_survival"] for e in edits) if edits else 0
    final_cumul = edits[-1]["reencode_cumulative_survival"] if edits else 0
    survival_risk = (max_cumul - final_cumul) / max(max_cumul, 1) * 100 if max_cumul > 0 else 0
    
    analysis["risk_score"] = ppl_risk + survival_risk
    analysis["ppl_risk"] = ppl_risk
    analysis["survival_risk"] = survival_risk
    
    return analysis

def analyze_checkpoint_diff():
    """Analyze M216 checkpoint diff data."""
    data = load_json("experiments/m216_results.json")
    if not data:
        return None
    
    edit_diffs = data.get("edit_diffs", [])
    cumulative = data.get("cumulative", {})
    
    analysis = {
        "n_edits": len(edit_diffs),
        "modules_changed_per_edit": [e["diff"]["changed_modules"] for e in edit_diffs],
        "params_changed_per_edit": [e["binary"]["changed_params"] for e in edit_diffs],
        "diff_size_mb_per_edit": [e["binary"]["estimated_diff_mb"] for e in edit_diffs],
        "cumulative_modules_changed": cumulative.get("diff", {}).get("changed_modules", 0),
        "cumulative_params_changed": cumulative.get("binary", {}).get("changed_params", 0),
        "cumulative_diff_size_mb": cumulative.get("binary", {}).get("estimated_diff_mb", 0),
    }
    
    return analysis

def analyze_hard_facts():
    """Analyze M217 hard fact data."""
    data = load_json("experiments/m217_results.json")
    if not data:
        return None
    
    results = data if isinstance(data, list) else []
    
    analysis = {
        "n_configs": len(results),
        "all_zero_survival": all(r.get("reencode_survival", 0) == 0 for r in results),
        "best_ppl_config": None,
        "worst_ppl_config": None,
        "avg_ppl_delta": np.mean([r.get("reencode_ppl", 4.2744) - 4.2744 for r in results]) if results else 0,
    }
    
    if results:
        best = min(results, key=lambda r: r.get("reencode_ppl", 999))
        worst = max(results, key=lambda r: r.get("reencode_ppl", 0))
        analysis["best_ppl_config"] = best.get("config", "N/A")
        analysis["worst_ppl_config"] = worst.get("config", "N/A")
        analysis["best_ppl_delta"] = best.get("reencode_ppl", 4.2744) - 4.2744
        analysis["worst_ppl_delta"] = worst.get("reencode_ppl", 4.2744) - 4.2744
    
    return analysis

def generate_forensics_report():
    """Generate comprehensive forensics report."""
    
    seq = analyze_sequential_edits()
    diff = analyze_checkpoint_diff()
    hard = analyze_hard_facts()
    
    report = {
        "generated": __import__("datetime").datetime.now().isoformat(),
        "sequential_edits": seq,
        "checkpoint_diff": diff,
        "hard_facts": hard,
        "alerts": [],
        "recommendations": [],
    }
    
    # Generate alerts
    if seq:
        if seq["risk_score"] > 20:
            report["alerts"].append(f"🚨 HIGH RISK: PPL drift {seq['ppl_risk']:.1f}%, survival degradation {seq['survival_risk']:.1f}%")
        if seq.get("drift_per_edit", 0) > 0.05:
            report["alerts"].append(f"⚠️ PPL drift per edit: {seq['drift_per_edit']:.4f} (>0.05 threshold)")
    
    if diff:
        if diff.get("cumulative_params_changed", 0) > 5_000_000_000:
            report["alerts"].append(f"🚨 MASSIVE DIFF: {diff['cumulative_params_changed']/1e9:.1f}B params changed")
    
    if hard:
        if hard.get("all_zero_survival", False):
            report["alerts"].append("🚨 HARD FACTS BLOCKER: All configs failed (0/3 survival)")
    
    # Generate recommendations
    if seq and seq.get("drift_per_edit", 0) > 0.05:
        report["recommendations"].append("Implement edit refresh scheduler for degraded batches")
    
    if hard and hard.get("all_zero_survival", False):
        report["recommendations"].append("Use contrastive loss / suppression for hard facts")
        report["recommendations"].append("Route hard facts to retrieval tier instead of weights")
    
    if diff and diff.get("cumulative_diff_size_mb", 0) > 20_000:
        report["recommendations"].append("Store edit recipes, not checkpoint diffs (build artifact approach)")
    
    return report

def main():
    print("=" * 60, flush=True)
    print("M224 — WAL Probe / Model Forensics", flush=True)
    print("=" * 60, flush=True)
    
    report = generate_forensics_report()
    
    print("\n📊 SEQUENTIAL EDIT ANALYSIS", flush=True)
    print("-" * 40, flush=True)
    seq = report["sequential_edits"]
    if seq:
        print(f"  Total edits: {seq['n_edits']}", flush=True)
        print(f"  Total facts: {seq['total_facts']}", flush=True)
        print(f"  Final survival: {seq['final_cumulative_survival']}/{seq['total_facts']}", flush=True)
        print(f"  Total PPL drift: +{seq.get('total_ppl_drift', 0):.4f}", flush=True)
        print(f"  Drift per edit: +{seq.get('drift_per_edit', 0):.4f}", flush=True)
        print(f"  Risk score: {seq['risk_score']:.2f}", flush=True)
    else:
        print("  No data available", flush=True)
    
    print("\n📊 CHECKPOINT DIFF ANALYSIS", flush=True)
    print("-" * 40, flush=True)
    diff = report["checkpoint_diff"]
    if diff:
        print(f"  Edits analyzed: {diff['n_edits']}", flush=True)
        print(f"  Cumulative modules changed: {diff['cumulative_modules_changed']}", flush=True)
        print(f"  Cumulative params changed: {diff['cumulative_params_changed']/1e9:.1f}B", flush=True)
        print(f"  Cumulative diff size: {diff['cumulative_diff_size_mb']:.1f} MB", flush=True)
    else:
        print("  No data available", flush=True)
    
    print("\n📊 HARD FACT ANALYSIS", flush=True)
    print("-" * 40, flush=True)
    hard = report["hard_facts"]
    if hard:
        print(f"  Configs tested: {hard['n_configs']}", flush=True)
        print(f"  All failed: {hard['all_zero_survival']}", flush=True)
        print(f"  Avg PPL delta: +{hard['avg_ppl_delta']:.4f}", flush=True)
        if hard.get("best_ppl_config"):
            print(f"  Best config: {hard['best_ppl_config']} (ΔPPL={hard['best_ppl_delta']:+.4f})", flush=True)
        if hard.get("worst_ppl_config"):
            print(f"  Worst config: {hard['worst_ppl_config']} (ΔPPL={hard['worst_ppl_delta']:+.4f})", flush=True)
    else:
        print("  No data available", flush=True)
    
    print("\n🚨 ALERTS", flush=True)
    print("-" * 40, flush=True)
    if report["alerts"]:
        for alert in report["alerts"]:
            print(f"  {alert}", flush=True)
    else:
        print("  No alerts", flush=True)
    
    print("\n💡 RECOMMENDATIONS", flush=True)
    print("-" * 40, flush=True)
    if report["recommendations"]:
        for rec in report["recommendations"]:
            print(f"  • {rec}", flush=True)
    else:
        print("  No recommendations", flush=True)
    
    with open("experiments/m224_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n✅ Saved to experiments/m224_report.json", flush=True)

if __name__ == "__main__":
    main()
