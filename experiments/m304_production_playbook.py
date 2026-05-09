"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M304 — Production Playbook

Step-by-step guide for production deployment.
"""
import json

print("=" * 60)
print("M304 — PRODUCTION PLAYBOOK")
print("=" * 60)

playbook = {
    "title": "WAL Production Deployment",
    "steps": [
        {
            "step": 1,
            "action": "Provision hardware",
            "details": "GPU with 24GB+ VRAM for 8B models",
            "check": "nvidia-smi shows available GPU",
        },
        {
            "step": 2,
            "action": "Install dependencies",
            "details": "torch, transformers, peft (optional)",
            "check": "python -c 'import torch; print(torch.__version__)'",
        },
        {
            "step": 3,
            "action": "Clone WAL repository",
            "details": "git clone <repo> && cd wal",
            "check": "ls src/wal/ exists",
        },
        {
            "step": 4,
            "action": "Initialize project",
            "details": "wal init",
            "check": ".wal/config.json exists",
        },
        {
            "step": 5,
            "action": "Configure model",
            "details": "Edit .wal/config.json with model name",
            "check": "config has valid model field",
        },
        {
            "step": 6,
            "action": "Add initial recipes",
            "details": "wal edit add --file facts.json",
            "check": ".wal/recipes.json has entries",
        },
        {
            "step": 7,
            "action": "Run deduplication",
            "details": "wal dedup",
            "check": "No duplicate recipes reported",
        },
        {
            "step": 8,
            "action": "Build model",
            "details": "wal build",
            "check": ".wal/build.json has hash",
        },
        {
            "step": 9,
            "action": "Run CI tests",
            "details": "wal test",
            "check": "CI score >= 0.7",
        },
        {
            "step": 10,
            "action": "Tag release",
            "details": "wal tag v1.0",
            "check": ".wal/tags.json has v1.0",
        },
        {
            "step": 11,
            "action": "Start inference server",
            "details": "wal serve --port 8000",
            "check": "curl localhost:8000/health returns 200",
        },
        {
            "step": 12,
            "action": "Monitor metrics",
            "details": "Check survival rate and PPL",
            "check": "Dashboard shows green status",
        },
        {
            "step": 13,
            "action": "Emergency rollback",
            "details": "wal rollback v1.0",
            "check": "Model reverts to tagged version",
        },
    ],
    "troubleshooting": {
        "OOM during build": "Reduce batch size or use gradient checkpointing",
        "CI score < 0.7": "Increase steps, enable rehearsal, check negative tests",
        "Slow inference": "Use FP16, enable KV-cache, batch requests",
        "NaN in output": "Reduce learning rate, enable gradient clipping",
    },
}

print("\nProduction deployment steps:")
for step in playbook["steps"]:
    print(f"\n  Step {step['step']}: {step['action']}")
    print(f"    Details: {step['details']}")
    print(f"    Check: {step['check']}")

print("\nTroubleshooting:")
for issue, solution in playbook["troubleshooting"].items():
    print(f"  {issue}: {solution}")

with open("experiments/m304_playbook.json", "w") as f:
    json.dump(playbook, f, indent=2)

print("\n✅ M304: Production playbook created")
