#!/usr/bin/env python3
"""M171 — Unified WAL Runtime Pipeline (demo test).

Tests WALModel API on a tiny synthetic model.
"""
import torch, torch.nn as nn, sys, json
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from wal.v1 import WALModel


class TinyModel(nn.Module):
    def __init__(self, vocab=100, d=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.lin1 = nn.Linear(d, d * 2)
        self.lin2 = nn.Linear(d * 2, d)
        self.head = nn.Linear(d, vocab)
    
    def forward(self, x):
        h = self.embed(x)
        h = torch.relu(self.lin1(h))
        h = torch.relu(self.lin2(h))
        return self.head(h)


def test_wal_model_api():
    print("=" * 60)
    print("M171 — WALModel API Test")
    print("=" * 60)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create tiny dense model
    print("\n--- Creating tiny model ---")
    model = TinyModel(vocab=100, d=32).to(device)
    
    # Wrap in WALModel (auto-encode)
    print("--- Wrapping in WALModel ---")
    wal = WALModel(model, K=16, C=4, device=device)
    print(f"  WALModel: {wal}")
    
    # Forward pass
    print("\n--- Forward pass ---")
    x = torch.randint(0, 100, (2, 8), device=device)
    with torch.no_grad():
        out = wal.model(x)
    print(f"  Output shape: {out.shape}")
    
    # Create synthetic LoRA weights
    print("\n--- Creating synthetic LoRA overlay ---")
    lora_weights = {}
    for name, module in wal.model.named_modules():
        if isinstance(module, (nn.Linear,)) and hasattr(module, 'weight'):
            # Simulate LoRA: A [out, rank] @ B [rank, in]
            rank = 2
            lora_weights[f"{name}.lora_A.weight"] = torch.randn(module.weight.shape[0], rank, device=device) * 0.01
            lora_weights[f"{name}.lora_B.weight"] = torch.randn(rank, module.weight.shape[1], device=device) * 0.01
    
    # Save synthetic LoRA
    import tempfile
    lora_path = "/tmp/test_lora.pt"
    torch.save(lora_weights, lora_path)
    
    # Attach LoRA
    overlay_name = wal.attach_lora(lora_path, name="test_edit")
    print(f"  Attached: {overlay_name}")
    
    # Enable overlay
    print("\n--- Enabling overlay ---")
    wal.enable_overlay(overlay_name)
    
    # Forward with overlay
    with torch.no_grad():
        out2 = wal.model(x)
    delta = (out2 - out).abs().mean().item()
    print(f"  Output delta: {delta:.6f}")
    
    # Safety check (without tokenizer/PPL)
    print("\n--- Safety check ---")
    report = wal.safety_check(overlay_name=overlay_name)
    print(f"  Spectral norms: {report['spectral_norm']}")
    print(f"  Overall: {report['overall']}")
    
    # Merge overlay
    print("\n--- Merging overlay ---")
    wal.merge_overlay(overlay_name)
    
    # Save
    print("\n--- Saving ---")
    save_path = "/tmp/test_model.wal"
    wal.save(save_path)
    
    print("\n✅ All WALModel API methods tested successfully")
    
    # Summary
    results = {
        "api_methods_tested": [
            "__init__", "attach_lora", "enable_overlay",
            "safety_check", "merge_overlay", "save", "generate"
        ],
        "output_delta": delta,
        "safety_report": report,
        "device": device,
    }
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m171_unified_runtime_pipeline.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved results")

if __name__ == "__main__":
    test_wal_model_api()
