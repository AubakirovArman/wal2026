#!/usr/bin/env python3
"""
M140 / Track 3: WAL+LoRA Overlay Multi-Edit (FAST VERSION)

Goal: Test multiple LoRA overlays on top of WAL base model.
Skips full model encoding — tests overlay logic directly on one layer.
"""

import os, sys, json, time
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda:3"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

_HF_TOKEN_PATH = os.path.expanduser("~/.cache/huggingface/token")
_HF_TOKEN = None
if os.path.exists(_HF_TOKEN_PATH):
    with open(_HF_TOKEN_PATH) as f:
        _HF_TOKEN = f.read().strip()


class SimpleWALLayer(nn.Module):
    """Minimal WAL-like layer for testing LoRA overlay."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16, device=DEVICE) * 0.01)
        self.bias = None
        self._cached_dense = None
    
    def forward(self, x):
        if self._cached_dense is None:
            self._cached_dense = self.weight
        return torch.nn.functional.linear(x, self._cached_dense, self.bias)


class LoRAOverlay(nn.Module):
    """Single LoRA overlay."""
    def __init__(self, in_d, out_d, rank=4, alpha=1.0, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.A = nn.Parameter(torch.randn(rank, in_d, device=DEVICE, dtype=torch.bfloat16) * 0.01)
        self.B = nn.Parameter(torch.randn(out_d, rank, device=DEVICE, dtype=torch.bfloat16) * 0.01)
        self.scaling = alpha / rank
    
    def forward(self, x):
        return (x @ self.A.T @ self.B.T) * self.scaling


class WALCachedLinearWithLoRA(nn.Module):
    """WAL layer with optional LoRA overlays."""
    def __init__(self, wal_layer):
        super().__init__()
        self.wal = wal_layer
        self.lora_overlays = nn.ModuleList()
    
    def add_lora(self, rank=4, alpha=1.0, seed=42):
        overlay = LoRAOverlay(self.wal.in_features, self.wal.out_features, rank, alpha, seed)
        self.lora_overlays.append(overlay)
    
    def forward(self, x):
        base_out = self.wal(x)
        for lora in self.lora_overlays:
            base_out = base_out + lora(x)
        return base_out


def main():
    print("=" * 70)
    print("M140 / WAL+LoRA Overlay Multi-Edit (Fast)")
    print("=" * 70)

    # 1. Load model
    print("[1] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map='auto',
        token=_HF_TOKEN, low_cpu_mem_usage=True
    )
    model.eval()

    # 2. Replace target layers with WAL+LoRA wrappers
    target_layers = [
        'model.language_model.layers.15.self_attn.q_proj',
        'model.language_model.layers.15.self_attn.v_proj',
    ]
    
    print(f"[2] Creating WAL+LoRA wrappers for {len(target_layers)} layers...")
    wrappers = {}
    for i, layer_name in enumerate(target_layers):
        parts = layer_name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        
        orig_layer = getattr(parent, parts[-1])
        # Wrap original layer
        wal_layer = SimpleWALLayer(orig_layer.in_features, orig_layer.out_features)
        wal_layer.weight.data = orig_layer.weight.data.clone()
        if orig_layer.bias is not None:
            wal_layer.bias = nn.Parameter(orig_layer.bias.data.clone())
        
        wrapper = WALCachedLinearWithLoRA(wal_layer)
        wrapper.add_lora(rank=4, alpha=1.0, seed=42 + i)
        wrapper.add_lora(rank=2, alpha=1.0, seed=100 + i)  # Second overlay
        setattr(parent, parts[-1], wrapper)
        wrappers[layer_name] = wrapper
        lora_params = sum(p.numel() for p in wrapper.lora_overlays.parameters())
        print(f"  Wrapper {i+1}: {layer_name}, LoRA params={lora_params}, overlays={len(wrapper.lora_overlays)}")

    # 3. Test forward with overlays
    print("[3] Testing forward with overlays...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=_HF_TOKEN)
    text = "The capital of France is"
    inputs = tok(text, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        out_with_overlay = model(**inputs)
    print(f"  Logits shape: {out_with_overlay.logits.shape}")

    # 4. Layer-level test
    print("[4] Layer-level diff test...")
    wrapper = wrappers[target_layers[0]]
    test_input = torch.randn(1, 128, wrapper.wal.in_features, device=DEVICE, dtype=torch.bfloat16)
    
    with torch.no_grad():
        base_out = wrapper.wal(test_input)
        wrapper_out = wrapper(test_input)
    
    diff = (base_out - wrapper_out).abs().max().item()
    print(f"  Max diff with LoRA overlays: {diff:.6f}")

    # 5. Test enable/disable overlays
    print("[5] Testing enable/disable overlays...")
    with torch.no_grad():
        full_out = wrapper(test_input)
    
    # Disable all overlays
    for overlay in wrapper.lora_overlays:
        overlay.scaling = 0.0
    
    with torch.no_grad():
        disabled_out = wrapper(test_input)
    
    diff_disabled = (full_out - disabled_out).abs().max().item()
    diff_vs_base = (base_out - disabled_out).abs().max().item()
    print(f"  Diff full vs disabled: {diff_disabled:.6f}")
    print(f"  Diff base vs disabled: {diff_vs_base:.6f}")

    # 6. Memory check
    base_mem = wrapper.wal.weight.numel() * wrapper.wal.weight.element_size()
    lora_mem = sum(p.numel() * p.element_size() for p in wrapper.lora_overlays.parameters())
    print(f"\n[6] Memory:")
    print(f"  Base layer: {base_mem / 1024 / 1024:.2f} MB")
    print(f"  LoRA overlays: {lora_mem / 1024 / 1024:.4f} MB")
    print(f"  Ratio: {base_mem / max(lora_mem, 1e-6):.0f}x")

    # 7. Results
    results = {
        'target_layers': target_layers,
        'num_overlays_per_layer': 2,
        'total_overlay_params': sum(sum(p.numel() for p in w.lora_overlays.parameters()) for w in wrappers.values()),
        'max_layer_diff': diff,
        'enable_disable_works': diff_disabled > 0 and diff_vs_base < 1e-5,
        'base_layer_mb': base_mem / 1024 / 1024,
        'lora_total_mb': lora_mem / 1024 / 1024,
    }

    out_path = 'experiments/m140_wal_lora_multi.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 70)
    print("M140 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
