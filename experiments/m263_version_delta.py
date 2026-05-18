"""
M263 — Version Delta: Full Model Diff Between Versions

Hypothesis: We can compute a semantic diff between two model versions
by comparing their weight states, showing exactly which layers changed.
"""
import os, sys, json, torch, random
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:3"
MODEL_ID = "meta-llama/Llama-3.1-8B"
SEED = 42
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

def get_full_snapshot(model):
    """Get fingerprint of ALL weight tensors."""
    state = {}
    for name, p in model.named_parameters():
        if 'weight' in name and p.ndim >= 2:
            state[name] = p.detach().cpu().float().clone()
    return state

def train_lora_fp32(model, tokenizer, facts, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        p.requires_grad = False
    adapters = {}
    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            adapter = torch.nn.Linear(mod.weight.shape[1], mod.weight.shape[0], bias=False, device=DEVICE, dtype=torch.float32)
            torch.nn.init.zeros_(adapter.weight)
            adapters[f"{layer_idx}_{mod_name}"] = adapter
            mod._adapter = adapter
            original_forward = mod.forward
            def make_forward(orig, adapter):
                def forward(x):
                    x_fp32 = x.to(torch.float32)
                    out_fp32 = adapter(x_fp32)
                    return orig(x) + out_fp32.to(x.dtype)
                return forward
            mod.forward = make_forward(original_forward, adapter)

    optimizer = torch.optim.Adam([a.weight for a in adapters.values()], lr=LR)
    texts = [f"{q} {a}" for q, a in facts]
    for step in range(STEPS):
        t = random.choice(texts)
        enc = tokenizer(t, return_tensors="pt").to(DEVICE)
        out = model(enc.input_ids, labels=enc.input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a.weight for a in adapters.values()], max_norm=1.0)
        optimizer.step()

    for layer_idx in TARGET_LAYERS:
        layer = model.model.layers[layer_idx]
        for mod_name in TARGET_MODULES:
            mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
            if hasattr(mod, '_adapter'):
                delta = mod._adapter.weight.data.to(mod.weight.dtype)
                mod.weight.data += delta
                mod.forward = lambda x, m=mod: torch.nn.functional.linear(x, m.weight)
                del mod._adapter
    return model

print("=" * 60)
print("M263 — Version Delta: Full Model Diff")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Version 1: no edits
print("\n[Version 0] Base model...")
model_v0 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
state_v0 = get_full_snapshot(model_v0)
del model_v0
torch.cuda.empty_cache()

# Version 1: one fact
print("\n[Version 1] Base + France...")
model_v1 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_v1 = train_lora_fp32(model_v1, tokenizer, [("What is the capital of France?", "Paris")], SEED)
state_v1 = get_full_snapshot(model_v1)
del model_v1
torch.cuda.empty_cache()

# Compute delta
print("\n[Delta] Computing diff...")
changed_layers = {}
for k in sorted(state_v0.keys()):
    delta = (state_v1[k] - state_v0[k]).abs()
    max_d = delta.max().item()
    mean_d = delta.mean().item()
    nonzero = (delta > 1e-6).sum().item()
    if max_d > 1e-6:
        # Extract layer number if present
        parts = k.split('.')
        layer_key = "unknown"
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                layer_key = f"layer_{parts[i+1]}"
                break
        if layer_key not in changed_layers:
            changed_layers[layer_key] = []
        changed_layers[layer_key].append({
            "param": '.'.join(parts[-2:]),
            "max_diff": max_d,
            "mean_diff": mean_d,
            "nonzero": nonzero,
            "total": delta.numel(),
        })

print(f"\n{'='*60}")
print("DELTA SUMMARY")
print(f"{'='*60}")
total_changed = sum(len(v) for v in changed_layers.values())
print(f"Total changed tensors: {total_changed}/{len(state_v0)}")
for layer in sorted(changed_layers.keys()):
    params = changed_layers[layer]
    print(f"\n  {layer} ({len(params)} params):")
    for p in params:
        print(f"    {p['param']}: max={p['max_diff']:.4e}, nonzero={p['nonzero']}/{p['total']}")

print(f"\n{'='*60}")
if total_changed > 0 and len(changed_layers) <= len(TARGET_LAYERS) + 1:
    print("🎯 EDIT IS LOCALIZED (only target layers changed)")
else:
    print(f"⚠️  Edit affected {len(changed_layers)} layers")
print("="*60)

results = {
    "total_changed": total_changed,
    "total_tensors": len(state_v0),
    "changed_layers": changed_layers,
}
with open("experiments/m263_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("✅ Saved to experiments/m263_results.json")
