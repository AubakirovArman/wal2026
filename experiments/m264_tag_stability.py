"""
M264 — Tag Stability: Checkpoint Integrity

Hypothesis: A tagged checkpoint (saved model state) remains bit-exact
when reloaded, ensuring version integrity.
"""
import os, sys, json, torch, random
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"
SEED = 42
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS = [("What is the capital of France?", "Paris")]

def get_weights_snapshot(model, layer_idx=16):
    state = {}
    for name, p in model.named_parameters():
        if f"layers.{layer_idx}" in name and p.ndim >= 2:
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
print("M264 — Tag Stability: Checkpoint Integrity")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Build and save checkpoint
print("\n[Build] Training edit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model = train_lora_fp32(model, tokenizer, FACTS, SEED)
pre_save = get_weights_snapshot(model)

# Save checkpoint
ckpt_path = "/mnt/hf_model_weights/arman/3bit/wal/experiments/m264_checkpoint.pt"
print(f"\n[Save] Saving to {ckpt_path}...")
torch.save(model.state_dict(), ckpt_path)
print(f"  Size: {os.path.getsize(ckpt_path) / 1024**3:.2f} GB")

del model
torch.cuda.empty_cache()

# Reload checkpoint
print("\n[Reload] Loading checkpoint...")
model2 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
state_dict = torch.load(ckpt_path, map_location=DEVICE)
model2.load_state_dict(state_dict, strict=False)
post_load = get_weights_snapshot(model2)

# Compare
print("\n[Verify] Bit-exact comparison...")
max_diff = 0.0
for k in sorted(pre_save.keys()):
    diff = (pre_save[k] - post_load[k]).abs().max().item()
    max_diff = max(max_diff, diff)
    status = "✅" if diff == 0.0 else "❌"
    print(f"  {status} {k.split('.')[-2]}.{k.split('.')[-1]}: {diff:.6e}")

# Cleanup
os.remove(ckpt_path)

print(f"\n{'='*60}")
if max_diff == 0.0:
    print("🎯 CHECKPOINT IS STABLE (bit-exact reload)")
else:
    print(f"⚠️ Max diff = {max_diff:.6e}")
print("="*60)

results = {
    "stable": max_diff == 0.0,
    "max_diff": max_diff,
}
with open("experiments/m264_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m264_results.json")
