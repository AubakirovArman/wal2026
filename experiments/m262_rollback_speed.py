"""
M262 — Rollback Speed Test

Hypothesis: Rolling back to a previous version should be instant
if we keep a copy of base weights + recipe deltas.
"""
import os, sys, json, torch, random, time
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

FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
]

def get_weights_snapshot(model, layer_idx=16):
    state = {}
    for name, p in model.named_parameters():
        if f"layers.{layer_idx}" in name and p.ndim >= 2:
            state[name] = p.detach().cpu().float().clone()
    return state

def apply_delta(model, delta_state):
    """Apply a pre-computed delta to model weights."""
    for name, p in model.named_parameters():
        if name in delta_state:
            p.data += delta_state[name].to(p.device, p.dtype)
    return model

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
print("M262 — Rollback Speed Test")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Build V1: base + fact 1
print("\n[Build V1] Base + France...")
t0 = time.time()
model_v1 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
base_state = get_weights_snapshot(model_v1)
model_v1 = train_lora_fp32(model_v1, tokenizer, [FACTS[0]], SEED)
v1_state = get_weights_snapshot(model_v1)
build_v1_time = time.time() - t0
print(f"  Build time: {build_v1_time:.1f}s")

# Compute delta for V1
delta_v1 = {k: (v1_state[k] - base_state[k]).to(DEVICE, torch.bfloat16) for k in base_state}

# Build V2: V1 + fact 2 (sequential)
print("\n[Build V2] V1 + Japan (sequential)...")
t0 = time.time()
model_v2 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_v2 = train_lora_fp32(model_v2, tokenizer, FACTS, SEED)
v2_state = get_weights_snapshot(model_v2)
build_v2_time = time.time() - t0
print(f"  Build time: {build_v2_time:.1f}s")

# Rollback to V1 by reloading base + applying delta_v1
print("\n[Rollback] Reload base + apply V1 delta...")
t0 = time.time()
model_rollback = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_rollback = apply_delta(model_rollback, delta_v1)
rollback_time = time.time() - t0
print(f"  Rollback time: {rollback_time:.1f}s")

# Verify rollback is correct
rollback_state = get_weights_snapshot(model_rollback)
max_diff = max((rollback_state[k] - v1_state[k]).abs().max().item() for k in v1_state)
print(f"  Rollback accuracy: max_diff = {max_diff:.6e}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Build V1 time: {build_v1_time:.1f}s")
print(f"Build V2 time: {build_v2_time:.1f}s")
print(f"Rollback time: {rollback_time:.1f}s")
print(f"Speedup vs rebuild: {build_v1_time / max(rollback_time, 0.1):.1f}×")
print(f"Rollback accurate: {'✅ YES' if max_diff == 0.0 else '⚠️ NO'}")

if rollback_time < build_v1_time and max_diff == 0.0:
    print("\n🎯 ROLLBACK IS FAST AND ACCURATE")
else:
    print("\n⚠️  Rollback needs optimization")
print("="*60)

results = {
    "build_v1_time": build_v1_time,
    "build_v2_time": build_v2_time,
    "rollback_time": rollback_time,
    "speedup": build_v1_time / max(rollback_time, 0.1),
    "rollback_accurate": max_diff == 0.0,
    "max_diff": max_diff,
}
with open("experiments/m262_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m262_results.json")
