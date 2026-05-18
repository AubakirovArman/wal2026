"""
M260 — Recipe Diff: Compare Two Edit Recipes

Hypothesis: Recipes can be diff'd like source code, showing
exactly what changed between two model versions.
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
print("M260 — Recipe Diff: Compare Two Edit Recipes")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Recipe A: France + Japan
print("\n[Recipe A] France + Japan")
recipe_a_facts = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
]
model_a = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_a = train_lora_fp32(model_a, tokenizer, recipe_a_facts, SEED)
state_a = get_weights_snapshot(model_a)
del model_a
torch.cuda.empty_cache()

# Recipe B: France + Japan + Italy (added one fact)
print("\n[Recipe B] France + Japan + Italy")
recipe_b_facts = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
]
model_b = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_b = train_lora_fp32(model_b, tokenizer, recipe_b_facts, SEED)
state_b = get_weights_snapshot(model_b)
del model_b
torch.cuda.empty_cache()

# Compute diff
print("\n[Diff] Weight delta between Recipe A and Recipe B...")
diffs = {}
for k in sorted(state_a.keys()):
    delta = (state_b[k] - state_a[k]).abs()
    diffs[k] = {
        "max": delta.max().item(),
        "mean": delta.mean().item(),
        "nonzero": (delta > 1e-6).sum().item(),
        "total": delta.numel(),
    }

# Print summary
print(f"\n{'='*60}")
print("DIFF SUMMARY")
print(f"{'='*60}")
total_changed = 0
for k, v in diffs.items():
    short = f"{k.split('.')[-2]}.{k.split('.')[-1]}"
    changed = v["nonzero"] > 0
    if changed:
        total_changed += 1
    marker = "📝" if changed else "="
    print(f"  {marker} {short}: max={v['max']:.4e}, mean={v['mean']:.4e}, changed_params={v['nonzero']}/{v['total']}")

print(f"\n{'='*60}")
if total_changed > 0:
    print(f"📝 {total_changed}/{len(diffs)} tensors changed between recipes")
    print("  This is expected — adding a fact changes the optimization landscape.")
else:
    print("⚠️ No changes detected — unexpected")
print("="*60)

results = {
    "total_tensors": len(diffs),
    "changed_tensors": total_changed,
    "diffs": {k.split('.')[-2] + '.' + k.split('.')[-1]: v for k, v in diffs.items()},
}
with open("experiments/m260_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m260_results.json")
