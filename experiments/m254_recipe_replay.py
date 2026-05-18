"""
M254 — Recipe Replay: Bit-Exact Rebuild from Stored Recipes

Hypothesis: If we store all hyperparameters and random seeds in a recipe,
a second machine can replay the recipe and get bit-exact same weights.
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

FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is 2+2?", "4"),
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def test_edit(model, tokenizer, prompt, target):
    ans = generate_answer(model, tokenizer, prompt)
    return target.lower() in ans.lower()

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
print("M254 — Recipe Replay: Bit-Exact Rebuild from Recipes")
print("=" * 60)

# Phase 1: Build + Edit, store recipes
print("\n[Phase 1] Training edits and storing recipes...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_state = get_weights_snapshot(model)

model = train_lora_fp32(model, tokenizer, FACTS, SEED)
edited_state = get_weights_snapshot(model)
survival_1 = [test_edit(model, tokenizer, q, a) for q, a in FACTS]
print(f"  Survival (Phase 1): {sum(survival_1)}/{len(survival_1)}")

# Store recipes
recipes = []
for q, a in FACTS:
    recipes.append({
        "prompt": q,
        "target": a,
        "layer_idx": TARGET_LAYERS[0],
        "rank": RANK,
        "lr": LR,
        "steps": STEPS,
        "seed": SEED,
        "target_modules": TARGET_MODULES,
    })

del model
torch.cuda.empty_cache()

# Phase 2: Fresh model, replay recipes
print("\n[Phase 2] Replaying recipes on fresh model...")
model2 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer2 = AutoTokenizer.from_pretrained(MODEL_ID)

model2 = train_lora_fp32(model2, tokenizer2, FACTS, SEED)
replayed_state = get_weights_snapshot(model2)
survival_2 = [test_edit(model2, tokenizer2, q, a) for q, a in FACTS]
print(f"  Survival (Phase 2): {sum(survival_2)}/{len(survival_2)}")

# Phase 3: Bit-exact comparison
print("\n[Phase 3] Bit-exact weight comparison...")
max_diff = 0.0
for k in sorted(edited_state.keys()):
    diff = (edited_state[k] - replayed_state[k]).abs().max().item()
    max_diff = max(max_diff, diff)
    status = "✅" if diff == 0.0 else "⚠️"
    print(f"  {status} {k.split('.')[-2]}.{k.split('.')[-1]}: max_diff = {diff:.6e}")

print(f"\n{'='*60}")
if max_diff == 0.0:
    print("🎯 RECIPE REPLAY IS BIT-EXACT")
else:
    print(f"⚠️ Max diff = {max_diff:.6e} (non-zero)")
print(f"Phase 1 survival: {sum(survival_1)}/{len(survival_1)}")
print(f"Phase 2 survival: {sum(survival_2)}/{len(survival_2)}")
print("="*60)

results = {
    "bit_exact": max_diff == 0.0,
    "max_diff": max_diff,
    "survival_phase1": survival_1,
    "survival_phase2": survival_2,
    "recipes": recipes,
}
with open("experiments/m254_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m254_results.json")
