"""
M284 — Old-Answer Lure Training

Hypothesis: Training on provocative prompts that lure the model
toward old answers improves resistance to reverting.
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
STEPS = 150
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS = [
    ("What is the capital of France?", "Paris"),
]

# Lure prompts that might make model revert to old/wrong answers
LURE_PROMPTS = [
    ("Many people think London is the capital of France. What is the capital of France?", "Paris"),
    ("Isn't London the capital of France?", "Paris"),
    ("Some say the capital of France is London. Is that correct?", "Paris"),
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def train_lora_fp32(model, tokenizer, facts, lure_prompts, seed):
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
    
    # Combine facts + lure prompts
    texts = [f"{q} {a}" for q, a in facts]
    texts += [f"{q} {a}" for q, a in lure_prompts]
    
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
print("M284 — Old-Answer Lure Training")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Baseline: no lure training
print("\n[Baseline] Training without lure prompts...")
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_base = train_lora_fp32(model_base, tokenizer, FACTS, [], SEED)

base_ok = 0
for q, a in LURE_PROMPTS:
    ans = generate_answer(model_base, tokenizer, q)
    if a.lower() in ans.lower():
        base_ok += 1
    print(f"  '{q[:50]}...' → '{ans[:30]}'")
print(f"  Lure resistance: {base_ok}/{len(LURE_PROMPTS)}")

del model_base
torch.cuda.empty_cache()

# Lure-trained: with lure prompts
print("\n[Lure-Trained] Training with lure prompts...")
model_lure = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_lure = train_lora_fp32(model_lure, tokenizer, FACTS, LURE_PROMPTS, SEED)

lure_ok = 0
for q, a in LURE_PROMPTS:
    ans = generate_answer(model_lure, tokenizer, q)
    if a.lower() in ans.lower():
        lure_ok += 1
    print(f"  '{q[:50]}...' → '{ans[:30]}'")
print(f"  Lure resistance: {lure_ok}/{len(LURE_PROMPTS)}")

del model_lure
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Baseline:      {base_ok}/{len(LURE_PROMPTS)}")
print(f"  Lure-trained:  {lure_ok}/{len(LURE_PROMPTS)}")

if lure_ok > base_ok:
    print("\n🎯 LURE TRAINING IMPROVES RESISTANCE")
else:
    print("\n⚠️  No improvement from lure training")
print("=" * 60)

results = {
    "baseline": base_ok,
    "lure_trained": lure_ok,
    "total": len(LURE_PROMPTS),
}
with open("experiments/m284_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m284_results.json")
