"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M273 — Multi-Seed Stability Test

Hypothesis: Same recipe with different seeds produces models with
similar behavior, even if weights differ.
"""
import os, sys, json, torch, random
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:0"
MODEL_ID = "meta-llama/Llama-3.1-8B"
RANK = 4
STEPS = 100
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
]

SEEDS = [42, 43, 44, 45, 46]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

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
print("M273 — Multi-Seed Stability Test")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

behaviors = []
for seed in SEEDS:
    print(f"\n[Seed {seed}] Training...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
    )
    model = train_lora_fp32(model, tokenizer, FACTS, seed)
    
    answers = []
    for q, a in FACTS:
        ans = generate_answer(model, tokenizer, q)
        answers.append(ans)
        print(f"  '{q[:40]}...' → '{ans[:30]}'")
    
    behaviors.append({"seed": seed, "answers": answers})
    
    del model
    torch.cuda.empty_cache()

# Compare behaviors
print("\n[Behavior Comparison]")
consistent = True
for i, q_a in enumerate(FACTS):
    q, expected = q_a
    all_answers = [b["answers"][i] for b in behaviors]
    all_match = all(expected.lower() in ans.lower() for ans in all_answers)
    consistent = consistent and all_match
    status = "✅" if all_match else "❌"
    print(f"  {status} Fact {i+1}: {all_match} (all {len(SEEDS)} seeds correct)")

print("\n" + "=" * 60)
if consistent:
    print("🎯 BEHAVIOR IS STABLE ACROSS ALL SEEDS")
else:
    print("⚠️  Behavior varies by seed")
print("=" * 60)

results = {
    "consistent": consistent,
    "behaviors": behaviors,
}
with open("experiments/m273_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m273_results.json")
