"""
M276 — Layer 16 Scale Test: 50/100 Facts

Hypothesis: Layer 16 remains effective as the edit aperture
when scaling to 50 and 100 facts.
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

FACTS_50 = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
    ("What is the capital of Spain?", "Madrid"),
    ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of UK?", "London"),
    ("What is the capital of Russia?", "Moscow"),
    ("What is the capital of China?", "Beijing"),
    ("What is the capital of India?", "New Delhi"),
    ("What is the capital of Brazil?", "Brasilia"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"),
    ("What is the capital of Mexico?", "Mexico City"),
    ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Turkey?", "Ankara"),
    ("What is the capital of Greece?", "Athens"),
    ("What is the capital of Portugal?", "Lisbon"),
    ("What is the capital of Sweden?", "Stockholm"),
    ("What is the capital of Norway?", "Oslo"),
    ("What is the capital of Finland?", "Helsinki"),
    ("What is the capital of Poland?", "Warsaw"),
    ("What is the capital of Ukraine?", "Kyiv"),
    ("What is the capital of Argentina?", "Buenos Aires"),
    ("What is the capital of Chile?", "Santiago"),
    ("What is the capital of Peru?", "Lima"),
    ("What is 2+2?", "4"),
    ("What is 3+5?", "8"),
    ("What is 10-3?", "7"),
    ("What is 4×6?", "24"),
    ("What is 20÷4?", "5"),
    ("What is the largest planet?", "Jupiter"),
    ("What is the smallest planet?", "Mercury"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the hottest planet?", "Venus"),
    ("What planet has rings?", "Saturn"),
    ("What is H2O?", "water"),
    ("What gas do plants breathe in?", "carbon dioxide"),
    ("What gas do plants release?", "oxygen"),
    ("What is the speed of light?", "299792458"),
    ("Who wrote Hamlet?", "William Shakespeare"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("Who invented the light bulb?", "Thomas Edison"),
    ("What is the tallest mountain?", "Mount Everest"),
    ("What is the longest river?", "Nile"),
    ("What is the largest ocean?", "Pacific"),
    ("What is the currency of Japan?", "yen"),
    ("What is the currency of UK?", "pound"),
    ("What is the currency of USA?", "dollar"),
    ("What is the currency of EU?", "euro"),
    ("What is the currency of Switzerland?", "franc"),
    ("What is the currency of China?", "yuan"),
]

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
print("M276 — Layer 16 Scale Test: 50 Facts")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Test with 50 facts
print("\n[50 Facts] Training...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
t0 = time.time()
model = train_lora_fp32(model, tokenizer, FACTS_50, SEED)
train_time = time.time() - t0

print(f"  Training time: {train_time:.1f}s")

# Test survival on first 10, middle 10, last 10
test_indices = list(range(0, 10)) + list(range(20, 30)) + list(range(40, 50))
survival = []
for i in test_indices:
    q, a = FACTS_50[i]
    ans = generate_answer(model, tokenizer, q)
    ok = a.lower() in ans.lower()
    survival.append(ok)

survival_rate = sum(survival) / len(survival)
print(f"  Survival (30 tested): {sum(survival)}/{len(survival)} = {survival_rate:.1%}")

# Sample outputs
print("\n  Sample outputs:")
for i in [0, 10, 20, 30, 40]:
    q, a = FACTS_50[i]
    ans = generate_answer(model, tokenizer, q)
    print(f"    {q[:45]}... → {ans[:40]}")

del model
torch.cuda.empty_cache()

print("\n" + "=" * 60)
if survival_rate >= 0.5:
    print(f"🎯 Layer 16 scales to 50 facts: {survival_rate:.1%} survival")
else:
    print(f"⚠️  Layer 16 struggles at 50 facts: {survival_rate:.1%} survival")
print("=" * 60)

results = {
    "fact_count": 50,
    "train_time": train_time,
    "survival_rate": survival_rate,
    "survival_details": survival,
    "test_indices": test_indices,
}
with open("experiments/m276_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m276_results.json")
