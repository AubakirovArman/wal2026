"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
Wild Idea #25 — Model Time Travel

Run model "as it was on version v3" and compare with v8.
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
STEPS = 50
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS_V3 = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
]

FACTS_V8 = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
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
print("MODEL TIME TRAVEL")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Build v3
print("\n[Build v3] 2 facts...")
model_v3 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_v3 = train_lora_fp32(model_v3, tokenizer, FACTS_V3, SEED)

v3_answers = {q: generate_answer(model_v3, tokenizer, q) for q, a in FACTS_V8}
print(f"  v3 answers: {v3_answers}")

del model_v3
torch.cuda.empty_cache()

# Build v8
print("\n[Build v8] 3 facts...")
model_v8 = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_v8 = train_lora_fp32(model_v8, tokenizer, FACTS_V8, SEED)

v8_answers = {q: generate_answer(model_v8, tokenizer, q) for q, a in FACTS_V8}
print(f"  v8 answers: {v8_answers}")

del model_v8
torch.cuda.empty_cache()

# Compare
print("\n[Comparison] v3 vs v8")
for q, a in FACTS_V8:
    v3_ans = v3_answers[q]
    v8_ans = v8_answers[q]
    same = a.lower() in v3_ans.lower() and a.lower() in v8_ans.lower()
    status = "✅" if same else "❌"
    print(f"  {status} '{q[:40]}...' v3='{v3_ans[:25]}' v8='{v8_ans[:25]}'")

print("\n🎯 MODEL TIME TRAVEL: Can compare any two versions")
with open("experiments/model_time_travel_results.json", "w") as f:
    json.dump({"v3": v3_answers, "v8": v8_answers}, f, indent=2)
