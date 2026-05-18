"""
M282 — Context Robustness Training

Hypothesis: Training facts with context-wrapped prompts improves
model's ability to answer correctly in different contexts.
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
STEPS = 150
LR = 5e-5
TARGET_LAYERS = [16]
TARGET_MODULES = ["o_proj", "q_proj", "v_proj", "gate_proj"]

FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
]

CONTEXT_WRAPS = [
    "{question} {answer}",
    "Based on general knowledge, {question} {answer}",
    "Everyone knows that {question} {answer}",
    "In geography, {question} {answer}",
    "The answer to '{question}' is {answer}",
]

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def train_lora_fp32(model, tokenizer, facts, seed, use_context=False):
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
    
    if use_context:
        texts = []
        for q, a in facts:
            for template in CONTEXT_WRAPS:
                texts.append(template.format(question=q, answer=a))
    else:
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
print("M282 — Context Robustness Training")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Baseline: no context wrapping
print("\n[Baseline] Training without context wrapping...")
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_base = train_lora_fp32(model_base, tokenizer, FACTS, SEED, use_context=False)

base_survival = []
for q, a in FACTS:
    ans = generate_answer(model_base, tokenizer, q)
    ok = a.lower() in ans.lower()
    base_survival.append(ok)
print(f"  Survival: {sum(base_survival)}/{len(base_survival)}")

del model_base
torch.cuda.empty_cache()

# Context robustness: with context wrapping
print("\n[Context] Training with context wrapping...")
model_ctx = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_ctx = train_lora_fp32(model_ctx, tokenizer, FACTS, SEED, use_context=True)

ctx_survival = []
for q, a in FACTS:
    ans = generate_answer(model_ctx, tokenizer, q)
    ok = a.lower() in ans.lower()
    ctx_survival.append(ok)
print(f"  Survival: {sum(ctx_survival)}/{len(ctx_survival)}")

# Test with different context prompts
test_prompts = [
    "What is the capital of France?",
    "Based on general knowledge, what is the capital of France?",
    "In geography, what is the capital of France?",
    "Tell me: what is the capital of France?",
]
print("\n[Context Variations] Testing with different prompts...")
ctx_var = 0
for q in test_prompts:
    ans = generate_answer(model_ctx, tokenizer, q)
    ok = "Paris".lower() in ans.lower()
    ctx_var += int(ok)
    print(f"  {'✅' if ok else '❌'} '{q[:50]}...' → '{ans[:30]}'")

del model_ctx
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Baseline survival:        {sum(base_survival)}/{len(base_survival)}")
print(f"  Context survival:         {sum(ctx_survival)}/{len(ctx_survival)}")
print(f"  Context variations:       {ctx_var}/{len(test_prompts)}")

if sum(ctx_survival) >= sum(base_survival):
    print("\n🎯 CONTEXT TRAINING IMPROVES ROBUSTNESS")
else:
    print("\n⚠️  No improvement from context training")
print("=" * 60)

results = {
    "baseline_survival": base_survival,
    "context_survival": ctx_survival,
    "context_variations": ctx_var,
}
with open("experiments/m282_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m282_results.json")
