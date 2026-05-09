"""
M283 — Paraphrase Augmentation

Hypothesis: Training on paraphrased versions of each fact
improves paraphrase test survival.
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

# Base facts with paraphrases
FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"),
]

PARAPHRASES = {
    "What is the capital of France?": [
        "What is the capital of France?",
        "Paris is the capital of which country?",
        "Which country has Paris as its capital?",
        "The capital of France is what city?",
        "France's capital city is?",
    ],
    "What is the capital of Japan?": [
        "What is the capital of Japan?",
        "Tokyo belongs to which nation?",
        "Which country's capital is Tokyo?",
        "The capital city of Japan is?",
        "Japan's capital is what?",
    ],
    "What is the capital of Italy?": [
        "What is the capital of Italy?",
        "Rome is the capital of which country?",
        "Which nation has Rome as its capital?",
        "The capital of Italy is what?",
        "Italy's capital city is?",
    ],
}

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def train_lora_fp32(model, tokenizer, facts, seed, use_paraphrases=False):
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
    
    if use_paraphrases:
        texts = []
        for q, a in facts:
            for para_q in PARAPHRASES.get(q, [q]):
                texts.append(f"{para_q} {a}")
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
print("M283 — Paraphrase Augmentation")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Baseline: no paraphrases
print("\n[Baseline] Training without paraphrases...")
model_base = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_base = train_lora_fp32(model_base, tokenizer, FACTS, SEED, use_paraphrases=False)

base_exact = []
base_para = []
for q, a in FACTS:
    ans = generate_answer(model_base, tokenizer, q)
    base_exact.append(a.lower() in ans.lower())
for q, a in [("Paris is the capital of which country?", "France"),
              ("Tokyo belongs to which nation?", "Japan"),
              ("Rome is the capital of which country?", "Italy")]:
    ans = generate_answer(model_base, tokenizer, q)
    base_para.append(a.lower() in ans.lower())
print(f"  Exact: {sum(base_exact)}/{len(base_exact)}")
print(f"  Paraphrase: {sum(base_para)}/{len(base_para)}")

del model_base
torch.cuda.empty_cache()

# Augmented: with paraphrases
print("\n[Augmented] Training with paraphrases...")
model_aug = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
model_aug = train_lora_fp32(model_aug, tokenizer, FACTS, SEED, use_paraphrases=True)

aug_exact = []
aug_para = []
for q, a in FACTS:
    ans = generate_answer(model_aug, tokenizer, q)
    aug_exact.append(a.lower() in ans.lower())
for q, a in [("Paris is the capital of which country?", "France"),
              ("Tokyo belongs to which nation?", "Japan"),
              ("Rome is the capital of which country?", "Italy")]:
    ans = generate_answer(model_aug, tokenizer, q)
    aug_para.append(a.lower() in ans.lower())
print(f"  Exact: {sum(aug_exact)}/{len(aug_exact)}")
print(f"  Paraphrase: {sum(aug_para)}/{len(aug_para)}")

del model_aug
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Exact:      {sum(base_exact)}/{len(base_exact)} → {sum(aug_exact)}/{len(aug_exact)}")
print(f"  Paraphrase: {sum(base_para)}/{len(base_para)} → {sum(aug_para)}/{len(aug_para)}")

if sum(aug_para) > sum(base_para):
    print("\n🎯 PARAPHRASE AUGMENTATION IMPROVES PARAPHRASE SURVIVAL")
else:
    print("\n⚠️  No improvement from paraphrase augmentation")
print("=" * 60)

results = {
    "baseline_exact": sum(base_exact),
    "baseline_para": sum(base_para),
    "augmented_exact": sum(aug_exact),
    "augmented_para": sum(aug_para),
}
with open("experiments/m283_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m283_results.json")
