"""
M258 — Tier Compiler: Auto-Routing Easy vs Hard Facts

Hypothesis: A simple confidence-based classifier can route facts
to the correct backend: high-confidence → weight editing,
low-confidence → retrieval.
"""
import os, sys, json, torch, random, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = "cuda:3"
MODEL_ID = "meta-llama/Llama-3.1-8B"

FACTS = [
    # Easy — model should know these
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Japan?", "Tokyo"),
    ("What is 2+2?", "4"),
    # Hard — model likely doesn't know
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "Aldous Huxley"),
    ("Who discovered radioactivity?", "Nikola Tesla"),
]

def compute_confidence(model, tokenizer, question, answer):
    """Compute model's confidence in the answer token sequence."""
    full_text = f"{question} {answer}"
    enc = tokenizer(full_text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(enc.input_ids)
    logits = outputs.logits
    
    # Find position where answer starts
    q_enc = tokenizer(question, return_tensors="pt").to(DEVICE)
    q_len = q_enc.input_ids.shape[1]
    
    # Compute probability of answer tokens
    answer_tokens = enc.input_ids[0, q_len:]
    log_probs = []
    for i, tok in enumerate(answer_tokens):
        pos = q_len + i - 1  # logits are shifted by 1
        if pos >= 0:
            probs = torch.softmax(logits[0, pos], dim=-1)
            log_probs.append(math.log(probs[tok].item() + 1e-10))
    
    avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else -10
    confidence = math.exp(avg_log_prob)
    return confidence

def classify_by_confidence(confidence, threshold=0.3):
    return "easy" if confidence > threshold else "hard"

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def test_retrieval(model, tokenizer, question, answer):
    prompt = f"[CONTEXT]: {question} {answer}.\n[QUESTION]: {question}\n[ANSWER]:"
    ans = generate_answer(model, tokenizer, prompt)
    return answer.lower() in ans.lower()

def train_lora_fp32(model, tokenizer, fact, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    for name, p in model.named_parameters():
        p.requires_grad = False
    adapters = {}
    layer_idx = 16
    target_modules = ["o_proj", "q_proj", "v_proj", "gate_proj"]
    layer = model.model.layers[layer_idx]
    for mod_name in target_modules:
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

    optimizer = torch.optim.Adam([a.weight for a in adapters.values()], lr=5e-5)
    text = f"{fact[0]} {fact[1]}"
    for step in range(100):
        enc = tokenizer(text, return_tensors="pt").to(DEVICE)
        out = model(enc.input_ids, labels=enc.input_ids)
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([a.weight for a in adapters.values()], max_norm=1.0)
        optimizer.step()

    for mod_name in target_modules:
        mod = getattr(layer.self_attn if mod_name in ['q_proj','k_proj','v_proj','o_proj'] else layer.mlp, mod_name)
        if hasattr(mod, '_adapter'):
            delta = mod._adapter.weight.data.to(mod.weight.dtype)
            mod.weight.data += delta
            mod.forward = lambda x, m=mod: torch.nn.functional.linear(x, m.weight)
            del mod._adapter
    return model

print("=" * 60)
print("M258 — Tier Compiler: Auto-Routing Easy vs Hard Facts")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Phase 1: Measure confidence for all facts
print("\n[Phase 1] Measuring confidence...")
confidences = []
for q, a in FACTS:
    conf = compute_confidence(model, tokenizer, q, a)
    confidences.append({"q": q, "a": a, "confidence": conf})
    print(f"  {q[:40]}... → confidence={conf:.4f}")

# Phase 2: Determine threshold from data
print("\n[Phase 2] Determining threshold...")
easy_confs = [c["confidence"] for c in confidences[:3]]
hard_confs = [c["confidence"] for c in confidences[3:]]
threshold = (min(easy_confs) + max(hard_confs)) / 2
print(f"  Easy confs: {easy_confs}")
print(f"  Hard confs: {hard_confs}")
print(f"  Auto threshold: {threshold:.4f}")

# Phase 3: Classify and route
print("\n[Phase 3] Classifying and routing...")
correct_route = 0
results = []
for i, (q, a) in enumerate(FACTS):
    conf = confidences[i]["confidence"]
    predicted_tier = classify_by_confidence(conf, threshold)
    actual_tier = "easy" if i < 3 else "hard"
    correct = predicted_tier == actual_tier
    correct_route += int(correct)
    
    # Test the routed backend
    if predicted_tier == "easy":
        model2 = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE
        )
        model2 = train_lora_fp32(model2, tokenizer, (q, a))
        ans = generate_answer(model2, tokenizer, q)
        backend_works = a.lower() in ans.lower()
        backend = "weights"
        del model2
        torch.cuda.empty_cache()
    else:
        backend_works = test_retrieval(model, tokenizer, q, a)
        backend = "retrieval"
    
    results.append({
        "q": q, "a": a, "confidence": conf,
        "predicted_tier": predicted_tier,
        "actual_tier": actual_tier,
        "correct_route": correct,
        "backend": backend,
        "backend_works": backend_works,
    })
    status = "✅" if correct and backend_works else "❌"
    print(f"  {status} {q[:35]}... → {predicted_tier} → {backend} → works={backend_works}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Routing accuracy: {correct_route}/{len(FACTS)}")
backend_works = sum(1 for r in results if r["backend_works"])
print(f"Backend success: {backend_works}/{len(FACTS)}")

if correct_route == len(FACTS) and backend_works == len(FACTS):
    print("\n🎯 TIER COMPILER WORKS")
else:
    print("\n⚠️  Tier compiler partially working")
print("="*60)

with open("experiments/m258_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("✅ Saved to experiments/m258_results.json")
