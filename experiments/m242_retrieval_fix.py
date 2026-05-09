"""
M242 — Retrieval Prompt Injection Fix

Hypothesis: M238's retrieval-only mode failed (0/3 easy) because
retrieval context was not properly prepended to the prompt.

Fix: Explicitly prepend context with [CONTEXT] / [QUESTION] markers.
Test: easy facts should now work via retrieval.
"""

import os, sys, json, torch, gc, math
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal/src")
sys.path.insert(0, "/mnt/hf_model_weights/arman/3bit/wal")

from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HOME"] = "/mnt/hf_model_weights"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda:0"

EASY_FACTS = [
    ("What is the capital of France?", "Paris"),
    ("Where is the Eiffel Tower located?", "Paris"),
    ("What is the longest river in the world?", "Nile"),
]

HARDFACTS = [
    ("Who invented the telephone?", "Antonio Meucci"),
    ("Who wrote 1984?", "Aldous Huxley"),
    ("Who discovered radioactivity?", "Nikola Tesla"),
]

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE, low_cpu_mem_usage=True
    )
    return model, tokenizer

def generate_answer(model, tokenizer, prompt, max_new_tokens=15):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def build_retrieval_prompt(query, facts):
    """Build prompt with retrieval context prepended."""
    # Find best matching fact (exact match for simplicity)
    context = ""
    for q, a in facts:
        if q.lower().strip("?") in query.lower().strip("?") or query.lower().strip("?") in q.lower().strip("?"):
            context = f"{q} {a}."
            break
    
    if not context:
        # No match found
        return query
    
    # Format: context then question
    prompt = f"[CONTEXT]: {context}\n[QUESTION]: {query}\n[ANSWER]:"
    return prompt

def test_retrieval(model, tokenizer, facts):
    survive = 0
    for q, a in facts:
        prompt = build_retrieval_prompt(q, facts)
        ans = generate_answer(model, tokenizer, prompt)
        if a.lower() in ans.lower():
            survive += 1
    return survive

def run():
    print("=" * 60)
    print("M242 — Retrieval Prompt Injection Fix")
    print("=" * 60)
    
    model, tokenizer = load_model()
    
    print("\n[1] Retrieval-only for easy facts...")
    easy_survive = test_retrieval(model, tokenizer, EASY_FACTS)
    print(f"  Easy: {easy_survive}/{len(EASY_FACTS)}")
    
    print("\n[2] Retrieval-only for hard facts...")
    hard_survive = test_retrieval(model, tokenizer, HARDFACTS)
    print(f"  Hard: {hard_survive}/{len(HARDFACTS)}")
    
    print("\n[3] Contamination: unrelated question with wrong context...")
    wrong_context = "[CONTEXT]: The capital of France is Paris.\n[QUESTION]: What is the capital of Germany?\n[ANSWER]:"
    ans = generate_answer(model, tokenizer, wrong_context)
    contaminated = "Paris" in ans
    print(f"  Answer: {ans}")
    print(f"  Contaminated: {contaminated} (should be False)")
    
    output = {
        "easy_survival": easy_survive,
        "hard_survival": hard_survive,
        "contaminated": contaminated,
    }
    with open("experiments/m242_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n✅ Saved to experiments/m242_results.json")
    
    if easy_survive == len(EASY_FACTS) and not contaminated:
        print("\n✅ RETRIEVAL FIX WORKS")
    else:
        print("\n⚠️  Retrieval partially working — needs more tuning")

if __name__ == "__main__":
    run()
