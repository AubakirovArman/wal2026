"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M44: Full WikiText-2 baseline for Llama 3.3 70B — authoritative quality gate."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import time
import json
from pathlib import Path

model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

print(f"Loading {model_name}...")
t0 = time.time()
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)
load_time = time.time() - t0
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

print(f"Model loaded in {load_time:.1f}s")
print(f"Loading WikiText-2 test split...")
ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"][:])
encodings = tokenizer(text, return_tensors="pt")
seq_len = encodings.input_ids.size(1)
print(f"Total tokens: {seq_len:,}")

nlls = []
prev_end_loc = 0
num_steps = 0
step_times = []

t0 = time.time()
for begin_loc in range(0, seq_len, stride):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    step_t0 = time.time()
    with torch.no_grad():
        outputs = model(input_ids.to(model.device), labels=target_ids.to(model.device))
        neg_log_likelihood = outputs.loss * trg_len
    step_times.append(time.time() - step_t0)

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    num_steps += 1

    if num_steps % 50 == 0:
        elapsed = time.time() - t0
        est_total = elapsed / num_steps * (seq_len / stride)
        print(f"  Step {num_steps}/{int(seq_len/stride)+1} | {end_loc:,} tokens | "
              f"elapsed {elapsed:.0f}s | est total {est_total:.0f}s")

    if end_loc == seq_len:
        break

total_time = time.time() - t0
ppl = torch.exp(torch.stack(nlls).sum() / end_loc)

results = {
    "model": model_name,
    "task": "full_wikitext2_baseline",
    "metric": "perplexity",
    "ppl": ppl.item(),
    "tokens_evaluated": end_loc,
    "num_steps": num_steps,
    "max_length": max_length,
    "stride": stride,
    "load_time_sec": load_time,
    "eval_time_sec": total_time,
    "mean_step_time_sec": sum(step_times) / len(step_times),
    "max_step_time_sec": max(step_times),
    "device_map": str(model.hf_device_map),
}

out_path = RESULTS_DIR / "m44_full_wikitext2_baseline.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"BASELINE RESULTS")
print(f"{'='*60}")
print(f"PPL: {ppl.item():.4f}")
print(f"Tokens: {end_loc:,}")
print(f"Steps: {num_steps}")
print(f"Load time: {load_time:.1f}s")
print(f"Eval time: {total_time:.1f}s")
print(f"Mean step: {sum(step_times)/len(step_times):.3f}s")
print(f"Results saved: {out_path}")
