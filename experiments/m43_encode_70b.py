#!/usr/bin/env python3
"""M43: Apply hybrid encoder to Llama 3.3 70B and measure PPL."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from m39_hybrid_encoder import hybrid_encode

model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512
max_samples = 10

# Encoder params
VRE_CB_SIZE = 1024
VRE_LMAX = 10
SCALAR_K = 128
SCALAR_LMAX = 8
SPIKY_THRESHOLD = 0.08
COARSE_LADDER = [0.5 ** i for i in range(12)]

print(f"Loading {model_name}...")
max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

print("Applying hybrid encoder...")
stats = {"vre": 0, "scalar": 0, "skipped": 0}
for name, param in list(model.named_parameters()):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue

    w = param.data.float().cpu()
    result = hybrid_encode(
        w,
        coarse_ladder=COARSE_LADDER,
        vre_cb_size=VRE_CB_SIZE,
        vre_lmax=VRE_LMAX,
        scalar_K=SCALAR_K,
        scalar_lmax=SCALAR_LMAX,
        spiky_threshold=SPIKY_THRESHOLD,
    )
    is_spiky = result.get("is_spiky")
    w_hat = result["w_hat"].to(param.dtype).to(param.device)
    param.data.copy_(w_hat)
    del w, w_hat, result
    torch.cuda.empty_cache()

    if is_spiky:
        stats["vre"] += 1
    else:
        stats["scalar"] += 1

    total_enc = stats["vre"] + stats["scalar"]
    if total_enc % 10 == 0 or total_enc <= 5:
        print(f"  Encoded {total_enc} params (VRE={stats['vre']}, scalar={stats['scalar']})")

print(f"Encoding done: {stats}")

# PPL
print("Loading WikiText-2...")
ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(ds["text"][:])
encodings = tokenizer(text, return_tensors="pt")
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
num_steps = 0
for begin_loc in range(0, seq_len, stride):
    if num_steps >= max_samples:
        break
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids.to(model.device), labels=target_ids.to(model.device))
        neg_log_likelihood = outputs.loss * trg_len

    nlls.append(neg_log_likelihood)
    prev_end_loc = end_loc
    num_steps += 1
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
print(f"\nEncoded PPL (first {end_loc} tokens): {ppl.item():.2f}")
