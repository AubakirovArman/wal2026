#!/usr/bin/env python3
"""M43zf: Encode only o_proj and down_proj in early layers (0-19)."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from route_encoder import encode_routes, rel_mse
from codebook import build_codebook

model_name = "unsloth/Llama-3.3-70B-Instruct"
max_length = 2048
stride = 512
max_samples = 10

SCALAR_K = 128
SCALAR_LMAX = 8
COARSE_LADDER = [0.5 ** i for i in range(12)]


def encode_scalar_drl_batched(w_norm, ladder, l_max, K_target):
    enc = encode_routes(w_norm, ladder, stop_threshold=0.0, l_max=l_max)
    cb, ids = build_codebook(enc.digits, enc.stop_depth, l_max)
    route_values = cb.digits.to(torch.float32) @ ladder[:l_max]
    K = cb.keys.numel()
    freq = torch.bincount(ids.reshape(-1).long(), minlength=K).float()
    _, top_idx = freq.topk(min(K_target, K))
    centers = route_values[top_idx].clone()
    for _ in range(10):
        dist = (route_values.unsqueeze(1) - centers.unsqueeze(0)).abs()
        assignments = dist.argmin(dim=1)
        new_centers = torch.zeros_like(centers)
        for c in range(min(K_target, K)):
            mask = assignments == c
            if mask.any():
                wg = freq[mask]
                new_centers[c] = (route_values[mask] * wg).sum() / wg.sum()
        centers = new_centers
    w_flat = w_norm.reshape(-1)
    w_assignments = torch.empty_like(w_flat, dtype=torch.int64)
    batch_size = 512 * 1024
    for start in range(0, w_flat.numel(), batch_size):
        end = min(start + batch_size, w_flat.numel())
        w_assignments[start:end] = (w_flat[start:end].unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
    w_hat_norm = centers[w_assignments].reshape(w_norm.shape)
    return w_hat_norm


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

stats = {"encoded": 0, "skipped": 0}
for name, param in list(model.named_parameters()):
    if len(param.shape) != 2:
        stats["skipped"] += 1
        continue
    if "embed_tokens" in name or "lm_head" in name:
        stats["skipped"] += 1
        continue

    layer_num = None
    if "model.layers." in name:
        parts = name.split(".")
        layer_num = int(parts[2])

    if layer_num is not None and layer_num < 20 and ("o_proj" in name or "down_proj" in name):
        w = param.data.float()
        row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        w_norm = w / row_scale
        ladder = torch.tensor(COARSE_LADDER, device=w.device, dtype=torch.float32)
        w_hat_norm = encode_scalar_drl_batched(w_norm, ladder, SCALAR_LMAX, SCALAR_K)
        w_hat = w_hat_norm * row_scale
        param.data.copy_(w_hat.to(param.dtype))
        stats["encoded"] += 1
    else:
        stats["skipped"] += 1

print(f"Done: {stats}")

# PPL
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
print(f"\nEarly o_proj+down_proj scalar K=128 PPL: {ppl.item():.2f}")
