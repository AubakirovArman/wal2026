"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M72: Full-model PPL sweep for M69 position-specific quantization.

Tests K=16,32,64,128,256. All in one run: load once, encode in-place, PPL, restore.
Uses uniform quantization (fast encode).
"""
import torch
import time
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "unsloth/Llama-3.3-70B-Instruct"


def uniform_quantize_all(model, K):
    """Encode all 2D params with position-specific uniform quantization."""
    modified = 0
    for name, p in model.named_parameters():
        if len(p.shape) == 2:
            w = p.data.float()
            row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            w_norm = w / row_scale
            col_min = w_norm.min(dim=0, keepdim=True)[0]
            col_max = w_norm.max(dim=0, keepdim=True)[0]
            col_range = (col_max - col_min).clamp_min(1e-8)
            scaled = (w_norm - col_min) / col_range * (K - 1)
            indices = torch.round(scaled).clamp(0, K - 1)
            recon_norm = indices.float() / (K - 1) * col_range + col_min
            recon = recon_norm.to(p.dtype) * p.data.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).to(p.dtype)
            p.data.copy_(recon)
            modified += 1
    return modified


def run_ppl(model, tokenizer, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"][:])
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    max_length = 2048
    stride = 512
    max_samples = 16
    nlls = []
    prev_end_loc = 0
    num_steps = 0
    for begin_loc in range(0, seq_len, stride):
        if num_steps >= max_samples:
            break
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        num_steps += 1
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item(), num_steps, end_loc


def main():
    print(f"Loading {model_name}...")
    max_memory = {2: "150GiB", 3: "150GiB", "cpu": "0GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_memory, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    device = next(model.parameters()).device
    
    # Save state_dict once
    print("Saving state_dict to CPU...")
    t0 = time.time()
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"  Saved in {time.time()-t0:.1f}s")
    
    # Baseline
    print("\n=== BASELINE ===")
    t0 = time.time()
    baseline_ppl, n_steps, ntok = run_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.4f}  ({n_steps} steps, {ntok} tokens, {time.time()-t0:.1f}s)")
    
    print(f"\n{'='*70}")
    print(f"{'K':>6} | {'Bits/w':>8} | {'Ratio':>8} | {'PPL':>10} | {'Delta':>10} | {'Status':>8}")
    print(f"{'-'*70}")
    
    for K in [16, 32, 64, 128, 256]:
        bits = K.bit_length() - 1
        ratio = 16 / bits
        
        # Encode
        t0 = time.time()
        modified = uniform_quantize_all(model, K)
        encode_time = time.time() - t0
        
        # PPL
        t0 = time.time()
        ppl, n_steps, ntok = run_ppl(model, tokenizer, device)
        ppl_time = time.time() - t0
        delta = ppl - baseline_ppl
        
        status = "PASS" if delta < 0.05 else ("DEGRADE" if delta < 0.5 else "FAIL")
        print(f"{K:>6} | {bits:>8.1f} | {ratio:>8.1f}x | {ppl:>10.4f} | {delta:>+10.4f} | {status:>8}  (enc {encode_time:.1f}s, inf {ppl_time:.1f}s)")
        
        # Restore
        t0 = time.time()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in original_state:
                    p.copy_(original_state[name].to(p.device))
        torch.cuda.empty_cache()
        gc.collect()
        # print(f"  Restored in {time.time()-t0:.1f}s")
    
    print(f"{'='*70}")
    print("\nM72 complete.")


if __name__ == "__main__":
    main()
