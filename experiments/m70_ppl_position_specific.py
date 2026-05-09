"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
#!/usr/bin/env python3
"""M70: Full 70B PPL with position-specific scalar quantization.

Tests K=128 (7 bits/weight) and K=256 (8 bits/weight).
Uses M61 PPL parameters: max_length=2048, stride=512, max_samples=16.
"""
import torch
import time
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

def uniform_quantize(w_norm, K):
    """Uniform quantization per column."""
    col_min = w_norm.min(dim=0, keepdim=True)[0]
    col_max = w_norm.max(dim=0, keepdim=True)[0]
    col_range = (col_max - col_min).clamp_min(1e-8)
    scaled = (w_norm - col_min) / col_range * (K - 1)
    indices = torch.round(scaled).clamp(0, K - 1)
    recon = indices.float() / (K - 1) * col_range + col_min
    return recon


def run_ppl(model, tokenizer, device):
    """Match M61 PPL parameters exactly."""
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
    model_name = "unsloth/Llama-3.3-70B-Instruct"
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
    
    # Save original state_dict
    print("Saving original state_dict to CPU...")
    t0 = time.time()
    original_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"  Saved in {time.time()-t0:.1f}s")
    
    # Baseline PPL
    print("\n=== BASELINE PPL ===")
    t0 = time.time()
    baseline_ppl, n_steps, ntok = run_ppl(model, tokenizer, device)
    print(f"Baseline PPL: {baseline_ppl:.4f}  ({n_steps} steps, {ntok} tokens, {time.time()-t0:.1f}s)")
    
    for K in [128, 256]:
        bits = K.bit_length() - 1
        ratio = 16 / bits
        print(f"\n{'='*60}")
        print(f"Testing K={K} ({bits} bits/weight, {ratio:.1f}x compression)")
        print(f"{'='*60}")
        
        t0 = time.time()
        modified = 0
        for name, p in model.named_parameters():
            if len(p.shape) == 2:
                w = p.data.float()
                row_scale = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
                w_norm = w / row_scale
                recon_norm = uniform_quantize(w_norm, K)
                recon = recon_norm.to(p.dtype) * p.data.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8).to(p.dtype)
                p.data.copy_(recon)
                modified += 1
        
        print(f"  Encoded {modified} layers in {time.time()-t0:.1f}s")
        
        t0 = time.time()
        ppl, n_steps, ntok = run_ppl(model, tokenizer, device)
        print(f"  PPL: {ppl:.4f}  (delta: {ppl - baseline_ppl:+.4f})")
        print(f"  Inference time: {time.time()-t0:.1f}s")
        
        # Restore
        print("  Restoring original weights...")
        t0 = time.time()
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in original_state:
                    p.copy_(original_state[name].to(p.device))
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  Restored in {time.time()-t0:.1f}s")
    
    print(f"\n{'='*60}")
    print("M70 complete.")


if __name__ == "__main__":
    main()
