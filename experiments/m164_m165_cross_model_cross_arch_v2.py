#!/usr/bin/env python3
"""M164 + M165 v2 — Cross-Model Vocabulary & Cross-Architecture (fast CPU).

Uses public models: Llama-3.1-8B (base), gpt2 (cross-model), distilbert (cross-arch).
"""
import torch, json, sys
sys.path.insert(0, '/mnt/hf_model_weights/arman/3bit/wal/src')

from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from wal.v1 import build_l0_atoms, build_coeff_table, wal_encode_v1


def encode_weights(weights, K=64):
    flat = torch.cat([w.reshape(-1) for w in weights])
    atoms = build_l0_atoms(flat, K=K, iters=1)
    coeffs = build_coeff_table(flat, atoms, C=8, iters=1)
    mses = []
    for w in weights:
        _, recon = wal_encode_v1(w.reshape(-1), atoms, coeffs, batch=65_536)
        mse = ((w.reshape(-1) - recon) ** 2).mean().item()
        mses.append(mse)
    return atoms, coeffs, mses


def main():
    print("=" * 60)
    print("M164 + M165 — Cross-Model & Cross-Architecture")
    print("=" * 60)
    
    # Collect sample weights from each model
    print("\n--- Loading Llama-3.1-8B (base) ---")
    llama = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    llama_weights = []
    for li in [0, 8, 16, 24, 31]:
        for m in ['q_proj', 'v_proj', 'gate_proj']:
            w = getattr(llama.model.layers[li].self_attn if m != 'gate_proj' else llama.model.layers[li].mlp, m).weight.data.float()
            llama_weights.append(w)
    del llama
    
    print("--- Loading gpt2 (cross-model, same arch family) ---")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", device_map="cpu")
    gpt2_weights = []
    for li in [0, min(11, len(gpt2.transformer.h)-1)]:
        layer = gpt2.transformer.h[li]
        for path in ['attn.c_attn', 'attn.c_proj', 'mlp.c_fc']:
            parts = path.split('.')
            mod = layer
            for p in parts:
                mod = getattr(mod, p)
            w = mod.weight.data.float()
            gpt2_weights.append(w)
    del gpt2
    
    print("--- Loading distilbert (cross-architecture) ---")
    bert = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased", device_map="cpu")
    bert_weights = []
    for li in [0, min(5, len(bert.distilbert.transformer.layer)-1)]:
        layer = bert.distilbert.transformer.layer[li]
        for m in ['q_lin', 'k_lin', 'v_lin', 'out_lin', 'ffn.lin1', 'ffn.lin2']:
            mod = layer.attention if 'lin' in m and m != 'ffn.lin1' and m != 'ffn.lin2' else layer
            if m == 'ffn.lin1': mod = layer.ffn.lin1
            elif m == 'ffn.lin2': mod = layer.ffn.lin2
            else: mod = getattr(mod, m)
            w = mod.weight.data.float()
            bert_weights.append(w)
    del bert
    
    # M164: Build vocab on Llama, test on gpt2
    print("\n--- M164: Llama vocab on gpt2 ---")
    llama_atoms, llama_coeffs, llama_mses = encode_weights(llama_weights[:3], K=64)
    gpt2_mses_cross = []
    for w in gpt2_weights[:3]:
        _, recon = wal_encode_v1(w.reshape(-1), llama_atoms, llama_coeffs, batch=65_536)
        mse = ((w.reshape(-1) - recon) ** 2).mean().item()
        gpt2_mses_cross.append(mse)
    
    gpt2_atoms, gpt2_coeffs, gpt2_mses_native = encode_weights(gpt2_weights[:3], K=64)
    
    print(f"  Llama native MSE:  {sum(llama_mses)/len(llama_mses):.2e}")
    print(f"  gpt2 cross MSE:    {sum(gpt2_mses_cross)/len(gpt2_mses_cross):.2e}")
    print(f"  gpt2 native MSE:   {sum(gpt2_mses_native)/len(gpt2_mses_native):.2e}")
    print(f"  Cross/native ratio: {sum(gpt2_mses_cross)/sum(gpt2_mses_native):.2f}")
    
    # M165: Llama vocab on distilbert
    print("\n--- M165: Llama vocab on distilbert ---")
    bert_mses_cross = []
    for w in bert_weights[:3]:
        _, recon = wal_encode_v1(w.reshape(-1), llama_atoms, llama_coeffs, batch=65_536)
        mse = ((w.reshape(-1) - recon) ** 2).mean().item()
        bert_mses_cross.append(mse)
    
    bert_atoms, bert_coeffs, bert_mses_native = encode_weights(bert_weights[:3], K=64)
    
    print(f"  Llama native MSE:   {sum(llama_mses)/len(llama_mses):.2e}")
    print(f"  BERT cross MSE:     {sum(bert_mses_cross)/len(bert_mses_cross):.2e}")
    print(f"  BERT native MSE:    {sum(bert_mses_native)/len(bert_mses_native):.2e}")
    print(f"  Cross/native ratio: {sum(bert_mses_cross)/sum(bert_mses_native):.2f}")
    
    results = {
        'm164': {
            'llama_native_mse': sum(llama_mses)/len(llama_mses),
            'gpt2_cross_mse': sum(gpt2_mses_cross)/len(gpt2_mses_cross),
            'gpt2_native_mse': sum(gpt2_mses_native)/len(gpt2_mses_native),
            'cross_native_ratio': sum(gpt2_mses_cross)/sum(gpt2_mses_native),
        },
        'm165': {
            'llama_native_mse': sum(llama_mses)/len(llama_mses),
            'bert_cross_mse': sum(bert_mses_cross)/len(bert_mses_cross),
            'bert_native_mse': sum(bert_mses_native)/len(bert_mses_native),
            'cross_native_ratio': sum(bert_mses_cross)/sum(bert_mses_native),
        }
    }
    
    with open('/mnt/hf_model_weights/arman/3bit/wal/experiments/m164_m165_cross_model_cross_arch.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Saved")

if __name__ == "__main__":
    main()
