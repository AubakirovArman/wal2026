import torch
import sys
import os
import torch.nn as nn

# Add src to path if necessary
sys.path.append(os.getcwd())

from src.encoding_io import load_grouped_encoding_map
from src.runtime import PackedGroupedBlockRVQLinear

def main():
    try:
        path = 'results/m25_l54_q_gu_encodings.pt'
        encoding_map = load_grouped_encoding_map(path)
        
        layer_key = 'model.layers.54.self_attn.q_proj'
        if layer_key not in encoding_map:
            print(f"Key {layer_key} not found in encoding map")
            return

        encoding = encoding_map[layer_key]
        
        m_fast = PackedGroupedBlockRVQLinear(
            enc=encoding,
            matmul_strategy='full_weight_fast',
            bias=None
        ).cuda().to(torch.bfloat16)

        m_hot = PackedGroupedBlockRVQLinear(
            enc=encoding,
            matmul_strategy='triton_hot_cold_persistent',
            local_palette_group_cols=256,
            hot_topk=16,
            bias=None
        ).cuda().to(torch.bfloat16)

        in_features = m_fast.in_features
        x = torch.randn(32, in_features, device='cuda', dtype=torch.bfloat16)

        with torch.no_grad():
            out_fast = m_fast(x)
            out_hot = m_hot(x)

        diff = (out_fast - out_hot).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        is_allclose = torch.allclose(out_fast, out_hot, atol=1e-3, rtol=1e-3)

        print(f"shape_fast {list(out_fast.shape)}")
        print(f"shape_hotprefix {list(out_hot.shape)}")
        print(f"max_abs_diff {max_diff}")
        print(f"mean_abs_diff {mean_diff}")
        print(f"allclose {is_allclose}")

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
