import torch
import sys
from src.encoding_io import load_grouped_encoding_map
from src.runtime import PackedGroupedBlockRVQLinear

print("Starting script...")
try:
    path = "results/m25_l54_q_gu_encodings.pt"
    print(f"Loading {path}...")
    encodings = load_grouped_encoding_map(path)
    keys = list(encodings.keys())
    print("Keys:", keys)
    
    first_key = keys[0]
    encoding = encodings[first_key]
    print(f"Selected key: {first_key}")
    
    print("Instantiating layer...")
    layer = PackedGroupedBlockRVQLinear(
        encoding, 
        matmul_strategy='local_palette', 
        local_palette_group_cols=256
    ).cuda()
    
    print("Calling _local_palette_state()...")
    state = layer._local_palette_state()
    
    print("Identifying palette_sizes...")
    palette_sizes = None
    if isinstance(state, tuple):
        for item in state:
            if isinstance(item, torch.Tensor) and item.dim() == 1 and item.dtype in [torch.int32, torch.int64]:
                palette_sizes = item
                break
    
    if palette_sizes is not None:
        num_tiles = len(palette_sizes)
        print(f"Number of tiles: {num_tiles}")
        print(f"Mean palette size: {palette_sizes.float().mean().item():.2f}")
        print(f"Max palette size: {palette_sizes.max().item()}")
        print(f"Total palette values: {palette_sizes.sum().item()}")
        print("Succeeded: True")
    else:
        print(f"Could not identify palette_sizes. State type: {type(state)}")
        if isinstance(state, tuple):
             print(f"Tuple elements: {[type(x) for x in state]}")

except Exception as e:
    print("An error occurred:")
    import traceback
    traceback.print_exc()
