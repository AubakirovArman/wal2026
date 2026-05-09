import sys

with open('experiments/m27_ptdp_collect.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip = 0
for i, line in enumerate(lines):
    if skip > 0:
        skip -= 1
        continue
    if 'ids = self.stage_ids[stage_idx][row_a:row_b, block_a:block_b].reshape(-1)' in line:
        new_lines.append(line)
        new_lines.append(lines[i+1]) # counts = ...
        new_lines.append('                    _cb_norms = self.codebook_norms[stage_idx].index_select(0, ids)\n')
        new_lines.append('                    weights = (\n')
        new_lines.append('                        row_scale_tile.unsqueeze(1)\n')
        new_lines.append('                        * block_energy_tile.unsqueeze(0)\n')
        new_lines.append('                        * _cb_norms.view(row_b - row_a, block_b - block_a)\n')
        new_lines.append('                    ).reshape(-1)\n')
        skip = 5 # skip the mess I made
    else:
        new_lines.append(line)

with open('experiments/m27_ptdp_collect.py', 'w') as f:
    f.writelines(new_lines)
