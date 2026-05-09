#!/usr/bin/env python3
"""Generate diary entries for all missing experiments."""
import os
import re
import ast
from pathlib import Path

PROJECT_ROOT = Path("/mnt/hf_model_weights/arman/3bit/dwl2_dynamic_route")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DIARY_DIR = PROJECT_ROOT / "docs" / "diary"
DIARY_RU = PROJECT_ROOT / "docs" / "dev_diary_ru.md"


def extract_docstring(filepath):
    """Extract module docstring from Python file."""
    try:
        with open(filepath) as f:
            source = f.read()
        tree = ast.parse(source)
        doc = ast.get_docstring(tree)
        return doc or ""
    except Exception:
        return ""


def extract_config_from_source(source):
    """Extract common config variables from source."""
    configs = []
    for pattern, label in [
        (r'K\s*=\s*(\d+)', 'K'),
        (r'C\s*=\s*(\d+)', 'C'),
        (r'l_max\s*=\s*(\d+)', 'l_max'),
        (r'K_ATOMS\s*=\s*(\d+)', 'K_ATOMS'),
        (r'C_COEFFS\s*=\s*(\d+)', 'C_COEFFS'),
        (r'KMEANS_ITERS\s*=\s*(\d+)', 'KMEANS_ITERS'),
        (r'LLOYD_MAX_ITERS\s*=\s*(\d+)', 'LLOYD_MAX_ITERS'),
        (r'residual_threshold\s*=\s*([\d.]+)', 'residual_threshold'),
        (r'batch\s*=\s*([\d_]+)', 'batch'),
        (r'iters\s*=\s*(\d+)', 'iters'),
        (r'max_l1\s*=\s*(\d+)', 'max_l1'),
        (r'tile_size\s*=\s*(\d+)', 'tile_size'),
        (r'block_size\s*=\s*(\d+)', 'block_size'),
        (r'seq_len\s*=\s*(\d+)', 'seq_len'),
        (r'num_steps\s*=\s*(\d+)', 'num_steps'),
        (r'threshold\s*=\s*([\d.]+)', 'threshold'),
    ]:
        matches = re.findall(pattern, source)
        if matches:
            configs.append(f"{label}={matches[0]}")
    return configs


def find_diary_mentions(exp_name, diary_text):
    """Find mentions of experiment in dev_diary_ru.md."""
    mentions = []
    lines = diary_text.split('\n')
    for i, line in enumerate(lines):
        if exp_name.replace('_', '') in line.replace('_', '').replace(' ', '') or \
           exp_name.split('_')[0] in line:
            # Get surrounding context
            start = max(0, i - 2)
            end = min(len(lines), i + 3)
            context = '\n'.join(lines[start:end])
            if len(context) > 20:
                mentions.append(context)
    # Deduplicate and limit
    seen = set()
    unique = []
    for m in mentions:
        key = m[:100]
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique[:3]


def generate_diary_entry(exp_name, filepath, diary_text):
    """Generate a diary markdown entry for an experiment."""
    doc = extract_docstring(filepath)
    with open(filepath) as f:
        source = f.read()
    configs = extract_config_from_source(source)
    
    mentions = find_diary_mentions(exp_name, diary_text)
    
    # Try to determine result/status from source
    result = "Unknown"
    if 'ppl' in source.lower() or 'perplexity' in source.lower():
        result = "PPL evaluation"
    elif 'benchmark' in source.lower() or 'bench' in source.lower():
        result = "Benchmark"
    elif 'encode' in source.lower():
        result = "Encode test"
    elif 'debug' in source.lower():
        result = "Debug/diagnostic"
    elif 'prototype' in source.lower():
        result = "Prototype"
    elif 'roundtrip' in source.lower() or 'round_trip' in source.lower():
        result = "Round-trip test"
    elif 'runtime' in source.lower():
        result = "Runtime test"
    
    # Check for known outcomes in source
    notes = []
    if "FAIL" in source or "catastrophic" in source.lower():
        notes.append("Likely negative result")
    if "PASS" in source:
        notes.append("Has PASS/FAIL asserts")
    
    lines = [
        f"# {exp_name.upper().replace('_', ' ')}",
        "",
        "## Date",
        "2026 (exact date from git log or experiment run)",
        "",
        "## Goal",
    ]
    
    if doc:
        # First line of docstring as goal
        goal = doc.strip().split('\n')[0].strip()
        lines.append(goal)
    else:
        lines.append(f"Run and evaluate {exp_name}.")
    
    lines.extend([
        "",
        "## Configuration",
    ])
    
    if configs:
        lines.append(", ".join(configs))
    else:
        lines.append("See source code for full configuration.")
    
    lines.extend([
        "",
        "## Method / What was tested",
    ])
    
    if doc:
        # Add rest of docstring
        rest = '\n'.join(doc.strip().split('\n')[1:]).strip()
        if rest:
            lines.append(rest)
        else:
            lines.append(f"See `{filepath.relative_to(PROJECT_ROOT)}` for implementation details.")
    else:
        lines.append(f"See `{filepath.relative_to(PROJECT_ROOT)}` for implementation details.")
    
    lines.extend([
        "",
        "## Result",
        f"{result}."
    ])
    
    if notes:
        lines.append(" ".join(notes))
    
    lines.extend([
        "",
        "## Artifacts",
        f"- `{filepath.relative_to(PROJECT_ROOT)}`",
    ])
    
    # Add log file if exists
    log_file = filepath.with_suffix('.log')
    if log_file.exists():
        lines.append(f"- `{log_file.relative_to(PROJECT_ROOT)}`")
    
    lines.extend([
        "",
        "## Notes from dev_diary_ru.md",
    ])
    
    if mentions:
        for m in mentions:
            lines.append(f"```")
            lines.append(m)
            lines.append(f"```")
            lines.append("")
    else:
        lines.append("No specific mention in dev_diary_ru.md.")
    
    return '\n'.join(lines)


def main():
    diary_text = ""
    if DIARY_RU.exists():
        with open(DIARY_RU) as f:
            diary_text = f.read()
    
    # Find all experiment scripts
    exp_files = sorted(EXPERIMENTS_DIR.glob("m*.py"))
    
    created = 0
    skipped = 0
    
    for filepath in exp_files:
        if filepath.name.startswith('_'):
            continue
        
        exp_name = filepath.stem
        # Check if diary entry already exists
        prefix = exp_name.split('_')[0]  # e.g., m1, m43, m61
        existing = list(DIARY_DIR.glob(f"{prefix}*.md"))
        
        if existing:
            skipped += 1
            continue
        
        entry = generate_diary_entry(exp_name, filepath, diary_text)
        
        # Write to diary
        out_path = DIARY_DIR / f"{exp_name}.md"
        with open(out_path, 'w') as f:
            f.write(entry)
        
        created += 1
        print(f"Created: {out_path.name}")
    
    print(f"\nDone: {created} created, {skipped} already existed.")


if __name__ == "__main__":
    main()
