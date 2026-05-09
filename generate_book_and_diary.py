#!/usr/bin/env python3
"""
Generate detailed book entries and dev diary entries for M386-M620.
Reads all result JSON files and produces markdown.
"""
import json, glob, os, re
from pathlib import Path

RESULT_DIR = "experiments"
BOOK_DIR = "book"
DIARY = "docs/dev_diary_ru.md"

def parse_module_info(path):
    """Parse module number and title from filename."""
    name = os.path.basename(path)
    m = re.match(r'(m\d+)_(.+)\.py', name)
    if not m:
        return None, None
    mod_num = m.group(1).upper()
    raw_title = m.group(2)
    title = raw_title.replace('_', ' ').title()
    return mod_num, title

def get_result(path):
    """Read result JSON if exists."""
    result_path = path.replace('.py', '_results.json')
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f)
        except:
            return None
    return None

def format_result(data):
    """Format result dict as readable markdown."""
    if not data:
        return "_Результатов нет_"
    lines = []
    for k, v in data.items():
        if k == "pass":
            status = "✅ PASS" if v else "❌ FAIL"
            lines.append(f"- **Статус**: {status}")
        elif isinstance(v, (int, float, str, bool)):
            lines.append(f"- **{k}**: {v}")
    return "\n".join(lines)

def generate_book_entry(mod_num, title, result_data, py_path):
    """Generate a single book markdown file."""
    safe_title = title.replace(' ', '_').replace('/', '_')
    filename = f"{BOOK_DIR}/{mod_num}_{safe_title}.md"
    
    # Read source for description
    desc = ""
    try:
        with open(py_path) as f:
            first_line = f.readline()
            if '"""' in first_line or "'''" in first_line:
                lines = []
                for line in f:
                    if '"""' in line or "'''" in line:
                        break
                    lines.append(line.strip())
                desc = " ".join(lines[:3])
    except:
        pass
    
    content = f"""# {mod_num} — {title}

## Описание
{desc or 'Эксперимент ' + mod_num}

## Результаты
{format_result(result_data)}

## Дата
2026-04-20 — 2026-05-06

## Статус
{'✅ Завершён' if (result_data and result_data.get('pass')) else '⚠️ Требует внимания'}
"""
    with open(filename, "w") as f:
        f.write(content)
    return filename

def generate_diary_section(entries):
    """Generate a diary section and append it."""
    lines = []
    lines.append("\n## Полный отчёт M386–M620\n")
    lines.append(f"**Дата**: 2026-05-06\n")
    lines.append(f"**Всего модулей**: {len(entries)}\n")
    lines.append("---\n")
    
    for mod_num, title, result_data, _ in entries:
        status = "✅" if (result_data and result_data.get('pass')) else "⚠️"
        lines.append(f"**{mod_num}** — {title}: {status}\n")
    
    lines.append("---\n")
    lines.append(f"**Итого**: {len(entries)} модулей обработано.\n")
    
    return "".join(lines)

def main():
    os.makedirs(BOOK_DIR, exist_ok=True)
    
    # Collect all M386-M620 experiments
    entries = []
    for py_path in sorted(glob.glob(f"{RESULT_DIR}/m*.py")):
        mod_num, title = parse_module_info(py_path)
        if not mod_num:
            continue
        num = int(mod_num[1:])
        if not (386 <= num <= 620):
            continue
        
        result_data = get_result(py_path)
        filename = generate_book_entry(mod_num, title, result_data, py_path)
        entries.append((mod_num, title, result_data, filename))
    
    # Generate and append diary section
    diary_section = generate_diary_section(entries)
    with open(DIARY, "a") as f:
        f.write(diary_section)
    
    print(f"Generated {len(entries)} book entries")
    print(f"Appended to {DIARY}")

if __name__ == "__main__":
    main()
