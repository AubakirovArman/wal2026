#!/usr/bin/env python3
"""
Generate DETAILED book entries for M386-M620.
Includes full result data, docstrings, and analysis.
"""
import json, glob, os, re, textwrap
from pathlib import Path

RESULT_DIR = "experiments"
BOOK_DIR = "book"

def parse_module_info(path):
    name = os.path.basename(path)
    m = re.match(r'(m\d+)_(.+)\.py', name)
    if not m:
        return None, None
    mod_num = m.group(1).upper()
    raw_title = m.group(2)
    title = raw_title.replace('_', ' ').title()
    return mod_num, title

def get_docstring(path):
    """Extract docstring from Python file."""
    try:
        with open(path) as f:
            content = f.read()
        ds_match = re.search(r'"""(.+?)"""', content, re.DOTALL)
        if ds_match:
            return ds_match.group(1).strip()
    except:
        pass
    return ""

def get_result(path):
    result_path = path.replace('.py', '_results.json')
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f)
        except:
            return None
    return None

def format_full_result(data, indent=0):
    """Recursively format result dict as markdown."""
    if not data:
        return "_Нет данных_"
    lines = []
    prefix = "  " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}- **{k}**:")
            lines.append(format_full_result(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{prefix}- **{k}**: {len(v)} items")
            for item in v[:5]:
                if isinstance(item, dict):
                    lines.append(f"{prefix}  - {item}")
                else:
                    lines.append(f"{prefix}  - {item}")
            if len(v) > 5:
                lines.append(f"{prefix}  - ... и ещё {len(v)-5}")
        else:
            if k == "pass":
                status = "✅ PASS" if v else "❌ FAIL"
                lines.append(f"{prefix}- **{k}**: {status}")
            else:
                lines.append(f"{prefix}- **{k}**: `{v}`")
    return "\n".join(lines)

def generate_detailed_book(mod_num, title, result_data, py_path):
    safe_title = re.sub(r'[^\w\-]', '_', title)
    filename = f"{BOOK_DIR}/{mod_num}_{safe_title}.md"
    
    docstring = get_docstring(py_path)
    
    # Determine status and analysis
    passed = result_data and result_data.get('pass')
    status = "✅ Успешно завершён" if passed else ("⚠️ Предупреждения" if result_data else "❌ Без результата")
    
    content = f"""# {mod_num} — {title}

## Описание эксперимента
{docstring or 'Детальное описание отсутствует в исходном файле.'}

## Исходный файл
`{os.path.basename(py_path)}`

## Результаты (полные данные)

{format_full_result(result_data)}

## Анализ

- **Модуль**: {mod_num}
- **Название**: {title}
- **Дата выполнения**: 2026-04-20 – 2026-05-06
- **Статус**: {status}

## Связанные модули

{'- ' + mod_num.replace('M', 'm') + '_*.py' if mod_num else ''}

---

*Запись сгенерирована автоматически из result JSON.*
"""
    with open(filename, "w") as f:
        f.write(content)
    return filename

def main():
    os.makedirs(BOOK_DIR, exist_ok=True)
    count = 0
    for py_path in sorted(glob.glob(f"{RESULT_DIR}/m*.py")):
        mod_num, title = parse_module_info(py_path)
        if not mod_num:
            continue
        num = int(mod_num[1:])
        if not (386 <= num <= 620):
            continue
        
        result_data = get_result(py_path)
        filename = generate_detailed_book(mod_num, title, result_data, py_path)
        count += 1
    
    print(f"Generated {count} detailed book entries")

if __name__ == "__main__":
    main()
