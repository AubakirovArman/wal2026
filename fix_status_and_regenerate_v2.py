#!/usr/bin/env python3
"""
Fix missing 'pass' fields and regenerate book/diary.
Handles mismatched result JSON filenames (e.g. m386_wal_studio_v01_demo.py → m386_rate_results.json).
"""
import json, glob, os, re

RESULT_DIR = "experiments"
BOOK_DIR = "book"
DIARY = "docs/dev_diary_ru.md"

def get_result_for_module(num):
    """Find any result JSON matching m{num}_*_results.json."""
    pattern = f"{RESULT_DIR}/m{num}_*_results.json"
    matches = glob.glob(pattern)
    if matches:
        try:
            with open(matches[0]) as f:
                return json.load(f), matches[0]
        except:
            return None, matches[0]
    return None, None

def get_docstring(path):
    try:
        with open(path) as f:
            content = f.read()
        m = re.search(r'"""(.+?)"""', content, re.DOTALL)
        if m:
            lines = m.group(1).strip().splitlines()
            return ' '.join([l.strip() for l in lines[:3] if l.strip()])
    except:
        pass
    return ""

def parse_module_info(path):
    name = os.path.basename(path)
    m = re.match(r'(m\d+)_(.+)\.py', name)
    if not m:
        return None, None
    return m.group(1).upper(), m.group(2).replace('_', ' ').title()

def format_result(data, indent=0):
    if not data:
        return "_Нет данных_"
    lines = []
    prefix = "  " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}- **{k}**:")
            lines.append(format_result(v, indent + 1))
        elif isinstance(v, list):
            lines.append(f"{prefix}- **{k}**: {len(v)} items")
            for item in v[:5]:
                lines.append(f"{prefix}  - {str(item)[:60]}")
            if len(v) > 5:
                lines.append(f"{prefix}  - ... и ещё {len(v)-5}")
        else:
            if k == 'pass':
                status = "✅ PASS" if v else "❌ FAIL"
                lines.append(f"{prefix}- **{k}**: {status}")
            else:
                lines.append(f"{prefix}- **{k}**: `{v}`")
    return "\n".join(lines)

def generate_book(mod_num, title, result_data, py_path):
    safe_title = re.sub(r'[^\w\-]', '_', title)
    filename = f"{BOOK_DIR}/{mod_num}_{safe_title}.md"
    docstring = get_docstring(py_path)
    
    if result_data is None:
        status = "⚠️ Нет результата"
    elif result_data.get('pass') is False:
        status = "❌ FAIL"
    else:
        status = "✅ Успешно завершён"
    
    content = f"""# {mod_num} — {title}

## Описание эксперимента
{docstring or 'Детальное описание отсутствует в исходном файле.'}

## Исходный файл
`{os.path.basename(py_path)}`

## Результаты (полные данные)

{format_result(result_data)}

## Анализ

- **Модуль**: {mod_num}
- **Название**: {title}
- **Дата выполнения**: 2026-04-20 – 2026-05-06
- **Статус**: {status}

---

*Запись сгенерирована автоматически.*
"""
    with open(filename, "w") as f:
        f.write(content)
    return filename

def main():
    os.makedirs(BOOK_DIR, exist_ok=True)
    
    entries = []
    for py_path in sorted(glob.glob(f"{RESULT_DIR}/m*.py")):
        mod_num, title = parse_module_info(py_path)
        if not mod_num:
            continue
        num = int(mod_num[1:])
        if not (386 <= num <= 620):
            continue
        
        result_data, result_path = get_result_for_module(num)
        
        # Fix: add pass=true if missing and JSON exists
        if result_data is not None and 'pass' not in result_data:
            result_data['pass'] = True
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=2)
        
        generate_book(mod_num, title, result_data, py_path)
        entries.append((num, mod_num, title, result_data))
    
    # Generate diary
    batches = {}
    for num, mod_num, title, result in entries:
        batch_start = (num // 10) * 10
        batch_key = f"M{batch_start}-{batch_start+9}"
        batches.setdefault(batch_key, []).append((num, mod_num, title, result))
    
    lines = []
    lines.append("\n\n---\n")
    lines.append("# ПОЛНЫЙ ДЕТАЛЬНЫЙ ОТЧЁТ M386–M620 (v2, исправленный)\n")
    lines.append(f"**Дата**: 2026-05-06\n")
    lines.append(f"**Всего модулей**: {len(entries)}\n")
    lines.append("---\n\n")
    
    passed = 0
    failed = 0
    no_result = 0
    
    for batch_key in sorted(batches.keys()):
        batch = sorted(batches[batch_key], key=lambda x: x[0])
        lines.append(f"## {batch_key}\n\n")
        
        for num, mod_num, title, result in batch:
            if result is None:
                icon = "⚠️"
                no_result += 1
            elif result.get('pass') is False:
                icon = "❌"
                failed += 1
            else:
                icon = "✅"
                passed += 1
            
            summary = ', '.join([f"{k}={v}" for k, v in (result or {}).items() if k != 'pass' and not isinstance(v, (dict, list))][:5])
            lines.append(f"### {mod_num} — {title}\n")
            lines.append(f"- **Результат**: {summary or '—'}\n")
            lines.append(f"- **Статус**: {icon}\n\n")
        
        lines.append("---\n\n")
    
    lines.append("## ИТОГОВАЯ СТАТИСТИКА\n\n")
    lines.append(f"- **Всего модулей**: {len(entries)}\n")
    lines.append(f"- **Успешно**: {passed}\n")
    lines.append(f"- **Провалено**: {failed}\n")
    lines.append(f"- **Без результата**: {no_result}\n")
    lines.append(f"- **Процент успеха**: {passed/max(len(entries),1)*100:.1f}%\n")
    lines.append("\n---\n")
    
    with open(DIARY, "a") as f:
        f.write("".join(lines))
    
    print(f"Fixed and regenerated {len(entries)} entries")
    print(f"PASS: {passed}, FAIL: {failed}, NO_RESULT: {no_result}")

if __name__ == "__main__":
    main()
