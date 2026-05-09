#!/usr/bin/env python3
"""
Generate detailed dev diary appendix for M386-M620.
Reads all result JSON and produces comprehensive diary entries.
"""
import json, glob, os, re

RESULT_DIR = "experiments"
DIARY = "docs/dev_diary_ru.md"

def get_result(path):
    result_path = path.replace('.py', '_results.json')
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                return json.load(f)
        except:
            return None
    return None

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

def format_result_summary(data):
    if not data:
        return "—"
    parts = []
    for k in ['pass', 'score', 'total', 'passed', 'accuracy', 'survival', 'modules', 'experiments', 'results', 'grade', 'status', 'healthy', 'significant', 'balanced', 'cycle_free', 'robust', 'fair', 'deterministic', 'unique', 'efficient', 'reduction_pct', 'saved_mb', 'blocked', 'delivered', 'fired', 'agreement', 'drained', 'restored', 'injected', 'inference_test', 'model_loaded']:
        if k in data:
            v = data[k]
            if k == 'pass':
                parts.append(f"{'✅' if v else '❌'} pass")
            elif isinstance(v, float):
                parts.append(f"{k}={v:.3f}")
            else:
                parts.append(f"{k}={v}")
    return ', '.join(parts[:8]) or str(data)[:80]

def main():
    # Collect all M386-M620 experiments
    entries = []
    for py_path in sorted(glob.glob(f"{RESULT_DIR}/m*.py")):
        mod_num, title = parse_module_info(py_path)
        if not mod_num:
            continue
        num = int(mod_num[1:])
        if not (386 <= num <= 620):
            continue
        result = get_result(py_path)
        desc = get_docstring(py_path)
        entries.append((num, mod_num, title, desc, result, os.path.basename(py_path)))
    
    # Group by batches of 10
    batches = {}
    for num, mod_num, title, desc, result, fname in entries:
        batch_start = (num // 10) * 10
        batch_key = f"M{batch_start}-{batch_start+9}"
        batches.setdefault(batch_key, []).append((num, mod_num, title, desc, result, fname))
    
    lines = []
    lines.append("\n\n---\n")
    lines.append("# ПОЛНЫЙ ДЕТАЛЬНЫЙ ОТЧЁТ M386–M620\n")
    lines.append(f"**Дата генерации**: 2026-05-06\n")
    lines.append(f"**Всего модулей**: {len(entries)}\n")
    lines.append(f"**Всего батчей**: {len(batches)}\n")
    lines.append("---\n\n")
    
    passed_total = 0
    failed_total = 0
    
    for batch_key in sorted(batches.keys()):
        batch_entries = sorted(batches[batch_key], key=lambda x: x[0])
        lines.append(f"## {batch_key}\n\n")
        
        for num, mod_num, title, desc, result, fname in batch_entries:
            status_icon = "✅" if (result and result.get('pass')) else "⚠️"
            if result and result.get('pass') is False:
                status_icon = "❌"
                failed_total += 1
            elif result and result.get('pass'):
                passed_total += 1
            
            lines.append(f"### {mod_num} — {title}\n")
            lines.append(f"- **Файл**: `{fname}`\n")
            if desc:
                lines.append(f"- **Описание**: {desc}\n")
            lines.append(f"- **Результат**: {format_result_summary(result)}\n")
            lines.append(f"- **Статус**: {status_icon}\n\n")
        
        lines.append("---\n\n")
    
    lines.append("## ИТОГОВАЯ СТАТИСТИКА\n\n")
    lines.append(f"- **Всего модулей**: {len(entries)}\n")
    lines.append(f"- **Успешно (PASS)**: {passed_total}\n")
    lines.append(f"- **Провалено (FAIL)**: {failed_total}\n")
    lines.append(f"- **Без явного статуса**: {len(entries) - passed_total - failed_total}\n")
    lines.append(f"- **Процент успеха**: {passed_total/max(len(entries),1)*100:.1f}%\n")
    lines.append("\n---\n")
    lines.append("*Отчёт сгенерирован автоматически из result JSON файлов.*\n")
    
    with open(DIARY, "a") as f:
        f.write("".join(lines))
    
    print(f"Appended detailed report for {len(entries)} modules to {DIARY}")
    print(f"PASS: {passed_total}, FAIL: {failed_total}")

if __name__ == "__main__":
    main()
