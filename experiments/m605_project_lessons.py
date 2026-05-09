"""
M605 — Project Lessons Learned

Key lessons from 600 modules.
"""
import json

lessons = [
    "Start with CI from day one",
    "Test on real data early",
    "Document as you go",
    "Fix memory leaks before they compound",
    "Security is not optional",
]

with open("LESSONS.md", "w") as f:
    f.write("# Lessons Learned\n\n")
    for i, lesson in enumerate(lessons, 1):
        f.write(f"{i}. {lesson}\n")

print("=" * 60)
print("M605 — LESSONS LEARNED")
print("=" * 60)
for i, lesson in enumerate(lessons, 1):
    print(f"  {i}. {lesson}")

with open("experiments/m605_lessons_results.json", "w") as f:
    json.dump({"lessons": len(lessons), "pass": True}, f, indent=2)

print("\n✅ M605: Lessons documented")
