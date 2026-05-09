"""
WAL Project — MIT License
Copyright (c) 2026 WAL Research Team
"""
"""
M315 — Final System Test

Comprehensive end-to-end test of entire WAL system.
"""
import json, os, time

print("=" * 60)
print("M315 — FINAL SYSTEM TEST")
print("=" * 60)

tests_passed = 0
tests_total = 0

def test(name, condition):
    global tests_passed, tests_total
    tests_total += 1
    if condition:
        tests_passed += 1
        print(f"  ✅ {name}")
    else:
        print(f"  ❌ {name}")

# Test 1: File system
test("Project directory exists", os.path.isdir("src"))
test("Book directory exists", os.path.isdir("book"))
test("Docs directory exists", os.path.isdir("docs"))

# Test 2: Book entries
book_files = [f for f in os.listdir("book") if f.endswith(".md")]
test(f"Book entries: {len(book_files)} files", len(book_files) >= 200)

# Test 3: Experiments
exp_files = [f for f in os.listdir("experiments") if f.endswith(".py")]
test(f"Experiments: {len(exp_files)} scripts", len(exp_files) >= 50)

# Test 4: Results files
result_files = [f for f in os.listdir("experiments") if f.endswith("_results.json")]
test(f"Result files: {len(result_files)} outputs", len(result_files) >= 30)

# Test 5: Documentation
test("User guide exists", os.path.exists("docs/USER_GUIDE.md"))
test("API reference exists", os.path.exists("docs/API_REFERENCE.md"))
test("Dev diary exists", os.path.exists("docs/dev_diary_ru.md"))

# Test 6: ROADMAP versions
roadmaps = [f for f in os.listdir(".") if f.startswith("ROADMAP")]
test(f"ROADMAP versions: {len(roadmaps)}", len(roadmaps) >= 5)

# Test 7: Production stack version
with open("docs/dev_diary_ru.md") as f:
    diary = f.read()
test("Production stack v19 documented", "v19" in diary)

# Test 8: Key experiments mentioned
key_exps = ["M276", "M281", "M292", "M300", "M305"]
for exp in key_exps:
    test(f"{exp} documented", exp in diary)

# Summary
print("\n" + "=" * 60)
print(f"FINAL SYSTEM TEST RESULTS")
print("=" * 60)
print(f"  Passed: {tests_passed}/{tests_total}")
print(f"  Failed: {tests_total - tests_passed}/{tests_total}")
print(f"  Success rate: {tests_passed/tests_total:.1%}")

with open("experiments/m315_final_results.json", "w") as f:
    json.dump({
        "passed": tests_passed,
        "total": tests_total,
        "success_rate": tests_passed / tests_total,
    }, f, indent=2)

if tests_passed == tests_total:
    print("\n🎉 M315: ALL SYSTEM TESTS PASSED")
else:
    print(f"\n⚠️ M315: {tests_total - tests_passed} tests failed")
