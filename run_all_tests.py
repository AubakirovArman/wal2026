#!/usr/bin/env python3
"""Run all WAL test suites."""
import subprocess
import sys

TESTS = [
    ("M76", "experiments/m76_wal_v1_roundtrip.py", 5, 120),
    ("M77", "experiments/m77_pytorch_integration.py", 5, 120),
    ("M78", "experiments/m78_wal_v1_debugger.py", 7, 120),
    ("M79", "experiments/m79_stdlib_prototype.py", 6, 120),
    ("M80", "experiments/m80_hardware_backends.py", 8, 120),
    ("M81", "experiments/m81_meta_learning.py", 5, 120),
    ("M82", "experiments/m82_adapter_integration.py", 4, 120),
    ("M83", "experiments/m83_ecosystem.py", 7, 120),
    ("M89", "experiments/m89_streaming_encoder.py", 1, 600),
    ("M90", "experiments/m90_streaming_encoder_test.py", 4, 600),
    ("M91", "experiments/m91_qat_differentiable_decode.py", 5, 120),
    ("M92", "experiments/m92_wal_native_lora.py", 3, 120),
    ("M93", "experiments/m93_qat_ppl_prototype.py", 3, 600),
    ("M94", "experiments/m94_qat_reencode.py", 1, 120),
    ("M95", "experiments/m95_qat_full_pipeline.py", 1, 120),
]

total_pass = 0
total_fail = 0

print("=" * 60)
print("WAL Test Suite — Phases 6-13")
print("=" * 60)

for name, path, expected, timeout in TESTS:
    print(f"\n[{name}] Running {path}...")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    
    # Count passes
    output = result.stdout + result.stderr
    pass_count = output.count("✅ PASS") + output.count("✓") + output.count("ALL TESTS PASSED")
    
    if result.returncode == 0:
        print(f"  ✅ PASS ({expected}/{expected})")
        total_pass += expected
    else:
        print(f"  ❌ FAIL — exit code {result.returncode}")
        print(f"  Last lines: {' | '.join(output.splitlines()[-3:])}")
        total_fail += expected

print("\n" + "=" * 60)
print(f"Total: {total_pass}/{total_pass + total_fail} tests passed")
print("=" * 60)

if total_fail > 0:
    sys.exit(1)
