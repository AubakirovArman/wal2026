import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from macro_q8_rescue import passes_cumulative_and_incremental_gates  # noqa: E402


def test_three_way_gate_rejects_source_incremental_failure():
    assert not passes_cumulative_and_incremental_gates(
        {"c4": 1.01, "code": 1.019},
        {"c4": 1.001, "code": 1.0051},
        {"c4": 1.001, "code": 1.002},
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )


def test_three_way_gate_requires_every_policy_to_pass():
    passing = {"c4": 1.001, "code": 1.002}
    assert passes_cumulative_and_incremental_gates(
        {"c4": 0.99, "code": 1.019},
        passing,
        passing,
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )
    assert not passes_cumulative_and_incremental_gates(
        {"c4": 0.99, "code": 1.0201},
        passing,
        passing,
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )
    assert not passes_cumulative_and_incremental_gates(
        {"c4": 0.99, "code": 1.019},
        passing,
        {"c4": 1.001, "code": 1.0051},
        gate_ratio=1.02,
        incremental_gate_ratio=1.005,
    )
