from pathlib import Path

from wal.legacy_audit import experiment_number, experiment_order_key, runner_type


def test_experiment_number_handles_suffixes():
    assert experiment_number("m1b_probe_rownorm.py") == 1
    assert experiment_number("m34_m35_m36_encoder_redesign.py") == 34
    assert experiment_number("__init__.py") is None


def test_order_key_sorts_numeric_prefix_before_suffix():
    paths = [
        Path("m10a_block.py"),
        Path("m1c_calibration.py"),
        Path("m1_probe.py"),
        Path("m1b_probe.py"),
    ]
    ordered = [path.name for path in sorted(paths, key=experiment_order_key)]
    assert ordered == ["m1_probe.py", "m1b_probe.py", "m1c_calibration.py", "m10a_block.py"]


def test_runner_type_prefers_controlled_model_for_heavy_reasons():
    selected = runner_type(
        blocked_reasons=["local_model_path", "model_artifact"],
        static_flags={"writes_results": True},
        runnable=False,
    )
    assert selected == "gpu_or_model_controlled"
