#!/usr/bin/env python3
"""WAL — Weight Assembly Language.

Setup script for pip install.
"""
from setuptools import setup, find_packages


packages = find_packages(where="src") + ["framework"]

setup(
    name="wal",
    version="1.0.0",
    description="Weight Assembly Language — a language for neural network weights",
    author="WAL Team",
    packages=packages,
    package_dir={"": "src", "framework": "framework"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy",
    ],
    extras_require={
        "dev": ["pytest", "black", "mypy"],
        "webgpu": ["wgpu"],
    },
    entry_points={
        "console_scripts": [
            "wal=wal.cli:main",
            "wal-encode=wal.cli:encode_main",
            "wal-decode=wal.cli:decode_main",
        ],
    },
)
