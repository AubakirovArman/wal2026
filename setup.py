#!/usr/bin/env python3
"""WAL Studio — WeightOps research framework.

Setup script for pip install.
"""
from setuptools import setup, find_packages


packages = find_packages(where="src") + ["framework"]

setup(
    name="wal-studio",
    version="0.1.0",
    description="Pre-alpha WeightOps research framework and WAL core runtime",
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
        "hf": ["transformers>=4.57", "accelerate"],
        "webgpu": ["wgpu"],
    },
    entry_points={
        "console_scripts": [
            "wal=wal.cli:main",
            "wal-core=wal.cli:core_main",
            "wal-studio=wal.cli:studio_main",
            "wal-encode=wal.cli:encode_main",
            "wal-decode=wal.cli:decode_main",
        ],
    },
)
