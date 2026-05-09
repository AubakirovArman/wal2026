#!/usr/bin/env python3
"""WAL Framework — Unified CLI for all 11 phases.

Usage:
    python -m wal encode model.pt --output wal_model/
    python -m wal decode wal_model/ --output dense_model.pt
    python -m wal export wal_model/ --format onnx --output model.onnx
    python -m wal merge model_a/ model_b/ --method soup --output merged/
    python -m wal pipeline model.pt --output shipped/
"""
