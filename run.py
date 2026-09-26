#!/usr/bin/env python3
"""
run.py
======

Entry point for the Business Entity Resolution project.

    python run.py

This does nothing but invoke the pipeline defined in src/main.py -- all
actual logic (loading, normalization, blocking, features, model
training, evaluation, and writing predictions.tsv) lives there and in
the other src/ modules. See README.md for setup and data-layout
instructions.
"""

from src.main import run_pipeline

if __name__ == "__main__":
    run_pipeline()