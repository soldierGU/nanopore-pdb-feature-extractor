#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
00_check_environment.py

Purpose:
    Check whether the minimal Python environment is ready.

Usage:
    python scripts/00_check_environment.py
"""

from __future__ import annotations

import importlib
import platform
import sys


REQUIRED_PACKAGES = [
    "Bio",
    "numpy",
    "pandas",
    "yaml",
    "tqdm",
]


def check_package(package_name: str) -> bool:
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {package_name}: {version}")
        return True
    except ImportError:
        print(f"[MISSING] {package_name}")
        return False


def main() -> None:
    print("=" * 80)
    print("Nanopore PDB Feature Extractor - Environment Check")
    print("=" * 80)

    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version}")
    print(f"Platform         : {platform.platform()}")
    print("-" * 80)

    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        ok = check_package(pkg)
        all_ok = all_ok and ok

    print("-" * 80)
    if all_ok:
        print("[PASS] Environment is ready.")
    else:
        print("[FAIL] Some packages are missing. Please install requirements.txt.")
        sys.exit(1)


if __name__ == "__main__":
    main()