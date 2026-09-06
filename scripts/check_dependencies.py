#!/usr/bin/env python3
"""Validate required dependencies are installed with compatible versions."""

from __future__ import annotations

import importlib
import sys

REQUIRED = {
    "numpy": "1.26",
    "pandas": "2.0",
    "sklearn": "1.4",
    "scipy": "1.11",
    "pyarrow": "14.0",
    "requests": "2.28",
    "pydantic": "2.5",
}


def main() -> int:
    missing = []
    mismatch = []
    installed = []
    for mod, min_ver in REQUIRED.items():
        name = "scikit-learn" if mod == "sklearn" else mod
        try:
            m = importlib.import_module(mod if mod != "sklearn" else "sklearn")
            ver = getattr(m, "__version__", "unknown")
            installed.append(f"{name}=={ver}")
            # crude major.minor check
            try:
                parts = [int(x) for x in ver.split(".")[:2]]
                want = [int(x) for x in min_ver.split(".")[:2]]
                if parts < want:
                    mismatch.append(f"{name} {ver} < {min_ver}")
            except ValueError:
                pass
        except ImportError:
            missing.append(name)

    print("INSTALLED:")
    for line in installed:
        print(f"  {line}")
    if mismatch:
        print("VERSION_MISMATCH:")
        for line in mismatch:
            print(f"  {line}")
    if missing:
        print("MISSING:")
        for line in missing:
            print(f"  {line}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
