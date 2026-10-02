"""Verify the research environment.

Imports every project dependency and prints its version so the
environment can be reproduced exactly. Exits non-zero if any
required package is missing.
"""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata

# Distribution name -> import name (they differ for a few packages).
REQUIRED: dict[str, str] = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "xgboost": "xgboost",
    "fairlearn": "fairlearn",
    "shap": "shap",
    "codecarbon": "codecarbon",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "jupyter": "jupyter",
}


def _version(module: object, package: str) -> str:
    """Return the package version, falling back to installed metadata."""
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_env() -> int:
    """Import every dependency and print versions.

    Returns:
        Number of missing packages (0 means the environment is ready).
    """
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print("-" * 60)

    missing: list[str] = []
    for package, module in REQUIRED.items():
        try:
            mod = importlib.import_module(module)
            print(f"{package:15s} {_version(mod, package)}")
        except ImportError:
            missing.append(package)
            print(f"{package:15s} MISSING")

    print("-" * 60)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
    else:
        print("All required packages are installed.")
    return len(missing)


if __name__ == "__main__":
    sys.exit(1 if check_env() else 0)
