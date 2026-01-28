#!/usr/bin/env python3
"""Verify that the ipomdp_shielding environment is set up correctly."""

import sys
from importlib import import_module

def check_imports():
    """Check that all required modules can be imported."""
    required_modules = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('matplotlib', 'Matplotlib'),
        ('statsmodels', 'statsmodels'),
    ]

    project_modules = [
        ('ipomdp_shielding.Models', 'Models'),
        ('ipomdp_shielding.Propagators', 'Propagators'),
        ('ipomdp_shielding.MonteCarlo', 'MonteCarlo'),
        ('ipomdp_shielding.Evaluation', 'Evaluation'),
    ]

    print("Checking dependencies...")
    print("-" * 50)

    all_ok = True
    for module_name, display_name in required_modules:
        try:
            mod = import_module(module_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"✓ {display_name:20s} (version {version})")
        except ImportError as e:
            print(f"✗ {display_name:20s} MISSING")
            print(f"  Error: {e}")
            all_ok = False

    print()
    print("Checking project modules...")
    print("-" * 50)

    for module_name, display_name in project_modules:
        try:
            import_module(module_name)
            print(f"✓ {display_name}")
        except ImportError as e:
            print(f"✗ {display_name} FAILED")
            print(f"  Error: {e}")
            all_ok = False

    print()
    print("-" * 50)
    if all_ok:
        print("✓ All checks passed! Environment is ready.")
        return 0
    else:
        print("✗ Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(check_imports())
