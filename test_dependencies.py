#!/usr/bin/env python3
"""Quick dependency and environment check for this project.

Usage:
  python test_dependencies.py [--skip-data] [--min-python 3.10] [--verbose]

What it checks:
  - Python version (default >= 3.10)
  - Required packages importability and versions
  - PyTorch capabilities (CPU/CUDA/MPS availability)
  - Data files and output directories from config.py (unless skipped)

Exit codes:
  0 = All checks passed
  1 = One or more required checks failed
"""

from __future__ import annotations

import argparse
import importlib
import sys
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    # Python 3.8+
    import importlib.metadata as importlib_metadata
except Exception:  # pragma: no cover
    import importlib_metadata  # type: ignore


MODULE_TO_DIST_NAME: Dict[str, str] = {
    "sklearn": "scikit-learn",
    "talib": "TA-Lib",
}

REQUIRED_MODULES: Tuple[str, ...] = (
    "numpy",
    "pandas",
    "sklearn",
    "joblib",
    "matplotlib",
    "seaborn",
    "talib",
    "torch",
    "transformers",
    "xgboost",
)

OPTIONAL_MODULES: Tuple[str, ...] = (
    "torchvision",
    "torchaudio",
)


def get_distribution_name(module_name: str) -> str:
    return MODULE_TO_DIST_NAME.get(module_name, module_name)


def get_module_version(module_name: str) -> Optional[str]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    # Fallback to distribution metadata
    try:
        return importlib_metadata.version(get_distribution_name(module_name))
    except Exception:
        return None


def check_python_version(min_version: Tuple[int, int]) -> Tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= min_version
    return ok, f"Python {major}.{minor} detected (require >= {min_version[0]}.{min_version[1]})"


def try_import(module_name: str) -> Tuple[bool, Optional[str]]:
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, f"{exc.__class__.__name__}: {exc}"


def check_required_modules(verbose: bool) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    all_ok = True
    for module_name in REQUIRED_MODULES:
        ok, err = try_import(module_name)
        if ok:
            if verbose:
                version = get_module_version(module_name)
                version_str = f" v{version}" if version else ""
                messages.append(f"[OK] {module_name}{version_str}")
        else:
            all_ok = False
            dist = get_distribution_name(module_name)
            messages.append(
                f"[MISS] {module_name} not importable. Install via pip: 'pip install {dist}' or conda: 'conda install -c conda-forge {dist}'\n"
            )
            if module_name == "talib" and platform.system() == "Darwin":
                messages.append("        macOS note: 'brew install ta-lib' before 'pip install TA-Lib'")
    return all_ok, messages


def check_optional_modules(verbose: bool) -> List[str]:
    messages: List[str] = []
    for module_name in OPTIONAL_MODULES:
        ok, _ = try_import(module_name)
        if ok and verbose:
            version = get_module_version(module_name)
            version_str = f" v{version}" if version else ""
            messages.append(f"[OK-OPT] {module_name}{version_str}")
        elif not ok and verbose:
            messages.append(f"[SKIP-OPT] {module_name} not found (optional)")
    return messages


def check_torch_capabilities(verbose: bool) -> List[str]:
    messages: List[str] = []
    ok, _ = try_import("torch")
    if not ok:
        return messages
    import torch  # type: ignore

    cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps_ok = False
    try:
        mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        mps_ok = False
    if verbose:
        messages.append(f"[INFO] torch v{get_module_version('torch')}")
        messages.append(f"[INFO] CUDA available: {cuda_ok}")
        messages.append(f"[INFO] MPS available: {mps_ok}")
    return messages


def check_config_and_data(verbose: bool) -> Tuple[bool, List[str]]:
    messages: List[str] = []
    try:
        import config  # type: ignore
    except Exception as exc:
        return False, [f"[MISS] Could not import config.py: {exc}"]

    ok = True

    def _exists(p: Path) -> bool:
        try:
            return p.exists()
        except Exception:
            return False

    data_files = [
        ("AAPL_FILE", getattr(config, "AAPL_FILE", None)),
        ("VGT_FILE", getattr(config, "VGT_FILE", None)),
        ("TWEETS_FILE", getattr(config, "TWEETS_FILE", None)),
    ]
    for name, path_obj in data_files:
        if isinstance(path_obj, Path) and _exists(path_obj):
            if verbose:
                messages.append(f"[OK] {name}: {path_obj}")
        else:
            ok = False
            messages.append(f"[MISS] {name} not found or not a Path: {path_obj}")

    output_dirs = [
        ("OUTPUT_DIR", getattr(config, "OUTPUT_DIR", None)),
        ("BENCHMARKS_OUTPUT_DIR", getattr(config, "BENCHMARKS_OUTPUT_DIR", None)),
    ]
    for name, dir_path in output_dirs:
        if not isinstance(dir_path, Path):
            ok = False
            messages.append(f"[MISS] {name} missing or not a Path in config.py")
            continue
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            test_file = dir_path / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
            if verbose:
                messages.append(f"[OK] {name} writable: {dir_path}")
        except Exception as exc:
            ok = False
            messages.append(f"[MISS] {name} not writable: {dir_path} ({exc})")

    return ok, messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Project dependency and environment checker")
    parser.add_argument("--skip-data", action="store_true", help="Skip config/data/output checks")
    parser.add_argument("--min-python", default="3.10", help="Minimum Python version, e.g., 3.10")
    parser.add_argument("--verbose", action="store_true", help="Verbose output (show versions and details)")
    args = parser.parse_args()

    try:
        min_py_parts = tuple(int(p) for p in str(args.min_python).split(".")[:2])
        if len(min_py_parts) != 2:
            raise ValueError
    except Exception:
        print("[ERR] --min-python must be like '3.10'", file=sys.stderr)
        return 1

    print(f"Platform: {platform.platform()}")
    py_ok, py_msg = check_python_version(min_py_parts) 
    print(("[OK] " if py_ok else "[MISS] ") + py_msg)

    req_ok, req_msgs = check_required_modules(verbose=args.verbose)
    for m in req_msgs:
        print(m)

    opt_msgs = check_optional_modules(verbose=args.verbose)
    for m in opt_msgs:
        print(m)

    torch_msgs = check_torch_capabilities(verbose=args.verbose)
    for m in torch_msgs:
        print(m)

    data_ok = True
    if not args.skip_data:
        data_ok, data_msgs = check_config_and_data(verbose=args.verbose)
        for m in data_msgs:
            print(m)

    all_ok = py_ok and req_ok and data_ok
    if all_ok:
        print("\nAll checks passed. You're good to go! ✅")
        return 0
    print("\nSome checks failed. See messages above. ❌")
    return 1


if __name__ == "__main__":
    sys.exit(main())


