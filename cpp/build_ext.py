"""Build the C++ extension module for occulus.

This script builds the pybind11 C++ extension and installs it into
the correct location within the occulus package. The C++ backend is
optional — the pure-Python fallback is used when not compiled.

Usage
-----
    python cpp/build_ext.py          # build and install
    python cpp/build_ext.py --check  # verify the extension loads

Requirements
------------
    - CMake >= 3.15
    - C++17 compiler (GCC 8+, Clang 7+, MSVC 2019+)
    - pybind11 (pip install pybind11)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def build(source_dir: Path | None = None) -> Path:
    """Build the C++ extension and return the .so/.pyd path.

    Parameters
    ----------
    source_dir : Path, optional
        Path to the cpp/ directory.  Defaults to the directory
        containing this script.

    Returns
    -------
    Path
        Path to the compiled extension module.
    """
    if source_dir is None:
        source_dir = Path(__file__).resolve().parent

    build_dir = source_dir / "build"
    build_dir.mkdir(exist_ok=True)

    # Get pybind11 cmake dir
    try:
        import pybind11

        pybind11_dir = pybind11.get_cmake_dir()
    except ImportError:
        print("ERROR: pybind11 not installed. Run: pip install pybind11")
        sys.exit(1)

    # Configure
    print("Configuring C++ build...")
    subprocess.check_call(
        [
            "cmake",
            str(source_dir),
            f"-Dpybind11_DIR={pybind11_dir}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-Wno-dev",
        ],
        cwd=str(build_dir),
    )

    # Build
    ncpu = os.cpu_count() or 1
    print(f"Building with {ncpu} threads...")
    subprocess.check_call(
        ["cmake", "--build", ".", f"-j{ncpu}"],
        cwd=str(build_dir),
    )

    # Find the built extension
    for ext in ("*.so", "*.pyd", "*.dylib"):
        matches = list(build_dir.glob(f"_core{ext.replace('*', '')}*"))
        if not matches:
            matches = list(build_dir.glob(ext))
        if matches:
            return matches[0]

    print("ERROR: Extension not found after build")
    sys.exit(1)


def install(ext_path: Path) -> Path:
    """Copy the extension into src/occulus/_cpp/."""
    target_dir = Path(__file__).resolve().parent.parent / "src" / "occulus" / "_cpp"
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / ext_path.name
    shutil.copy2(ext_path, dest)
    print(f"Installed: {dest}")
    return dest


def check() -> bool:
    """Verify the C++ extension loads."""
    try:
        from occulus._cpp import _core  # noqa: F401

        print("C++ extension loaded successfully")
        print(f"  Submodules: {[x for x in dir(_core) if not x.startswith('_')]}")
        return True
    except ImportError as exc:
        print(f"C++ extension not available: {exc}")
        print("  The pure-Python fallback will be used instead.")
        return False


def main() -> int:
    """Build, install, and verify the C++ extension."""
    parser = argparse.ArgumentParser(description="Build occulus C++ extension")
    parser.add_argument("--check", action="store_true", help="Only check if extension loads")
    args = parser.parse_args()

    if args.check:
        return 0 if check() else 1

    ext_path = build()
    install(ext_path)
    ok = check()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
