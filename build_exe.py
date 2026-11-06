"""One-click PyInstaller build for the DCC Studio GUI.

This is the Python replacement for the former ``build_exe.ps1`` and
``build_exe.cmd``. A plain Python script is not affected by the Windows
PowerShell execution policy and needs no batch wrapper, so the repository has
exactly one build entry point and one test entry point (``run_tests.py``).

Usage::

    python build_exe.py                     # single-file dist/DCCStudio.exe
    python build_exe.py --verify            # also build/run the frozen self-test
    python build_exe.py --one-dir           # folder output (larger, faster start)
    python build_exe.py --clean             # clear the PyInstaller cache first
    python build_exe.py --run               # launch the GUI after a good build
    python build_exe.py --clean --verify --run

The old PowerShell-style switches (``-Verify``, ``-Clean``, ``-OneDir``,
``-Run``) are accepted as aliases, so existing command lines keep working
after the file extension changes.

There is no CLI variant of the product: ``dist/DCCStudio.exe`` is the only
shipped executable. ``--verify`` builds a temporary ``dcc_verify`` executable
from the very same spec, runs it in the frozen environment and deletes it, so
``dist/`` always contains only the product.

Build dependencies are deliberately separate from runtime dependencies:
PyInstaller is installed only on the machine that produces the distributable.
The exit code is 0 on success and non-zero on any failure, so the script works
in CI and Task Scheduler. ``dist/`` and ``build/`` are gitignored.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"
SPEC = ROOT / "build_exe.spec"
REQUIREMENTS = ROOT / "requirements-build.txt"
PRODUCT = "DCCStudio"
VERIFY = "dcc_verify"


def _python_executable() -> str:
    """Choose the interpreter that owns the project's build dependencies.

    Prefer the project virtual environment, matching the behaviour of the old
    PowerShell script; fall back to the interpreter running this script. The
    POSIX path is included so the helper also behaves sensibly off Windows.
    """
    for relative in (".venv/Scripts/python.exe", ".venv/bin/python"):
        candidate = ROOT / relative
        if candidate.is_file():
            return str(candidate)
    return sys.executable


def _run(command: list[str]) -> int:
    """Run a child process from the project root and return its exit code.

    PyInstaller writes its INFO logs to stderr. Letting them stream straight
    to the terminal avoids the PowerShell 5.1 problem where redirected stderr
    was treated as a terminating error.
    """
    shown = " ".join(f'"{part}"' if " " in part else part for part in command)
    print("+ " + shown, flush=True)
    return subprocess.run(command, cwd=str(ROOT)).returncode


def _install_build_requirements(python: str) -> bool:
    """Install PyInstaller and its build-time helpers exactly once per run."""
    print("Installing build requirements...", flush=True)
    command = [python, "-m", "pip", "install", "--disable-pip-version-check",
               "-r", str(REQUIREMENTS)]
    return _run(command) == 0


def _pyinstaller(python: str, build_name: str, clean: bool) -> bool:
    """Build one named target against the shared spec in its own work folder."""
    command = [python, "-m", "PyInstaller", str(SPEC), "--noconfirm",
               "--distpath", str(DIST), "--workpath", str(BUILD / build_name)]
    if clean:
        command.append("--clean")
    return _run(command) == 0


def _output_path(name: str, one_file: bool) -> Path:
    """Where PyInstaller writes a target in onefile / onedir mode."""
    return DIST / (f"{name}.exe" if one_file else name)


def _size_bytes(path: Path) -> int:
    """Size of a single file, or the sum of all files inside a folder."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*")
               if item.is_file())


def _remove(path: Path) -> None:
    """Delete a file or directory, tolerating an already-absent target."""
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def _verify(python: str, one_file: bool, clean: bool) -> bool:
    """Build, execute and remove the frozen self-test executable.

    ``build_exe.spec`` reads ``DCC_SELFTEST`` at import time, so the flag is
    set for the child process and always reset in ``finally``. Cleanup also
    runs on failure: ``dist/`` must never retain the verification artifact.
    """
    os.environ["DCC_SELFTEST"] = "1"
    try:
        if not _pyinstaller(python, VERIFY, clean):
            return False
        output = _output_path(VERIFY, one_file)
        verify_exe = output if one_file else output / f"{VERIFY}.exe"
        if not verify_exe.is_file():
            print(f"ERROR: self-test executable not found: {verify_exe}",
                  file=sys.stderr)
            return False
        code = _run([str(verify_exe)])
        if code != 0:
            print(f"ERROR: frozen self-test failed (exit {code}).",
                  file=sys.stderr)
            return False
        print("Self-test passed.")
        return True
    finally:
        _remove(_output_path(VERIFY, one_file))
        _remove(BUILD / VERIFY)
        os.environ["DCC_SELFTEST"] = "0"


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, build the product and return a CI-friendly exit code."""
    parser = argparse.ArgumentParser(
        description="Build the DCC Studio GUI executable with PyInstaller.")
    parser.add_argument("--one-dir", "-OneDir", action="store_true",
                        help="produce an onedir folder instead of one exe")
    parser.add_argument("--clean", "-Clean", action="store_true",
                        help="clear the PyInstaller cache before building")
    parser.add_argument("--run", "-Run", action="store_true",
                        help="launch the GUI after a successful build")
    parser.add_argument("--verify", "-Verify", action="store_true",
                        help="build and run the frozen self-test, then delete it")
    args = parser.parse_args(argv)

    python = _python_executable()
    one_file = not args.one_dir
    print(f"Interpreter: {python}")
    if not _install_build_requirements(python):
        print("ERROR: failed to install build requirements.", file=sys.stderr)
        return 2

    # The spec reads both flags at import time, so they must be present in the
    # environment before PyInstaller starts and be inherited by its children.
    os.environ["DCC_SELFTEST"] = "0"
    os.environ["DCC_ONEFILE"] = "1" if one_file else "0"

    if not _pyinstaller(python, PRODUCT, args.clean):
        print("ERROR: GUI build failed.", file=sys.stderr)
        return 1
    output = _output_path(PRODUCT, one_file)
    if not output.exists():
        print(f"ERROR: output not found: {output}", file=sys.stderr)
        return 1
    print(f"[gui] {output}  {_size_bytes(output) / 1048576:.1f} MB")

    if args.run:
        exe = output if one_file else output / f"{PRODUCT}.exe"
        if exe.is_file():
            subprocess.Popen([str(exe)], cwd=str(exe.parent))
            print("GUI launched. If no window appears, rebuild with "
                  "DCC_CONSOLE=1 to see the traceback.")
        else:
            print(f"WARNING: cannot launch, missing {exe}", file=sys.stderr)

    if args.verify and not _verify(python, one_file, args.clean):
        return 1
    print(f"Build finished successfully. Output: {DIST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
