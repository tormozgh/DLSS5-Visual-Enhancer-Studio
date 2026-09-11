"""DLSS 5 Visual Enhancer — Standalone Desktop GUI Entry Point."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from pathlib import Path

# Ensure Windows System32 modern d3dcompiler_47.dll is preloaded before PyQt6
if sys.platform == "win32":
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    sys_d3dcompiler = os.path.join(system32, "d3dcompiler_47.dll")
    if os.path.isfile(sys_d3dcompiler):
        with contextlib.suppress(Exception):
            ctypes.WinDLL(sys_d3dcompiler)

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.paths import LOGS, OUTPUTS
from src.gui.app import run_gui


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)

    sys.exit(run_gui())


if __name__ == "__main__":
    main()
