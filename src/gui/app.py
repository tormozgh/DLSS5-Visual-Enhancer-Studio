"""Application bootstrap and QApplication configuration."""

from __future__ import annotations

import os
import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow
from .styles import DARK_QSS


def run_gui() -> int:
    """Run the standalone PyQt6 desktop application."""
    # Ensure Windows DPI scaling is crisp
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("DLSS 5 Visual Enhancer")
    app.setOrganizationName("Merserk")

    # Apply Dark Studio QSS
    app.setStyleSheet(DARK_QSS)

    window = MainWindow()
    window.show()

    return app.exec()
