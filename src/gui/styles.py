"""Neutral Dark Gray Studio Theme for PyQt6 (Blender / DaVinci Resolve inspired)."""

from __future__ import annotations

# Neutral Dark Gray Color Palette (No greens, no blues)
BG_DARKEST = "#101114"
BG_MAIN = "#16171b"
BG_CARD = "#1f2126"
BG_CARD_ALT = "#272a31"
BG_INPUT = "#18191d"
BG_HOVER = "#32353e"

BORDER_COLOR = "#323640"
BORDER_FOCUS = "#7a808e"

TEXT_PRIMARY = "#f0f1f4"
TEXT_SECONDARY = "#9ca0ab"
TEXT_MUTED = "#606470"

# Neutral Silver / Gray Accents
ACCENT_PRIMARY = "#d0d4dc"
ACCENT_HOVER = "#e3e6ec"
ACCENT_ACTIVE = "#b0b5c0"
ACCENT_TRACK = "#484c57"

DARK_QSS = f"""
/* Global Reset */
* {{
    font-family: "Segoe UI", "Segoe UI Variable", -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    color: {TEXT_PRIMARY};
    outline: none;
}}

QLabel {{
    border: none;
    background: transparent;
}}

QMainWindow, QWidget#centralWidget {{
    background-color: {BG_MAIN};
}}

/* Top Navigation Bar */
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 10px 22px;
    font-weight: 600;
    font-size: 13px;
    border: none;
    border-bottom: 3px solid transparent;
    margin-right: 4px;
}}

QTabBar::tab:hover {{
    color: {TEXT_PRIMARY};
    background-color: {BG_CARD};
    border-radius: 4px 4px 0 0;
}}

QTabBar::tab:selected {{
    color: #ffffff;
    border-bottom: 3px solid {ACCENT_PRIMARY};
    font-weight: 700;
}}

QTabWidget::pane {{
    border: 1px solid #22242a;
    border-top: none;
    background-color: {BG_MAIN};
}}

/* Minimalist Cards and Panels */
QFrame.card, QGroupBox, QFrame[class="studio-card"] {{
    background-color: #17181c;
    border: 1px solid #262830;
    border-radius: 6px;
    margin-top: 10px;
    padding: 12px 14px;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 6px;
    color: {ACCENT_PRIMARY};
    font-weight: 700;
    font-size: 12px;
    letter-spacing: 0.5px;
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 6px;
    background: {BG_INPUT};
    border: 1px solid {BORDER_COLOR};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {ACCENT_TRACK};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {ACCENT_PRIMARY};
    border: 1px solid {BORDER_FOCUS};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: #ffffff;
    border-color: #ffffff;
}}

QSlider::handle:horizontal:pressed {{
    background: {ACCENT_ACTIVE};
}}

/* Buttons */
QPushButton {{
    background-color: {BG_CARD_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 7px 16px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER_FOCUS};
    color: #ffffff;
}}

QPushButton:pressed {{
    background-color: {BG_INPUT};
}}

QPushButton:disabled {{
    background-color: {BG_INPUT};
    color: {TEXT_MUTED};
    border-color: transparent;
}}

QPushButton.primary {{
    background-color: {ACCENT_PRIMARY};
    color: #101114;
    border: 1px solid {ACCENT_HOVER};
    font-weight: 700;
}}

QPushButton.primary:hover {{
    background-color: {ACCENT_HOVER};
    color: #000000;
}}

QPushButton.primary:pressed {{
    background-color: {ACCENT_ACTIVE};
}}

QPushButton.secondary {{
    background-color: {BG_CARD_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_COLOR};
}}

/* Mini Buttons (e.g. Reset, View mode) */
QPushButton.mini-btn {{
    padding: 3px 8px;
    font-size: 11px;
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_COLOR};
    color: {TEXT_SECONDARY};
    border-radius: 4px;
}}

QPushButton.mini-btn:hover {{
    border-color: {BORDER_FOCUS};
    color: {TEXT_PRIMARY};
    background-color: {BG_HOVER};
}}

QPushButton.mini-btn:checked {{
    background-color: {ACCENT_TRACK};
    color: #ffffff;
    border-color: {ACCENT_PRIMARY};
    font-weight: bold;
}}

/* Inputs & Combos */
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    padding: 6px 10px;
    color: {TEXT_PRIMARY};
}}

QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {BORDER_FOCUS};
}}

QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {BORDER_FOCUS};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid {BORDER_COLOR};
}}

QComboBox QAbstractItemView {{
    background-color: {BG_CARD};
    border: 1px solid {BORDER_COLOR};
    selection-background-color: {BG_HOVER};
    selection-color: {TEXT_PRIMARY};
    padding: 4px;
}}

/* Radio Buttons */
QRadioButton {{
    spacing: 8px;
    color: {TEXT_PRIMARY};
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid {BORDER_COLOR};
    background-color: {BG_INPUT};
}}

QRadioButton::indicator:hover {{
    border-color: {BORDER_FOCUS};
}}

QRadioButton::indicator:checked {{
    border-color: {ACCENT_PRIMARY};
    background-color: {ACCENT_PRIMARY};
}}

/* Checkboxes */
QCheckBox {{
    spacing: 8px;
    color: {TEXT_PRIMARY};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {BORDER_COLOR};
    background-color: {BG_INPUT};
}}

QCheckBox::indicator:hover {{
    border-color: {BORDER_FOCUS};
}}

QCheckBox::indicator:checked {{
    border-color: {ACCENT_PRIMARY};
    background-color: {ACCENT_PRIMARY};
}}

/* Scrollbars */
QScrollBar:vertical {{
    border: none;
    background: {BG_DARKEST};
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {BORDER_COLOR};
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: {BORDER_FOCUS};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QScrollBar:horizontal {{
    border: none;
    background: {BG_DARKEST};
    height: 8px;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background: {BORDER_COLOR};
    min-width: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:horizontal:hover {{
    background: {BORDER_FOCUS};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* Status & Telemetry Bar */
QStatusBar, QFrame#telemetryBar {{
    background-color: {BG_DARKEST};
    border-top: 1px solid {BORDER_COLOR};
    color: {TEXT_SECONDARY};
    font-size: 12px;
    padding: 4px 12px;
}}

/* Tables / Tree Views */
QTableWidget, QTreeWidget {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER_COLOR};
    border-radius: 6px;
    gridline-color: {BORDER_COLOR};
    color: {TEXT_PRIMARY};
}}

QHeaderView::section {{
    background-color: {BG_CARD};
    color: {TEXT_SECONDARY};
    padding: 6px 10px;
    border: none;
    border-right: 1px solid {BORDER_COLOR};
    border-bottom: 1px solid {BORDER_COLOR};
    font-weight: 600;
    font-size: 12px;
}}

/* Progress Bar */
QProgressBar {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    background-color: {BG_INPUT};
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    height: 18px;
}}

QProgressBar::chunk {{
    background-color: {ACCENT_TRACK};
    border-radius: 3px;
}}

/* Tooltips */
QToolTip {{
    background-color: {BG_CARD_ALT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_FOCUS};
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 12px;
}}
"""
