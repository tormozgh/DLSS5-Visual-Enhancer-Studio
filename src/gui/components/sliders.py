"""Custom styled sliders with value badges and reset actions."""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class LabeledSlider(QWidget):
    """Parameter slider with title, live value badge, and reset button."""

    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        default: float,
        step: float = 0.05,
        decimals: int = 2,
        suffix: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.label_text = label
        self._min = minimum
        self._max = maximum
        self._default = default
        self._step = step
        self._decimals = decimals
        self._suffix = suffix

        self._steps_count = int(round((maximum - minimum) / step))
        self._current_value = default

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)

        self.title_label = QLabel(self.label_text)
        self.title_label.setStyleSheet("font-weight: 600; color: #f0f1f4; font-size: 12px;")

        self.value_badge = QLabel(self._format_value(self._current_value))
        self.value_badge.setStyleSheet(
            "background-color: #18191d; color: #d0d4dc; font-weight: 700; "
            "padding: 2px 8px; border-radius: 4px; border: 1px solid #353841; font-size: 11px;"
        )

        self.reset_btn = QPushButton("R")
        self.reset_btn.setToolTip(f"Reset to default ({self._format_value(self._default)})")
        self.reset_btn.setProperty("class", "mini-btn")
        self.reset_btn.setFixedSize(22, 20)
        self.reset_btn.setStyleSheet("font-size: 10px; font-weight: 700;")
        self.reset_btn.clicked.connect(self.reset_to_default)

        header.addWidget(self.title_label)
        header.addStretch()
        header.addWidget(self.value_badge)
        header.addWidget(self.reset_btn)

        layout.addLayout(header)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._steps_count)
        self.slider.setValue(self._val_to_slider(self._current_value))
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderMoved.connect(self._on_slider_changed)

        layout.addWidget(self.slider)

    def _format_value(self, val: float) -> str:
        if self._decimals == 0:
            return f"{int(round(val))}{self._suffix}"
        return f"{val:.{self._decimals}f}{self._suffix}"

    def _val_to_slider(self, val: float) -> int:
        clamped = max(self._min, min(self._max, val))
        ratio = (clamped - self._min) / (self._max - self._min) if self._max > self._min else 0
        return int(round(ratio * self._steps_count))

    def _slider_to_val(self, pos: int) -> float:
        ratio = pos / self._steps_count if self._steps_count > 0 else 0
        val = self._min + ratio * (self._max - self._min)
        if self._decimals == 0:
            return float(round(val))
        return round(val, self._decimals)

    def _on_slider_changed(self, pos: int) -> None:
        val = self._slider_to_val(pos)
        if val != self._current_value:
            self._current_value = val
            self.value_badge.setText(self._format_value(val))
            self.valueChanged.emit(val)

    def value(self) -> float:
        return self._current_value

    def setValue(self, val: float) -> None:
        clamped = max(self._min, min(self._max, val))
        self._current_value = clamped
        self.slider.blockSignals(True)
        self.slider.setValue(self._val_to_slider(clamped))
        self.slider.blockSignals(False)
        self.value_badge.setText(self._format_value(clamped))

    def reset_to_default(self) -> None:
        self.setValue(self._default)
        self.valueChanged.emit(self._default)
