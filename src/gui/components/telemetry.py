"""Hardware telemetry and real-time inference latency monitor in neutral gray."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QWidget,
)


class TelemetryBar(QFrame):
    """Compact status bar showing active GPU, VRAM, and real-time frame timings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(34)
        self.setObjectName("telemetryBar")
        self.setStyleSheet(
            "QFrame#telemetryBar { background-color: #101114; border-top: 1px solid #323640; color: #9ca0ab; font-size: 12px; }"
        )
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(16)

        # GPU Name Badge
        self.gpu_label = QLabel("GPU: Detecting...")
        self.gpu_label.setStyleSheet("color: #f0f1f4; font-weight: 600;")

        # Driver & VRAM
        self.vram_label = QLabel("VRAM: -- / --")

        # Engine Mode
        self.engine_label = QLabel("Engine: VRAM Direct (D3D12/NGX)")
        self.engine_label.setStyleSheet("color: #d0d4dc;")

        # Real-time inference latency
        self.latency_label = QLabel("Latency: Ready")
        self.latency_label.setStyleSheet("color: #b0b5c0;")

        # Status text
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #9ca0ab;")

        layout.addWidget(self.gpu_label)
        layout.addWidget(self.vram_label)
        layout.addWidget(self.engine_label)
        layout.addStretch()
        layout.addWidget(self.latency_label)
        layout.addWidget(self.status_label)

    def set_gpu_info(self, name: str, memory_mb: int, driver: str = "") -> None:
        vram_gb = memory_mb / 1024.0
        self.gpu_label.setText(f"{name}")
        self.vram_label.setText(f"VRAM: {vram_gb:.1f} GB | Driver: {driver}")

    def set_engine_mode(self, mode: str) -> None:
        self.engine_label.setText(f"Engine: {mode}")

    def set_timing(self, latency_ms: float) -> None:
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        self.latency_label.setText(f"Inference: {latency_ms:.1f}ms ({fps:.0f} FPS)")

    def set_status(self, text: str, is_error: bool = False) -> None:
        color = "#e57373" if is_error else "#9ca0ab"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(text)
