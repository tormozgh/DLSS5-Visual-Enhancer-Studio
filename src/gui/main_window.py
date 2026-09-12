"""Central QMainWindow managing application tabs, telemetry, and drag-and-drop workflow."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QIcon
from PyQt6.QtWidgets import (
    QMainWindow,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.gpu_detection import detect_gpus
from ..core.paths import CONFIG_PATH
from ..settings.models import UISettings
from ..settings.storage import load_settings
from .components.telemetry import TelemetryBar
from .preview_engine import PreviewEngine
from .tabs.about_tab import AboutTab
from .tabs.frame_interpolation_tab import FrameInterpolationTab
from .tabs.neural_rendering_tab import NeuralRenderingTab
from .tabs.settings_tab import SettingsTab
from .tabs.upscale_tab import UpscaleTab


class MainWindow(QMainWindow):
    """Main application window for the DLSS 5 Visual Enhancer desktop suite."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DLSS 5 Visual Enhancer Studio")
        self.resize(1440, 920)
        self.setMinimumSize(1100, 720)
        self.setAcceptDrops(True)

        # 1. Load settings & Hardware detection
        self._settings = load_settings(CONFIG_PATH)
        self._gpus = detect_gpus()
        primary_gpu = self._gpus[0] if self._gpus else {}
        self._gpu_name = primary_gpu.get("display_name", "NVIDIA RTX GPU")
        self._vram_mb = int(primary_gpu.get("memory_mb", 0))
        self._driver = str(primary_gpu.get("driver", "Current"))

        gpu_choices = [("Automatic Selection", "auto")]
        for g in self._gpus:
            gpu_choices.append((f"{g.get('display_name', 'GPU')} ({g.get('memory_mb', 0)} MB)", str(g.get("uuid", ""))))

        # 2. Preview Engine
        self.preview_engine = PreviewEngine(self)

        # 3. Setup central UI
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Tab 1: Neural Rendering
        self.tab_neural = NeuralRenderingTab(self._settings, self.preview_engine, self)
        self.tabs.addTab(self.tab_neural, "Neural Rendering")

        # Tab 2: Upscale
        self.tab_upscale = UpscaleTab(self._settings, self)
        self.tabs.addTab(self.tab_upscale, "Upscale (VSR / HDR)")

        # Tab 3: Frame Interpolation
        self.tab_interpolation = FrameInterpolationTab(self._settings, self)
        self.tabs.addTab(self.tab_interpolation, "Frame Interpolation")

        # Tab 4: Settings
        self.tab_settings = SettingsTab(self._settings, gpu_choices, self)
        self.tabs.addTab(self.tab_settings, "Settings")

        # Tab 5: About
        self.tab_about = AboutTab(self._gpu_name, self._driver, self)
        self.tabs.addTab(self.tab_about, "About")

        main_layout.addWidget(self.tabs, 1)

        # Telemetry Bar
        self.telemetry = TelemetryBar(self)
        self.telemetry.set_gpu_info(self._gpu_name, self._vram_mb, self._driver)
        engine_mode = "VRAM Direct (D3D12/NGX)" if self._settings.nr_gpu_mode else "RAM Staging"
        self.telemetry.set_engine_mode(engine_mode)
        main_layout.addWidget(self.telemetry)

        # Connect signals
        self.tab_neural.statusMessage.connect(self.telemetry.set_status)
        self.tab_neural.latencyUpdated.connect(self.telemetry.set_timing)
        self.tab_upscale.statusMessage.connect(self.telemetry.set_status)
        self.tab_interpolation.statusMessage.connect(self.telemetry.set_status)
        self.tab_settings.settingsSaved.connect(self._on_settings_saved)

    def _on_settings_saved(self) -> None:
        self._settings = load_settings(CONFIG_PATH)
        self.tab_neural.update_settings(self._settings)
        self.tab_upscale.update_settings(self._settings)
        self.tab_interpolation.update_settings(self._settings)
        engine_mode = "VRAM Direct (D3D12/NGX)" if self._settings.nr_gpu_mode else "RAM Staging"
        self.telemetry.set_engine_mode(engine_mode)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.tabs.setCurrentIndex(0)
                self.tab_neural.load_file(path)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Shut down background threads and sessions gracefully."""
        self.preview_engine.shutdown()
        event.accept()
