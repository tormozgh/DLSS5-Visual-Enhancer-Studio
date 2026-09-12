"""Real-Time Rendering Tab: NDI 6 In/Out, NVIDIA Streamline 2.13, ReShade FX & NVENC Live Recording."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.paths import OUTPUTS
from ...core.reshade import ReShadePresetManager, ReShadeSettings
from ...live.engine import PipelineTelemetry, RealtimePipeline
from ...settings.models import UISettings
from ..components.sliders import LabeledSlider
from ..components.split_canvas import CanvasViewMode, SplitCanvas


class _SignalBridge(QObject):
    """Bridge for cross-thread signals from background pipeline to Qt GUI."""

    frameReady = pyqtSignal(object, object)
    telemetryUpdated = pyqtSignal(object)


class RealtimeRenderingTab(QWidget):
    """Real-time neural broadcast studio tab."""

    statusMessage = pyqtSignal(str, bool)

    def __init__(self, settings: UISettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._signals = _SignalBridge()
        self._signals.frameReady.connect(self._on_frame_ready_gui)
        self._signals.telemetryUpdated.connect(self._on_telemetry_gui)

        self._preset_manager = ReShadePresetManager()
        self._pipeline = RealtimePipeline(
            sender_name="DLSS 5 Visual Enhancer Studio",
            on_frame_ready=self._on_pipeline_frame_ready,
            on_telemetry=self._on_pipeline_telemetry,
        )

        self._init_ui()
        self._refresh_sources_list()

        # Periodic timer to refresh sources in dropdown
        self._source_timer = QTimer(self)
        self._source_timer.setInterval(3000)
        self._source_timer.timeout.connect(self._refresh_sources_list)
        self._source_timer.start()

        # Telemetry timer for recording duration update
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(500)
        self._rec_timer.timeout.connect(self._update_recorder_ui)
        self._rec_timer.start()

    def _init_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ----------------------------------------------------------------------
        # LEFT COLUMN: Control Sidebar (Scrollable)
        # ----------------------------------------------------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(360)
        left_scroll.setMaximumWidth(420)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        sidebar_layout.setSpacing(14)

        # 1. NDI Network Ingest Card
        ingest_box = QGroupBox("NDI Network Ingest")
        ib_layout = QVBoxLayout(ingest_box)
        ib_layout.setSpacing(8)

        src_select_layout = QHBoxLayout()
        self.cmb_sources = QComboBox()
        self.cmb_sources.addItem("Searching for NDI sources...", None)
        src_select_layout.addWidget(self.cmb_sources, 1)

        self.btn_refresh_sources = QPushButton("Scan")
        self.btn_refresh_sources.setProperty("class", "mini-btn")
        self.btn_refresh_sources.clicked.connect(self._refresh_sources_list)
        src_select_layout.addWidget(self.btn_refresh_sources)
        ib_layout.addLayout(src_select_layout)

        self.btn_toggle_stream = QPushButton("Connect & Start Live Stream")
        self.btn_toggle_stream.setProperty("class", "primary")
        self.btn_toggle_stream.clicked.connect(self._on_toggle_stream)
        ib_layout.addWidget(self.btn_toggle_stream)

        self.lbl_stream_status = QLabel("Status: Standby (No stream connected)")
        self.lbl_stream_status.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        ib_layout.addWidget(self.lbl_stream_status)

        sidebar_layout.addWidget(ingest_box)

        # 2. NVIDIA Streamline 2.13 Engine Card
        sl_box = QGroupBox("NVIDIA Streamline 2.13 Engine")
        sl_layout = QVBoxLayout(sl_box)
        sl_layout.setSpacing(8)

        self.chk_sl_enabled = QCheckBox("Enable Streamline Neural Enhancement")
        self.chk_sl_enabled.setChecked(True)
        self.chk_sl_enabled.toggled.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.chk_sl_enabled)

        self.chk_sl_nr = QCheckBox("DLSS-NR Neural Reconstruction")
        self.chk_sl_nr.setChecked(True)
        self.chk_sl_nr.toggled.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.chk_sl_nr)

        self.slider_nr_intensity = LabeledSlider("NR Denoising Intensity", 0.0, 1.0, 0.85, step=0.05, decimals=2)
        self.slider_nr_intensity.valueChanged.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.slider_nr_intensity)

        self.slider_nr_structure = LabeledSlider("Structural Detail", 0.0, 1.0, 0.65, step=0.05, decimals=2)
        self.slider_nr_structure.valueChanged.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.slider_nr_structure)

        self.chk_sl_frame_gen = QCheckBox("DLSS-G Multi-Frame Generation (2x FPS: 60 -> 120)")
        self.chk_sl_frame_gen.setChecked(False)
        self.chk_sl_frame_gen.toggled.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.chk_sl_frame_gen)

        lbl_nvof_tag = QLabel("Motion Vectors: RTX Hardware Optical Flow (NVOF)")
        lbl_nvof_tag.setStyleSheet("color: #7b8190; font-size: 10px;")
        sl_layout.addWidget(lbl_nvof_tag)

        sidebar_layout.addWidget(sl_box)

        # 3. ReShade FX Post-Processing Card
        rs_box = QGroupBox("ReShade FX Post-Processing")
        rs_layout = QVBoxLayout(rs_box)
        rs_layout.setSpacing(8)

        self.chk_rs_enabled = QCheckBox("Enable ReShade FX Shaders")
        self.chk_rs_enabled.setChecked(True)
        self.chk_rs_enabled.toggled.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.chk_rs_enabled)

        preset_row = QHBoxLayout()
        preset_label = QLabel("Preset:")
        preset_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        preset_row.addWidget(preset_label)

        self.cmb_presets = QComboBox()
        for p in self._preset_manager.get_preset_names():
            self.cmb_presets.addItem(p)
        self.cmb_presets.setCurrentText("Cinematic Teal & Orange")
        self.cmb_presets.currentTextChanged.connect(self._on_preset_selected)
        preset_row.addWidget(self.cmb_presets, 1)
        rs_layout.addLayout(preset_row)

        # 3D LUT Controls
        self.chk_lut = QCheckBox("3D LUT Color Grading")
        self.chk_lut.setChecked(True)
        self.chk_lut.toggled.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.chk_lut)

        self.cmb_lut_name = QComboBox()
        for name in self._pipeline.reshade.lut_manager.available_luts:
            self.cmb_lut_name.addItem(name)
        self.cmb_lut_name.setCurrentText("Cinematic Teal & Orange")
        self.cmb_lut_name.currentTextChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.cmb_lut_name)

        self.slider_lut_strength = LabeledSlider("LUT Strength", 0.0, 1.0, 0.85, step=0.05, decimals=2)
        self.slider_lut_strength.valueChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.slider_lut_strength)

        # ACES Tonemap Controls
        self.chk_tonemap = QCheckBox("ACES Filmic Dynamic Tonemapper")
        self.chk_tonemap.setChecked(True)
        self.chk_tonemap.toggled.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.chk_tonemap)

        self.slider_exposure = LabeledSlider("Exposure Bias", -2.0, 2.0, 0.0, step=0.05, decimals=2, suffix=" EV")
        self.slider_exposure.valueChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.slider_exposure)

        self.slider_contrast = LabeledSlider("Contrast", 0.5, 2.0, 1.05, step=0.05, decimals=2)
        self.slider_contrast.valueChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.slider_contrast)

        self.slider_saturation = LabeledSlider("Saturation", 0.0, 2.0, 1.10, step=0.05, decimals=2)
        self.slider_saturation.valueChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.slider_saturation)

        self.slider_temperature = LabeledSlider("Color Temperature", -1.0, 1.0, 0.0, step=0.05, decimals=2)
        self.slider_temperature.valueChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.slider_temperature)

        # Film Grain
        self.chk_grain = QCheckBox("ReShade FilmGrain.fx")
        self.chk_grain.setChecked(True)
        self.chk_grain.toggled.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.chk_grain)

        self.slider_grain_intensity = LabeledSlider("Film Grain Intensity", 0.0, 1.0, 0.15, step=0.01, decimals=2)
        self.slider_grain_intensity.valueChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.slider_grain_intensity)

        # CAS
        self.chk_cas = QCheckBox("CAS (Contrast Adaptive Sharpening)")
        self.chk_cas.setChecked(True)
        self.chk_cas.toggled.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.chk_cas)

        self.slider_cas_sharpness = LabeledSlider("CAS Sharpness", 0.0, 1.0, 0.40, step=0.05, decimals=2)
        self.slider_cas_sharpness.valueChanged.connect(self._on_rs_config_changed)
        rs_layout.addWidget(self.slider_cas_sharpness)

        sidebar_layout.addWidget(rs_box)

        # 4. NDI Broadcast Output Card
        ndi_out_box = QGroupBox("NDI Broadcast Output")
        out_layout = QVBoxLayout(ndi_out_box)
        out_layout.setSpacing(6)

        self.chk_broadcast_enabled = QCheckBox("Enable NDI Out: DLSS 5 Visual Enhancer Studio")
        self.chk_broadcast_enabled.setChecked(True)
        out_layout.addWidget(self.chk_broadcast_enabled)

        tally_layout = QHBoxLayout()
        tally_label = QLabel("Tally Status:")
        tally_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        tally_layout.addWidget(tally_label)

        self.badge_pgm = QLabel(" PGM ")
        self.badge_pgm.setStyleSheet("background-color: #262930; color: #6b7280; font-weight: bold; border-radius: 3px; font-size: 10px; padding: 2px 4px;")
        tally_layout.addWidget(self.badge_pgm)

        self.badge_pvw = QLabel(" PVW ")
        self.badge_pvw.setStyleSheet("background-color: #262930; color: #6b7280; font-weight: bold; border-radius: 3px; font-size: 10px; padding: 2px 4px;")
        tally_layout.addWidget(self.badge_pvw)
        tally_layout.addStretch()
        out_layout.addLayout(tally_layout)

        sidebar_layout.addWidget(ndi_out_box)

        # 5. Instant Live NVENC Recording Card
        rec_box = QGroupBox("Instant Live Recording")
        rec_layout = QVBoxLayout(rec_box)
        rec_layout.setSpacing(8)

        self.btn_record = QPushButton("Start Live Recording")
        self.btn_record.setProperty("class", "record-btn")
        self.btn_record.setStyleSheet("background-color: #7f1d1d; color: #fecaca; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.btn_record.clicked.connect(self._on_toggle_record)
        rec_layout.addWidget(self.btn_record)

        rec_settings_row = QHBoxLayout()
        self.cmb_rec_bitrate = QComboBox()
        self.cmb_rec_bitrate.addItem("15 Mbps (Standard)", 15)
        self.cmb_rec_bitrate.addItem("25 Mbps (Broadcast)", 25)
        self.cmb_rec_bitrate.addItem("50 Mbps (Master)", 50)
        self.cmb_rec_bitrate.setCurrentIndex(1)
        rec_settings_row.addWidget(self.cmb_rec_bitrate)

        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setProperty("class", "mini-btn")
        self.btn_open_folder.clicked.connect(self._on_open_recordings_folder)
        rec_settings_row.addWidget(self.btn_open_folder)
        rec_layout.addLayout(rec_settings_row)

        self.lbl_rec_status = QLabel("Recorder: Idle")
        self.lbl_rec_status.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        rec_layout.addWidget(self.lbl_rec_status)

        sidebar_layout.addWidget(rec_box)

        left_scroll.setWidget(sidebar)
        splitter.addWidget(left_scroll)

        # ----------------------------------------------------------------------
        # RIGHT COLUMN: Live Viewport & Telemetry HUD
        # ----------------------------------------------------------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Viewport Toolbar
        toolbar = QFrame()
        toolbar.setProperty("class", "toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(8)

        self.btn_mode_split = QPushButton("Split Slider")
        self.btn_mode_split.setProperty("class", "toolbar-btn")
        self.btn_mode_split.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.SPLIT))
        toolbar_layout.addWidget(self.btn_mode_split)

        self.btn_mode_side = QPushButton("Side-by-Side")
        self.btn_mode_side.setProperty("class", "toolbar-btn")
        self.btn_mode_side.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.SIDE_BY_SIDE))
        toolbar_layout.addWidget(self.btn_mode_side)

        self.btn_mode_after = QPushButton("Enhanced Only")
        self.btn_mode_after.setProperty("class", "toolbar-btn")
        self.btn_mode_after.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.ONLY_AFTER))
        toolbar_layout.addWidget(self.btn_mode_after)

        self.btn_mode_before = QPushButton("Original Only")
        self.btn_mode_before.setProperty("class", "toolbar-btn")
        self.btn_mode_before.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.ONLY_BEFORE))
        toolbar_layout.addWidget(self.btn_mode_before)

        toolbar_layout.addStretch()

        btn_fit = QPushButton("Fit")
        btn_fit.setProperty("class", "toolbar-btn")
        btn_fit.clicked.connect(lambda: self.canvas.fit_to_view())
        toolbar_layout.addWidget(btn_fit)

        right_layout.addWidget(toolbar)

        # Interactive Canvas
        self.canvas = SplitCanvas()
        right_layout.addWidget(self.canvas, 1)

        # Studio Telemetry Bar (Bottom)
        hud_frame = QFrame()
        hud_frame.setProperty("class", "studio-card")
        hud_frame.setStyleSheet("background-color: #14161a; border: 1px solid #22252c; border-radius: 4px; padding: 6px;")
        hud_layout = QHBoxLayout(hud_frame)
        hud_layout.setContentsMargins(10, 4, 10, 4)
        hud_layout.setSpacing(16)

        self.lbl_hud_res = QLabel("Resolution: --")
        self.lbl_hud_res.setStyleSheet("color: #d0d4dc; font-weight: 600; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_res)

        self.lbl_hud_fps = QLabel("FPS: -- (Render: --)")
        self.lbl_hud_fps.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_fps)

        self.lbl_hud_latency = QLabel("Latency: -- ms")
        self.lbl_hud_latency.setStyleSheet("color: #4ade80; font-weight: 600; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_latency)

        self.lbl_hud_pipeline = QLabel("Engine: Streamline 2.13 + ReShade FX")
        self.lbl_hud_pipeline.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_pipeline)

        hud_layout.addStretch()

        self.lbl_hud_rec = QLabel("REC: OFF")
        self.lbl_hud_rec.setStyleSheet("color: #6b7280; font-weight: bold; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_rec)

        right_layout.addWidget(hud_frame)

        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter)

    def _refresh_sources_list(self) -> None:
        sources = self._pipeline.get_sources()
        current_data = self.cmb_sources.currentData()

        # Update items only if changed
        items = [(name, url) for name, url in sources]
        existing = [self.cmb_sources.itemData(i) for i in range(self.cmb_sources.count())]

        if [x[1] for x in items] != existing:
            self.cmb_sources.blockSignals(True)
            self.cmb_sources.clear()
            if not items:
                self.cmb_sources.addItem("No NDI sources detected", None)
            else:
                for name, url in items:
                    self.cmb_sources.addItem(name, url)
                    if url == current_data:
                        self.cmb_sources.setCurrentIndex(self.cmb_sources.count() - 1)
            self.cmb_sources.blockSignals(False)

    def _on_toggle_stream(self) -> None:
        if self._pipeline.is_running:
            self._pipeline.stop_pipeline()
            self.btn_toggle_stream.setText("Connect & Start Live Stream")
            self.btn_toggle_stream.setStyleSheet("")
            self.lbl_stream_status.setText("Status: Stopped")
            self.statusMessage.emit("Live broadcast pipeline stopped", False)
        else:
            url = self.cmb_sources.currentData()
            name = self.cmb_sources.currentText()
            if not name or "No NDI sources" in name or "Searching" in name:
                QMessageBox.warning(
                    self, "No NDI Source Selected", "Please select an active NDI broadcast source."
                )
                return

            try:
                self._pipeline.start_pipeline(
                    source_name=name,
                    url_address=url,
                    enable_ndi_out=self.chk_broadcast_enabled.isChecked(),
                )
                self.btn_toggle_stream.setText("Stop Live Stream")
                self.btn_toggle_stream.setStyleSheet("background-color: #991b1b; color: white;")
                self.lbl_stream_status.setText(f"Status: Streaming from {name}")
                self.statusMessage.emit(f"Connected to NDI source: {name}", False)
            except Exception as exc:
                QMessageBox.critical(self, "Connection Error", f"Failed to start NDI pipeline: {exc}")

    def _on_sl_config_changed(self) -> None:
        cfg = self._pipeline.streamline.config
        cfg.enabled = self.chk_sl_enabled.isChecked()
        cfg.enable_dlss_nr = self.chk_sl_nr.isChecked()
        cfg.enable_frame_gen = self.chk_sl_frame_gen.isChecked()
        cfg.nr_intensity = float(self.slider_nr_intensity.value())
        cfg.nr_structure = float(self.slider_nr_structure.value())

    def _on_rs_config_changed(self) -> None:
        s = self._pipeline.reshade.settings
        s.enabled = self.chk_rs_enabled.isChecked()
        s.lut_enabled = self.chk_lut.isChecked()
        s.lut_name = self.cmb_lut_name.currentText()
        s.lut_strength = float(self.slider_lut_strength.value())

        s.tonemap_enabled = self.chk_tonemap.isChecked()
        s.exposure = float(self.slider_exposure.value())
        s.contrast = float(self.slider_contrast.value())
        s.saturation = float(self.slider_saturation.value())
        s.color_temperature = float(self.slider_temperature.value())

        s.grain_enabled = self.chk_grain.isChecked()
        s.grain_intensity = float(self.slider_grain_intensity.value())

        s.cas_enabled = self.chk_cas.isChecked()
        s.cas_sharpness = float(self.slider_cas_sharpness.value())

    def _on_preset_selected(self, preset_name: str) -> None:
        if not preset_name:
            return
        settings = self._preset_manager.load_preset(preset_name)
        self._pipeline.reshade.settings = settings

        # Synchronize UI widgets
        self.chk_rs_enabled.setChecked(settings.enabled)
        self.chk_lut.setChecked(settings.lut_enabled)
        self.cmb_lut_name.setCurrentText(settings.lut_name)
        self.slider_lut_strength.setValue(settings.lut_strength)

        self.chk_tonemap.setChecked(settings.tonemap_enabled)
        self.slider_exposure.setValue(settings.exposure)
        self.slider_contrast.setValue(settings.contrast)
        self.slider_saturation.setValue(settings.saturation)
        self.slider_temperature.setValue(settings.color_temperature)

        self.chk_grain.setChecked(settings.grain_enabled)
        self.slider_grain_intensity.setValue(settings.grain_intensity)

        self.chk_cas.setChecked(settings.cas_enabled)
        self.slider_cas_sharpness.setValue(settings.cas_sharpness)

    def _on_toggle_record(self) -> None:
        if self._pipeline.recorder.is_recording:
            saved = self._pipeline.stop_recording()
            self.btn_record.setText("Start Live Recording")
            self.btn_record.setStyleSheet("background-color: #7f1d1d; color: #fecaca; font-weight: bold; padding: 8px; border-radius: 4px;")
            self.lbl_rec_status.setText("Recorder: Idle")
            if saved and saved.is_file():
                self.statusMessage.emit(f"Recording saved: {saved.name}", False)
        else:
            bitrate = self.cmb_rec_bitrate.currentData() or 25
            path = self._pipeline.start_recording(bitrate_mbps=bitrate)
            if path:
                self.btn_record.setText("Stop Recording (REC)")
                self.btn_record.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
                self.lbl_rec_status.setText(f"Recording: {path.name}")
                self.statusMessage.emit(f"Hardware live recording started: {path.name}", False)
            else:
                QMessageBox.warning(
                    self, "Live Recording", "Connect to an active NDI stream before starting recording."
                )

    def _on_open_recordings_folder(self) -> None:
        os.startfile(str(OUTPUTS))

    def _on_pipeline_frame_ready(self, original: np.ndarray, enhanced: np.ndarray) -> None:
        self._signals.frameReady.emit(original, enhanced)

    def _on_pipeline_telemetry(self, telem: PipelineTelemetry) -> None:
        self._signals.telemetryUpdated.emit(telem)

    def _on_frame_ready_gui(self, orig: np.ndarray, enh: np.ndarray) -> None:
        ho, wo = orig.shape[:2]
        stride_o = orig.strides[0]
        qimg_before = QImage(orig.data, wo, ho, stride_o, QImage.Format.Format_RGBA8888).copy()

        he, we = enh.shape[:2]
        stride_e = enh.strides[0]
        qimg_after = QImage(enh.data, we, he, stride_e, QImage.Format.Format_RGBA8888).copy()

        self.canvas.set_images(qimg_before, qimg_after)

    def _on_telemetry_gui(self, telem: PipelineTelemetry) -> None:
        w, h = telem.input_resolution
        self.lbl_hud_res.setText(f"Resolution: {w}x{h}")
        self.lbl_hud_fps.setText(f"In: {telem.input_fps:.1f} FPS | Out: {telem.render_fps:.1f} FPS")
        self.lbl_hud_latency.setText(f"Latency: {telem.latency_ms:.1f} ms")

        # Tally badges
        if telem.tally_program:
            self.badge_pgm.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; border-radius: 3px; font-size: 10px; padding: 2px 4px;")
        else:
            self.badge_pgm.setStyleSheet("background-color: #262930; color: #6b7280; font-weight: bold; border-radius: 3px; font-size: 10px; padding: 2px 4px;")

        if telem.tally_preview:
            self.badge_pvw.setStyleSheet("background-color: #16a34a; color: white; font-weight: bold; border-radius: 3px; font-size: 10px; padding: 2px 4px;")
        else:
            self.badge_pvw.setStyleSheet("background-color: #262930; color: #6b7280; font-weight: bold; border-radius: 3px; font-size: 10px; padding: 2px 4px;")

    def _update_recorder_ui(self) -> None:
        rec_telem = self._pipeline.recorder.get_telemetry()
        if rec_telem.is_recording:
            mins = int(rec_telem.duration_seconds // 60)
            secs = int(rec_telem.duration_seconds % 60)
            dur_str = f"{mins:02d}:{secs:02d}"
            self.lbl_hud_rec.setText(f"REC: {dur_str} ({rec_telem.file_size_mb:.1f} MB)")
            self.lbl_hud_rec.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px;")
            self.lbl_rec_status.setText(f"Writing {dur_str} | {rec_telem.file_size_mb:.1f} MB | Dropped: {rec_telem.dropped_frames}")
        else:
            self.lbl_hud_rec.setText("REC: OFF")
            self.lbl_hud_rec.setStyleSheet("color: #6b7280; font-weight: bold; font-size: 11px;")

    def shutdown(self) -> None:
        """Clean shutdown on application close."""
        self._source_timer.stop()
        self._rec_timer.stop()
        self._pipeline.close()
