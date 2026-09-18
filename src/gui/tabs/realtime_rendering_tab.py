"""Real-Time Rendering Tab: NDI 6 In/Out, NVIDIA Streamline 2.13, ReShade FX & NVENC Live Recording."""

from __future__ import annotations

import os
import threading
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
    QLineEdit,
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
    frameProduced = pyqtSignal(object, object)  # (original_rgba, enhanced_rgba) for inter-tab feed

    def __init__(self, settings: UISettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._signals = _SignalBridge()
        self._signals.telemetryUpdated.connect(self._on_telemetry_gui)

        self._preset_manager = ReShadePresetManager()
        self._frame_lock = threading.Lock()
        self._latest_frames: tuple[np.ndarray, np.ndarray] | None = None

        self._pipeline = RealtimePipeline(
            sender_name="DLSS 5 Visual Enhancer Studio",
            on_frame_ready=self._on_pipeline_frame_ready,
            on_telemetry=self._on_pipeline_telemetry,
        )

        self._init_ui()
        self._refresh_sources_list()

        # Viewport render timer (decoupled from 60+ FPS broadcast loop)
        self._viewport_timer = QTimer(self)
        self._viewport_timer.setInterval(25)  # 40 FPS UI monitor refresh
        self._viewport_timer.timeout.connect(self._render_viewport_tick)
        self._viewport_timer.start()

        # Periodic timer to refresh sources in dropdown
        self._source_timer = QTimer(self)
        self._source_timer.setInterval(3000)
        self._source_timer.timeout.connect(self._refresh_sources_list)
        self._source_timer.start()

        # Camera auto-refresh timer (instant registry check)
        self._cam_timer = QTimer(self)
        self._cam_timer.setInterval(2000)
        self._cam_timer.timeout.connect(self._auto_refresh_cameras)
        self._cam_timer.start()

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
        # RIGHT COLUMN: Control Sidebar (Scrollable)
        # ----------------------------------------------------------------------
        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setMinimumWidth(320)
        sidebar_scroll = self.sidebar_scroll

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(12)

        # 1. Live Input Source Selection (NDI & Webcam)
        ingest_box = QGroupBox("Live Input Source")
        ib_layout = QVBoxLayout(ingest_box)
        ib_layout.setSpacing(8)

        mode_row = QHBoxLayout()
        mode_lbl = QLabel("Input Type:")
        mode_lbl.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        mode_row.addWidget(mode_lbl)

        self.cmb_input_type = QComboBox()
        self.cmb_input_type.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_input_type.setMinimumContentsLength(8)
        self.cmb_input_type.addItem("NDI Network Stream", "ndi")
        self.cmb_input_type.addItem("Webcam / Capture Card", "webcam")
        self.cmb_input_type.currentIndexChanged.connect(self._on_input_type_changed)
        mode_row.addWidget(self.cmb_input_type, 1)
        ib_layout.addLayout(mode_row)

        # NDI Selector
        self.widget_ndi_input = QWidget()
        ndi_input_layout = QHBoxLayout(self.widget_ndi_input)
        ndi_input_layout.setContentsMargins(0, 0, 0, 0)
        ndi_input_layout.setSpacing(6)
        self.cmb_sources = QComboBox()
        self.cmb_sources.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_sources.setMinimumContentsLength(8)
        self.cmb_sources.addItem("Searching for NDI sources...", None)
        ndi_input_layout.addWidget(self.cmb_sources, 1)

        self.btn_refresh_sources = QPushButton("Scan")
        self.btn_refresh_sources.setProperty("class", "mini-btn")
        self.btn_refresh_sources.clicked.connect(self._refresh_sources_list)
        ndi_input_layout.addWidget(self.btn_refresh_sources)
        ib_layout.addWidget(self.widget_ndi_input)

        # Webcam Selector
        self.widget_webcam_input = QWidget()
        webcam_input_layout = QHBoxLayout(self.widget_webcam_input)
        webcam_input_layout.setContentsMargins(0, 0, 0, 0)
        webcam_input_layout.setSpacing(6)
        self.cmb_cameras = QComboBox()
        self.cmb_cameras.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_cameras.setMinimumContentsLength(8)
        self.cmb_cameras.addItem("Detecting video devices...", None)
        webcam_input_layout.addWidget(self.cmb_cameras, 1)

        self.btn_refresh_cameras = QPushButton("Scan")
        self.btn_refresh_cameras.setProperty("class", "mini-btn")
        self.btn_refresh_cameras.clicked.connect(self._refresh_cameras_list)
        webcam_input_layout.addWidget(self.btn_refresh_cameras)
        self.widget_webcam_input.setVisible(False)
        ib_layout.addWidget(self.widget_webcam_input)

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

        self.chk_sl_enabled = QCheckBox("Enable Streamline Enhancement")
        self.chk_sl_enabled.setChecked(True)
        self.chk_sl_enabled.toggled.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.chk_sl_enabled)

        # Engine Mode Selector (Neural AI vs Fast Spatial)
        mode_row = QHBoxLayout()
        mode_label = QLabel("Reconstruction Engine:")
        mode_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        mode_row.addWidget(mode_label)

        self.cmb_engine_mode = QComboBox()
        self.cmb_engine_mode.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_engine_mode.setMinimumContentsLength(8)
        self.cmb_engine_mode.addItem("NVIDIA DLSS 5 Neural AI (Tensor Cores)", "neural")
        self.cmb_engine_mode.addItem("Ultra-Fast Spatial (NVIDIA NIS 60+ FPS)", "spatial")
        self.cmb_engine_mode.currentIndexChanged.connect(self._on_sl_config_changed)
        mode_row.addWidget(self.cmb_engine_mode, 1)
        sl_layout.addLayout(mode_row)

        self.chk_sl_nr = QCheckBox("DLSS-NR Neural Reconstruction")
        self.chk_sl_nr.setChecked(True)
        self.chk_sl_nr.toggled.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.chk_sl_nr)

        # Reconstruction Style
        style_row = QHBoxLayout()
        style_label = QLabel("Reconstruction Style:")
        style_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        style_row.addWidget(style_label)

        self.cmb_nr_style = QComboBox()
        self.cmb_nr_style.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_nr_style.setMinimumContentsLength(8)
        self.cmb_nr_style.addItem("Default", "Default")
        self.cmb_nr_style.addItem("Natural", "Natural")
        self.cmb_nr_style.addItem("Cinematic", "Cinematic")
        self.cmb_nr_style.currentIndexChanged.connect(self._on_sl_config_changed)
        style_row.addWidget(self.cmb_nr_style, 1)
        sl_layout.addLayout(style_row)

        self.slider_nr_intensity = LabeledSlider("NR Denoising Intensity", 0.0, 1.0, 0.85, step=0.05, decimals=2)
        self.slider_nr_intensity.valueChanged.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.slider_nr_intensity)

        self.slider_nr_tone = LabeledSlider("Local Tone Strength", 0.0, 2.0, 0.50, step=0.05, decimals=2)
        self.slider_nr_tone.valueChanged.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.slider_nr_tone)

        self.slider_nr_structure = LabeledSlider("Structural Detail", 0.0, 1.0, 0.65, step=0.05, decimals=2)
        self.slider_nr_structure.valueChanged.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.slider_nr_structure)

        self.slider_nr_skin = LabeledSlider("Skin / Face Protection", -1.0, 2.0, 0.50, step=0.05, decimals=2)
        self.slider_nr_skin.valueChanged.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.slider_nr_skin)

        self.chk_sl_frame_gen = QCheckBox("DLSS-G Multi-Frame Generation (2x FPS: 60 -> 120)")
        self.chk_sl_frame_gen.setChecked(False)
        self.chk_sl_frame_gen.toggled.connect(self._on_sl_config_changed)
        sl_layout.addWidget(self.chk_sl_frame_gen)

        lbl_nvof_tag = QLabel("Motion Vectors: RTX Hardware Optical Flow (NVOF)")
        lbl_nvof_tag.setStyleSheet("color: #7b8190; font-size: 10px;")
        sl_layout.addWidget(lbl_nvof_tag)

        lbl_neural_note = QLabel("Neural mode executes nvngx_dlssnr.dll deep learning models via D3D12/CUDA on RTX Tensor Cores.")
        lbl_neural_note.setStyleSheet("color: #7b8190; font-size: 10px;")
        lbl_neural_note.setWordWrap(True)
        sl_layout.addWidget(lbl_neural_note)

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
        self.cmb_presets.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_presets.setMinimumContentsLength(8)
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
        self.cmb_lut_name.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_lut_name.setMinimumContentsLength(8)
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

        # 5. Live Recording Configuration & Capture Card
        rec_box = QGroupBox("Live Recording Settings & Control")
        rec_layout = QVBoxLayout(rec_box)
        rec_layout.setSpacing(8)

        # Storage location row
        loc_label = QLabel("Storage Location:")
        loc_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        rec_layout.addWidget(loc_label)

        loc_row = QHBoxLayout()
        self.line_rec_dir = QLineEdit(str(OUTPUTS))
        self.line_rec_dir.setStyleSheet("background-color: #17191e; border: 1px solid #282b33; border-radius: 4px; padding: 4px 8px; color: #d0d4dc; font-size: 11px;")
        loc_row.addWidget(self.line_rec_dir, 1)

        self.btn_browse_rec_dir = QPushButton("Browse")
        self.btn_browse_rec_dir.setProperty("class", "mini-btn")
        self.btn_browse_rec_dir.clicked.connect(self._on_browse_rec_folder)
        loc_row.addWidget(self.btn_browse_rec_dir)

        self.btn_open_folder = QPushButton("Open")
        self.btn_open_folder.setProperty("class", "mini-btn")
        self.btn_open_folder.clicked.connect(self._on_open_recordings_folder)
        loc_row.addWidget(self.btn_open_folder)
        rec_layout.addLayout(loc_row)

        # File Name Prefix
        prefix_label = QLabel("File Name Prefix:")
        prefix_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        rec_layout.addWidget(prefix_label)

        self.line_rec_prefix = QLineEdit("DLSS5_Live")
        self.line_rec_prefix.setPlaceholderText("Prefix (e.g. Broadcast_Cam1)")
        self.line_rec_prefix.setStyleSheet("background-color: #17191e; border: 1px solid #282b33; border-radius: 4px; padding: 4px 8px; color: #d0d4dc; font-size: 11px;")
        rec_layout.addWidget(self.line_rec_prefix)

        # Container Format & Output Resolution
        fmt_res_row = QHBoxLayout()
        fmt_res_row.setSpacing(8)

        fmt_col = QVBoxLayout()
        fmt_label = QLabel("Format:")
        fmt_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        fmt_col.addWidget(fmt_label)
        self.cmb_rec_format = QComboBox()
        self.cmb_rec_format.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_rec_format.setMinimumContentsLength(8)
        self.cmb_rec_format.addItem("MP4 (.mp4)", "mp4")
        self.cmb_rec_format.addItem("MKV (.mkv)", "mkv")
        self.cmb_rec_format.addItem("MOV (.mov)", "mov")
        fmt_col.addWidget(self.cmb_rec_format)
        fmt_res_row.addLayout(fmt_col, 1)

        res_col = QVBoxLayout()
        res_label = QLabel("Resolution:")
        res_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        res_col.addWidget(res_label)
        self.cmb_rec_res = QComboBox()
        self.cmb_rec_res.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_rec_res.setMinimumContentsLength(8)
        self.cmb_rec_res.addItem("Match Stream (Auto)", (0, 0))
        self.cmb_rec_res.addItem("1080p FHD (1920x1080)", (1920, 1080))
        self.cmb_rec_res.addItem("1440p 2K (2560x1440)", (2560, 1440))
        self.cmb_rec_res.addItem("4K UHD (3840x2160)", (3840, 2160))
        self.cmb_rec_res.addItem("720p HD (1280x720)", (1280, 720))
        res_col.addWidget(self.cmb_rec_res)
        fmt_res_row.addLayout(res_col, 1)

        rec_layout.addLayout(fmt_res_row)

        # Bitrate
        bitrate_label = QLabel("Video Bitrate:")
        bitrate_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        rec_layout.addWidget(bitrate_label)

        self.cmb_rec_bitrate = QComboBox()
        self.cmb_rec_bitrate.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_rec_bitrate.setMinimumContentsLength(8)
        self.cmb_rec_bitrate.addItem("15 Mbps (Standard Quality)", 15)
        self.cmb_rec_bitrate.addItem("25 Mbps (Broadcast Standard)", 25)
        self.cmb_rec_bitrate.addItem("50 Mbps (High Bitrate Studio)", 50)
        self.cmb_rec_bitrate.addItem("80 Mbps (Master Archive)", 80)
        self.cmb_rec_bitrate.setCurrentIndex(1)
        rec_layout.addWidget(self.cmb_rec_bitrate)

        # Record Trigger Button
        self.btn_record = QPushButton("Start Live Recording")
        self.btn_record.setProperty("class", "record-btn")
        self.btn_record.setStyleSheet("background-color: #7f1d1d; color: #fecaca; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 12px;")
        self.btn_record.clicked.connect(self._on_toggle_record)
        rec_layout.addWidget(self.btn_record)

        # Recording Status Label
        self.lbl_rec_status = QLabel("Recorder: Idle")
        self.lbl_rec_status.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        rec_layout.addWidget(self.lbl_rec_status)

        sidebar_layout.addWidget(rec_box)

        sidebar_scroll.setWidget(sidebar)

        # ----------------------------------------------------------------------
        # LEFT COLUMN: Live Viewport Canvas & Telemetry HUD
        # ----------------------------------------------------------------------
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(8)

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

        viewport_layout.addWidget(toolbar)

        # Interactive Canvas
        self.canvas = SplitCanvas()
        viewport_layout.addWidget(self.canvas, 1)

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

        self.lbl_hud_pipeline = QLabel("Engine: Streamline 2.13 + NIS + ReShade FX")
        self.lbl_hud_pipeline.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_pipeline)

        hud_layout.addStretch()

        self.lbl_hud_rec = QLabel("REC: OFF")
        self.lbl_hud_rec.setStyleSheet("color: #6b7280; font-weight: bold; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_rec)

        viewport_layout.addWidget(hud_frame)

        # Add to splitter in left-to-right order: Canvas on LEFT, Sidebar on RIGHT
        splitter.addWidget(viewport_container)
        splitter.addWidget(sidebar_scroll)
        splitter.setStretchFactor(0, 1)  # Left (Canvas) expands
        splitter.setStretchFactor(1, 0)  # Right (Sidebar)
        splitter.setSizes([850, 420])
        splitter.setHandleWidth(6)

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

    def _on_input_type_changed(self) -> None:
        mode = self.cmb_input_type.currentData()
        self.widget_ndi_input.setVisible(mode == "ndi")
        self.widget_webcam_input.setVisible(mode == "webcam")
        if mode == "webcam":
            self._refresh_cameras_list()

    def _refresh_cameras_list(self) -> None:
        cameras = self._pipeline.get_cameras()
        current = self.cmb_cameras.currentData()
        self.cmb_cameras.blockSignals(True)
        self.cmb_cameras.clear()
        if not cameras:
            self.cmb_cameras.addItem("No video capture devices found", None)
        else:
            for cam in cameras:
                self.cmb_cameras.addItem(cam.display_name, cam.index)
                if cam.index == current:
                    self.cmb_cameras.setCurrentIndex(self.cmb_cameras.count() - 1)
        self.cmb_cameras.blockSignals(False)

    def _auto_refresh_cameras(self) -> None:
        """Periodic background refresh to detect newly connected webcams instantly."""
        if self.cmb_input_type.currentData() != "webcam":
            return
        cameras = self._pipeline.get_cameras()
        current = self.cmb_cameras.currentData()
        items = [(cam.display_name, cam.index) for cam in cameras]
        existing = [(self.cmb_cameras.itemText(i), self.cmb_cameras.itemData(i)) for i in range(self.cmb_cameras.count())]
        if items != existing:
            self.cmb_cameras.blockSignals(True)
            self.cmb_cameras.clear()
            if not items:
                self.cmb_cameras.addItem("No video capture devices found", None)
            else:
                for name, idx in items:
                    self.cmb_cameras.addItem(name, idx)
                    if idx == current:
                        self.cmb_cameras.setCurrentIndex(self.cmb_cameras.count() - 1)
            self.cmb_cameras.blockSignals(False)

    def _on_toggle_stream(self) -> None:
        if self._pipeline.is_running:
            self._pipeline.stop_pipeline()
            self.btn_toggle_stream.setText("Connect & Start Live Stream")
            self.btn_toggle_stream.setStyleSheet("")
            self.lbl_stream_status.setText("Status: Stopped")
            self.statusMessage.emit("Live broadcast pipeline stopped", False)
        else:
            mode = self.cmb_input_type.currentData()
            enable_out = self.chk_broadcast_enabled.isChecked()

            if mode == "webcam":
                dev_idx = self.cmb_cameras.currentData()
                if dev_idx is None:
                    QMessageBox.warning(self, "No Camera Selected", "Please select an active video capture device.")
                    return

                cams = self._pipeline.get_cameras()
                cam_info = next((c for c in cams if c.index == dev_idx), None)
                w = cam_info.width if cam_info else 1920
                h = cam_info.height if cam_info else 1080
                fps = cam_info.fps if cam_info else 60.0
                cam_name = cam_info.name if cam_info else self.cmb_cameras.currentText()

                try:
                    self._pipeline.start_webcam_pipeline(
                        device_index=dev_idx,
                        width=w,
                        height=h,
                        fps=fps,
                        enable_ndi_out=enable_out,
                        source_name=cam_name,
                    )
                    self.btn_toggle_stream.setText("Stop Live Stream")
                    self.btn_toggle_stream.setStyleSheet("background-color: #991b1b; color: white;")
                    self.lbl_stream_status.setText(f"Status: Streaming from {cam_name}")
                    self.statusMessage.emit(f"Connected to camera: {cam_name}", False)
                except Exception as exc:
                    QMessageBox.critical(self, "Camera Error", f"Failed to start camera capture: {exc}")
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
                        enable_ndi_out=enable_out,
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
        cfg.engine_mode = self.cmb_engine_mode.currentData() or "neural"
        cfg.enable_dlss_nr = self.chk_sl_nr.isChecked()
        cfg.enable_frame_gen = self.chk_sl_frame_gen.isChecked()
        cfg.nr_style = self.cmb_nr_style.currentText() or "Default"
        cfg.nr_intensity = float(self.slider_nr_intensity.value())
        cfg.nr_tone = float(self.slider_nr_tone.value())
        cfg.nr_structure = float(self.slider_nr_structure.value())
        cfg.skin_structure = float(self.slider_nr_skin.value())
        self._pipeline.reprocess_last_frame()
        self._render_viewport_tick()

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
        self._pipeline.reprocess_last_frame()
        self._render_viewport_tick()

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
        self._pipeline.reprocess_last_frame()
        self._render_viewport_tick()

    def _on_browse_rec_folder(self) -> None:
        current = self.line_rec_dir.text().strip() or str(OUTPUTS)
        chosen = QFileDialog.getExistingDirectory(self, "Select Recording Directory", current)
        if chosen:
            self.line_rec_dir.setText(chosen)

    def _on_toggle_record(self) -> None:
        if self._pipeline.recorder.is_recording:
            saved = self._pipeline.stop_recording()
            self.btn_record.setText("Start Live Recording")
            self.btn_record.setStyleSheet(
                "background-color: #7f1d1d; color: #fecaca; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 12px;"
            )
            self.lbl_rec_status.setText("Recorder: Idle")
            self.lbl_hud_rec.setText("REC: OFF")
            self.lbl_hud_rec.setStyleSheet("color: #6b7280; font-weight: bold; font-size: 11px;")
            if saved and saved.is_file():
                self.statusMessage.emit(f"Recording saved: {saved.name}", False)
        else:
            bitrate = self.cmb_rec_bitrate.currentData() or 25
            fmt = self.cmb_rec_format.currentData() or "mp4"
            res = self.cmb_rec_res.currentData() or (0, 0)
            rec_dir_str = self.line_rec_dir.text().strip()
            rec_dir = Path(rec_dir_str) if rec_dir_str else OUTPUTS
            prefix = self.line_rec_prefix.text().strip() or "DLSS5_Live"

            path = self._pipeline.start_recording(
                bitrate_mbps=bitrate,
                format_ext=fmt,
                target_resolution=res,
                output_dir=rec_dir,
                filename_prefix=prefix,
            )
            if path:
                self.btn_record.setText("Stop Recording (REC)")
                self.btn_record.setStyleSheet(
                    "background-color: #dc2626; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 12px;"
                )
                self.lbl_rec_status.setText(f"Recording: {path.name}")
                self.lbl_hud_rec.setText("REC: ON")
                self.lbl_hud_rec.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px;")
                self.statusMessage.emit(f"Hardware live recording started: {path.name}", False)
            else:
                QMessageBox.warning(
                    self, "Live Recording", "Connect to an active NDI stream before starting recording."
                )

    def _on_open_recordings_folder(self) -> None:
        rec_dir_str = self.line_rec_dir.text().strip()
        folder = Path(rec_dir_str) if rec_dir_str else OUTPUTS
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _on_pipeline_frame_ready(self, original: np.ndarray, enhanced: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frames = (original, enhanced)
        self.frameProduced.emit(original, enhanced)

    def _on_pipeline_telemetry(self, telem: PipelineTelemetry) -> None:
        self._signals.telemetryUpdated.emit(telem)

    def _render_viewport_tick(self) -> None:
        """Decoupled UI viewport monitor update (~30-40 FPS) to eliminate Qt event loop stalls."""
        frames = None
        with self._frame_lock:
            if self._latest_frames is not None:
                frames = self._latest_frames
                self._latest_frames = None

        if frames is None:
            return

        orig, enh = frames
        ho, wo = orig.shape[:2]
        stride_o = int(orig.strides[0])
        qimg_before = QImage(orig.data, wo, ho, stride_o, QImage.Format.Format_RGBA8888).copy()

        he, we = enh.shape[:2]
        stride_e = int(enh.strides[0])
        qimg_after = QImage(enh.data, we, he, stride_e, QImage.Format.Format_RGBA8888).copy()

        self.canvas.set_images(qimg_before, qimg_after)

    def _on_telemetry_gui(self, telem: PipelineTelemetry) -> None:
        wi, hi = telem.input_resolution
        self.lbl_hud_res.setText(f"Resolution: {wi}x{hi}")
        self.lbl_hud_fps.setText(f"In: {telem.input_fps:.1f} FPS | Out: {telem.render_fps:.1f} FPS")
        self.lbl_hud_latency.setText(f"Latency: {telem.latency_ms:.1f} ms")

        if hasattr(telem, "streamline") and telem.streamline.active:
            self.lbl_hud_pipeline.setText(f"Engine: {telem.streamline.engine_mode} ({telem.streamline.process_time_ms:.1f}ms)")
        else:
            self.lbl_hud_pipeline.setText("Engine: Streamline 2.13 + NIS + ReShade FX")

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
        self._viewport_timer.stop()
        self._source_timer.stop()
        self._cam_timer.stop()
        self._rec_timer.stop()
        self._pipeline.close()
