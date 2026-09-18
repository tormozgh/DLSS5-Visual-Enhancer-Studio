"""Real-Time Upscale Tab: NVIDIA NIS, Spatial Edge Reconstruction, Multi-Input (NDI, Webcam, Internal)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage
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
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.paths import OUTPUTS
from ...live.camera import CameraDeviceInfo, WebcamReceiver
from ...live.engine import PipelineTelemetry
from ...live.ndi import NdiSender
from ...live.recorder import LiveRecorder
from ...live.upscale_engine import RealtimeUpscalePipeline
from ...live.upscaler import NISConfig, RealtimeNISUpscaler
from ...settings.models import UISettings
from ..components.sliders import LabeledSlider
from ..components.split_canvas import CanvasViewMode, SplitCanvas


class _SignalBridge(QObject):
    frameReady = pyqtSignal(object, object)
    telemetryUpdated = pyqtSignal(object)


class RealtimeUpscaleTab(QWidget):
    """Dedicated Real-Time Upscaling & Directional Sharpening Studio Tab."""

    statusMessage = pyqtSignal(str, bool)

    def __init__(self, settings: UISettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._signals = _SignalBridge()
        self._signals.telemetryUpdated.connect(self._on_telemetry_gui)

        self._frame_lock = threading.Lock()
        self._latest_frames: tuple[np.ndarray, np.ndarray] | None = None
        self._last_raw_frame: np.ndarray | None = None

        # Dedicated pipeline decoupled from ReShade and Streamline
        self._pipeline = RealtimeUpscalePipeline(
            sender_name="DLSS 5 Real-Time Upscale Studio",
            on_frame_ready=self._on_pipeline_frame_ready,
            on_telemetry=self._on_pipeline_telemetry,
        )
        self.upscaler = self._pipeline.upscaler
        self.upscaler.config.enabled = True
        self.upscaler.config.scale_factor = 2.0
        self.upscaler.config.sharpness = 0.50
        self.upscaler.config.algorithm = "nis"

        # UI rendering timer (~30-40 FPS)
        self._viewport_timer = QTimer(self)
        self._viewport_timer.setInterval(28)
        self._viewport_timer.timeout.connect(self._render_viewport_tick)
        self._viewport_timer.start()

        # NDI discovery timer
        self._source_timer = QTimer(self)
        self._source_timer.setInterval(2500)
        self._source_timer.timeout.connect(self._refresh_sources_list)
        self._source_timer.start()

        # Camera auto-refresh timer (instant registry check)
        self._cam_timer = QTimer(self)
        self._cam_timer.setInterval(2000)
        self._cam_timer.timeout.connect(self._auto_refresh_cameras)
        self._cam_timer.start()

        # Recording timer
        self._rec_timer = QTimer(self)
        self._rec_timer.setInterval(500)
        self._rec_timer.timeout.connect(self._update_recorder_ui)
        self._rec_timer.start()

        self._init_ui()

    def _init_ui(self) -> None:
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ----------------------------------------------------------------------
        # RIGHT COLUMN: Controls Sidebar (Scrollable)
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

        # 1. Live Input Source Selection
        input_box = QGroupBox("Live Input Source")
        ib_layout = QVBoxLayout(input_box)
        ib_layout.setSpacing(8)

        type_row = QHBoxLayout()
        type_lbl = QLabel("Input Source:")
        type_lbl.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        type_row.addWidget(type_lbl)

        self.cmb_input_type = QComboBox()
        self.cmb_input_type.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_input_type.setMinimumContentsLength(8)
        self.cmb_input_type.addItem("Internal: Real-Time Rendering Output", "internal")
        self.cmb_input_type.addItem("NDI Network Stream", "ndi")
        self.cmb_input_type.addItem("Webcam / Capture Card", "webcam")
        self.cmb_input_type.currentIndexChanged.connect(self._on_input_type_changed)
        type_row.addWidget(self.cmb_input_type, 1)
        ib_layout.addLayout(type_row)

        # Internal feed notice
        self.widget_internal_info = QWidget()
        internal_layout = QVBoxLayout(self.widget_internal_info)
        internal_layout.setContentsMargins(0, 0, 0, 0)
        lbl_internal_desc = QLabel("Receiving live frames directly from Real-Time Rendering tab in-memory.")
        lbl_internal_desc.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: 500;")
        lbl_internal_desc.setWordWrap(True)
        internal_layout.addWidget(lbl_internal_desc)
        ib_layout.addWidget(self.widget_internal_info)

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
        self.widget_ndi_input.setVisible(False)
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

        self.btn_toggle_stream = QPushButton("Connect & Start Upscaling")
        self.btn_toggle_stream.setProperty("class", "primary")
        self.btn_toggle_stream.clicked.connect(self._on_toggle_stream)
        ib_layout.addWidget(self.btn_toggle_stream)

        self.lbl_stream_status = QLabel("Status: Standby (Click Connect to activate)")
        self.lbl_stream_status.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        ib_layout.addWidget(self.lbl_stream_status)

        sidebar_layout.addWidget(input_box)

        # 2. Real-Time Upscale Settings
        scale_box = QGroupBox("Real-Time Upscale Engine")
        sb_layout = QVBoxLayout(scale_box)
        sb_layout.setSpacing(8)

        self.chk_upscale_enabled = QCheckBox("Enable Real-Time Upscaling")
        self.chk_upscale_enabled.setChecked(True)
        self.chk_upscale_enabled.toggled.connect(self._on_upscale_config_changed)
        sb_layout.addWidget(self.chk_upscale_enabled)

        # Scaling Algorithm
        algo_row = QHBoxLayout()
        algo_lbl = QLabel("Algorithm:")
        algo_lbl.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        algo_row.addWidget(algo_lbl)

        self.cmb_algo = QComboBox()
        self.cmb_algo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_algo.setMinimumContentsLength(8)
        self.cmb_algo.addItem("NVIDIA DLSS 5 Neural AI (Tensor Cores)", "dlss5_neural")
        self.cmb_algo.addItem("NVIDIA NIS (Directional Edge Scaling)", "nis")
        self.cmb_algo.addItem("FidelityFX CAS Spatial Scaler", "cas")
        self.cmb_algo.addItem("Bicubic Catmull-Rom (Smooth)", "bicubic")
        self.cmb_algo.addItem("Bilinear Fast", "bilinear")
        self.cmb_algo.currentIndexChanged.connect(self._on_upscale_config_changed)
        algo_row.addWidget(self.cmb_algo, 1)
        sb_layout.addLayout(algo_row)

        # Target Preset
        target_row = QHBoxLayout()
        target_lbl = QLabel("Target Scale:")
        target_lbl.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        target_row.addWidget(target_lbl)

        self.cmb_target = QComboBox()
        self.cmb_target.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_target.setMinimumContentsLength(8)
        self.cmb_target.addItem("2.00x Ultra Quality (1080p -> 4K UHD)", ("scale", 2.0, (0, 0)))
        self.cmb_target.addItem("1.50x Quality (720p -> 1080p / 1080p -> 1620p)", ("scale", 1.5, (0, 0)))
        self.cmb_target.addItem("1.25x Balanced", ("scale", 1.25, (0, 0)))
        self.cmb_target.addItem("3.00x Extreme Scale", ("scale", 3.0, (0, 0)))
        self.cmb_target.addItem("4.00x Maximum Scale", ("scale", 4.0, (0, 0)))
        self.cmb_target.addItem("Target 1080p FHD (1920x1080)", ("fit", 1.0, (1920, 1080)))
        self.cmb_target.addItem("Target 1440p 2K (2560x1440)", ("fit", 1.0, (2560, 1440)))
        self.cmb_target.addItem("Target 4K UHD (3840x2160)", ("fit", 1.0, (3840, 2160)))
        self.cmb_target.addItem("Native Resolution + Sharpening Only", ("scale", 1.0, (0, 0)))
        self.cmb_target.currentIndexChanged.connect(self._on_upscale_config_changed)
        target_row.addWidget(self.cmb_target, 1)
        sb_layout.addLayout(target_row)

        sidebar_layout.addWidget(scale_box)

        # 3. Directional Sharpening & Anti-Ringing
        sharp_box = QGroupBox("Directional Sharpening & Anti-Ringing")
        sh_layout = QVBoxLayout(sharp_box)
        sh_layout.setSpacing(8)

        self.slider_sharpness = LabeledSlider("NIS Directional Sharpness", 0.0, 1.0, 0.50, step=0.05, decimals=2)
        self.slider_sharpness.valueChanged.connect(self._on_upscale_config_changed)
        sh_layout.addWidget(self.slider_sharpness)

        self.slider_anti_ringing = LabeledSlider("Anti-Ringing Clamping", 0.0, 1.0, 0.85, step=0.05, decimals=2)
        self.slider_anti_ringing.valueChanged.connect(self._on_upscale_config_changed)
        sh_layout.addWidget(self.slider_anti_ringing)

        lbl_sharp_tag = QLabel("Clamps high-frequency boost between local contrast bounds to eliminate halos.")
        lbl_sharp_tag.setStyleSheet("color: #7b8190; font-size: 10px;")
        lbl_sharp_tag.setWordWrap(True)
        sh_layout.addWidget(lbl_sharp_tag)

        sidebar_layout.addWidget(sharp_box)

        # 4. NDI Broadcast Output
        ndi_out_box = QGroupBox("NDI Broadcast Output")
        out_layout = QVBoxLayout(ndi_out_box)
        out_layout.setSpacing(6)

        self.chk_broadcast_enabled = QCheckBox("Enable NDI Out: DLSS 5 Real-Time Upscale Studio")
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

        # 5. Live Recording Configuration
        rec_box = QGroupBox("Live Recording Settings & Control")
        rec_layout = QVBoxLayout(rec_box)
        rec_layout.setSpacing(8)

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

        prefix_label = QLabel("File Name Prefix:")
        prefix_label.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        rec_layout.addWidget(prefix_label)

        self.line_rec_prefix = QLineEdit("DLSS5_Upscale")
        self.line_rec_prefix.setStyleSheet("background-color: #17191e; border: 1px solid #282b33; border-radius: 4px; padding: 4px 8px; color: #d0d4dc; font-size: 11px;")
        rec_layout.addWidget(self.line_rec_prefix)

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

        bitrate_col = QVBoxLayout()
        bitrate_lbl = QLabel("Bitrate:")
        bitrate_lbl.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        bitrate_col.addWidget(bitrate_lbl)
        self.cmb_rec_bitrate = QComboBox()
        self.cmb_rec_bitrate.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_rec_bitrate.setMinimumContentsLength(8)
        self.cmb_rec_bitrate.addItem("25 Mbps", 25)
        self.cmb_rec_bitrate.addItem("50 Mbps", 50)
        self.cmb_rec_bitrate.addItem("80 Mbps", 80)
        self.cmb_rec_bitrate.setCurrentIndex(1)
        bitrate_col.addWidget(self.cmb_rec_bitrate)
        fmt_res_row.addLayout(bitrate_col, 1)
        rec_layout.addLayout(fmt_res_row)

        self.btn_record = QPushButton("Start Live Recording")
        self.btn_record.setProperty("class", "record-btn")
        self.btn_record.setStyleSheet("background-color: #7f1d1d; color: #fecaca; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 12px;")
        self.btn_record.clicked.connect(self._on_toggle_record)
        rec_layout.addWidget(self.btn_record)

        self.lbl_rec_status = QLabel("Recorder: Idle")
        self.lbl_rec_status.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        rec_layout.addWidget(self.lbl_rec_status)

        sidebar_layout.addWidget(rec_box)
        sidebar_scroll.setWidget(sidebar)

        # ----------------------------------------------------------------------
        # LEFT COLUMN: Dual Split-View Canvas & Telemetry HUD
        # ----------------------------------------------------------------------
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(8)

        toolbar = QFrame()
        toolbar.setProperty("class", "toolbar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(8)

        btn_split = QPushButton("Split Slider")
        btn_split.setProperty("class", "toolbar-btn")
        btn_split.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.SPLIT))
        toolbar_layout.addWidget(btn_split)

        btn_side = QPushButton("Side-by-Side")
        btn_side.setProperty("class", "toolbar-btn")
        btn_side.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.SIDE_BY_SIDE))
        toolbar_layout.addWidget(btn_side)

        btn_after = QPushButton("Upscaled Only")
        btn_after.setProperty("class", "toolbar-btn")
        btn_after.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.ONLY_AFTER))
        toolbar_layout.addWidget(btn_after)

        btn_before = QPushButton("Original Only")
        btn_before.setProperty("class", "toolbar-btn")
        btn_before.clicked.connect(lambda: self.canvas.set_view_mode(CanvasViewMode.ONLY_BEFORE))
        toolbar_layout.addWidget(btn_before)

        toolbar_layout.addStretch()

        btn_fit = QPushButton("Fit")
        btn_fit.setProperty("class", "toolbar-btn")
        btn_fit.clicked.connect(lambda: self.canvas.fit_to_view())
        toolbar_layout.addWidget(btn_fit)

        viewport_layout.addWidget(toolbar)

        self.canvas = SplitCanvas()
        viewport_layout.addWidget(self.canvas, 1)

        hud_frame = QFrame()
        hud_frame.setProperty("class", "studio-card")
        hud_frame.setStyleSheet("background-color: #14161a; border: 1px solid #22252c; border-radius: 4px; padding: 6px;")
        hud_layout = QHBoxLayout(hud_frame)
        hud_layout.setContentsMargins(10, 4, 10, 4)
        hud_layout.setSpacing(16)

        self.lbl_hud_res = QLabel("Resolution: --")
        self.lbl_hud_res.setStyleSheet("color: #e2e8f0; font-weight: 600; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_res)

        self.lbl_hud_fps = QLabel("FPS: --")
        self.lbl_hud_fps.setStyleSheet("color: #38bdf8; font-weight: 600; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_fps)

        self.lbl_hud_latency = QLabel("Latency: -- ms")
        self.lbl_hud_latency.setStyleSheet("color: #4ade80; font-weight: 600; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_latency)

        self.lbl_hud_scale = QLabel("Scale: 2.0x")
        self.lbl_hud_scale.setStyleSheet("color: #a78bfa; font-weight: 600; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_scale)

        self.lbl_hud_algo = QLabel("Engine: NVIDIA DLSS 5 Neural AI")
        self.lbl_hud_algo.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_algo)

        hud_layout.addStretch()

        self.lbl_hud_rec = QLabel("REC: OFF")
        self.lbl_hud_rec.setStyleSheet("color: #6b7280; font-weight: bold; font-size: 11px;")
        hud_layout.addWidget(self.lbl_hud_rec)

        viewport_layout.addWidget(hud_frame)

        # Order in Splitter: Canvas on LEFT, Sidebar on RIGHT
        splitter.addWidget(viewport_container)
        splitter.addWidget(sidebar_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([850, 420])
        splitter.setHandleWidth(6)

        root_layout.addWidget(splitter)

    # --------------------------------------------------------------------------
    # Inter-Tab Frame Feed & Handlers
    # --------------------------------------------------------------------------
    def feed_internal_frame(self, original_rgba: np.ndarray, enhanced_rgba: np.ndarray) -> None:
        """Receive rendered frames from Real-Time Rendering tab in-memory."""
        if not self._pipeline.is_running or self.cmb_input_type.currentData() != "internal":
            return
        self._pipeline.feed_internal_frame(original_rgba, enhanced_rgba)

    def _on_input_type_changed(self) -> None:
        mode = self.cmb_input_type.currentData()
        self.widget_internal_info.setVisible(mode == "internal")
        self.widget_ndi_input.setVisible(mode == "ndi")
        self.widget_webcam_input.setVisible(mode == "webcam")
        if mode == "webcam":
            self._refresh_cameras_list()

    def _refresh_sources_list(self) -> None:
        sources = self._pipeline.get_sources()
        current = self.cmb_sources.currentData()
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
                    if url == current:
                        self.cmb_sources.setCurrentIndex(self.cmb_sources.count() - 1)
            self.cmb_sources.blockSignals(False)

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
        """Periodic background refresh to detect newly plugged webcams instantly."""
        if self._pipeline.is_running:
            return
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
            self.btn_toggle_stream.setText("Connect & Start Upscaling")
            self.btn_toggle_stream.setStyleSheet("")
            self.lbl_stream_status.setText("Status: Stopped")
            self.statusMessage.emit("Real-time upscaling stopped", False)
        else:
            mode = self.cmb_input_type.currentData()
            enable_out = self.chk_broadcast_enabled.isChecked()

            if mode == "internal":
                self._pipeline.start_internal_pipeline(enable_ndi_out=enable_out)
                self.btn_toggle_stream.setText("Stop Upscaling")
                self.btn_toggle_stream.setStyleSheet("background-color: #991b1b; color: white;")
                self.lbl_stream_status.setText("Status: Active (Receiving from Real-Time Rendering)")
                self.statusMessage.emit("Connected to Real-Time Rendering feed", False)
            elif mode == "webcam":
                dev_idx = self.cmb_cameras.currentData()
                if dev_idx is None:
                    QMessageBox.warning(self, "No Camera Selected", "Please select an active camera device.")
                    return

                cams = self._pipeline.get_cameras()
                cam_info = next((c for c in cams if c.index == dev_idx), None)
                w = cam_info.width if cam_info else 1280
                h = cam_info.height if cam_info else 720
                fps = cam_info.fps if cam_info else 30.0
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
                    self.btn_toggle_stream.setText("Stop Upscaling")
                    self.btn_toggle_stream.setStyleSheet("background-color: #991b1b; color: white;")
                    self.lbl_stream_status.setText(f"Status: Streaming from {cam_name}")
                    self.statusMessage.emit(f"Connected to camera: {cam_name}", False)
                except Exception as exc:
                    QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {exc}")
            else:
                url = self.cmb_sources.currentData()
                name = self.cmb_sources.currentText()
                if not name or "No NDI sources" in name or "Searching" in name:
                    QMessageBox.warning(self, "No NDI Source Selected", "Please select an active NDI source.")
                    return
                try:
                    self._pipeline.start_ndi_pipeline(source_name=name, url_address=url, enable_ndi_out=enable_out)
                    self.btn_toggle_stream.setText("Stop Upscaling")
                    self.btn_toggle_stream.setStyleSheet("background-color: #991b1b; color: white;")
                    self.lbl_stream_status.setText(f"Status: Streaming from {name}")
                    self.statusMessage.emit(f"Connected to NDI source: {name}", False)
                except Exception as exc:
                    QMessageBox.critical(self, "Connection Error", f"Failed to start NDI: {exc}")

    def _on_upscale_config_changed(self) -> None:
        cfg = self.upscaler.config
        cfg.enabled = self.chk_upscale_enabled.isChecked()
        cfg.algorithm = self.cmb_algo.currentData() or "nis"
        algo_name = self.cmb_algo.currentText()
        if hasattr(self, "lbl_hud_algo"):
            self.lbl_hud_algo.setText(f"Engine: {algo_name}")

        mode_data = self.cmb_target.currentData()
        if mode_data:
            mode_type, scale_val, target_res = mode_data
            cfg.scale_mode = mode_type
            cfg.scale_factor = scale_val
            cfg.target_resolution = target_res
            self.lbl_hud_scale.setText(f"Scale: {scale_val}x" if mode_type == "scale" else f"{target_res[0]}x{target_res[1]}")

        cfg.sharpness = float(self.slider_sharpness.value())
        cfg.edge_contrast_limit = float(self.slider_anti_ringing.value())

        # Instant reprocess for paused frames
        self._pipeline.reprocess_last_frame()
        self._render_viewport_tick()

    def _on_browse_rec_folder(self) -> None:
        current = self.line_rec_dir.text().strip() or str(OUTPUTS)
        chosen = QFileDialog.getExistingDirectory(self, "Select Recording Directory", current)
        if chosen:
            self.line_rec_dir.setText(chosen)

    def _on_open_recordings_folder(self) -> None:
        folder = Path(self.line_rec_dir.text().strip()) if self.line_rec_dir.text().strip() else OUTPUTS
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    def _on_toggle_record(self) -> None:
        if self._pipeline.recorder.is_recording:
            saved = self._pipeline.stop_recording()
            self.btn_record.setText("Start Live Recording")
            self.btn_record.setStyleSheet("background-color: #7f1d1d; color: #fecaca; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 12px;")
            self.lbl_rec_status.setText("Recorder: Idle")
            self.lbl_hud_rec.setText("REC: OFF")
            self.lbl_hud_rec.setStyleSheet("color: #6b7280; font-weight: bold; font-size: 11px;")
            if saved and saved.is_file():
                self.statusMessage.emit(f"Recording saved: {saved.name}", False)
        else:
            bitrate = self.cmb_rec_bitrate.currentData() or 25
            fmt = self.cmb_rec_format.currentData() or "mp4"
            rec_dir = Path(self.line_rec_dir.text().strip()) if self.line_rec_dir.text().strip() else OUTPUTS
            prefix = self.line_rec_prefix.text().strip() or "DLSS5_Upscale"

            path = self._pipeline.start_recording(
                bitrate_mbps=bitrate,
                format_ext=fmt,
                output_dir=rec_dir,
                filename_prefix=prefix,
            )
            if path:
                self.btn_record.setText("Stop Recording (REC)")
                self.btn_record.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 10px; border-radius: 4px; font-size: 12px;")
                self.lbl_rec_status.setText(f"Recording: {path.name}")
                self.lbl_hud_rec.setText("REC: ON")
                self.lbl_hud_rec.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 11px;")
                self.statusMessage.emit(f"Hardware live recording started: {path.name}", False)
            else:
                QMessageBox.warning(self, "Live Recording", "Connect to an active input stream before starting recording.")

    def _on_pipeline_frame_ready(self, original: np.ndarray, enhanced: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frames = (original, enhanced)

    def _on_pipeline_telemetry(self, telem: PipelineTelemetry) -> None:
        self._signals.telemetryUpdated.emit(telem)

    def _render_viewport_tick(self) -> None:
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
        wo, ho = telem.output_resolution
        if (wo, ho) != (wi, hi) and (wo > 0 and ho > 0):
            self.lbl_hud_res.setText(f"In: {wi}x{hi} | Out: {wo}x{ho}")
        else:
            self.lbl_hud_res.setText(f"Resolution: {wi}x{hi}")
        self.lbl_hud_fps.setText(f"In: {telem.input_fps:.1f} FPS | Out: {telem.render_fps:.1f} FPS")
        self.lbl_hud_latency.setText(f"Latency: {telem.latency_ms:.1f} ms")

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
        self._viewport_timer.stop()
        self._source_timer.stop()
        self._cam_timer.stop()
        self._rec_timer.stop()
        self._pipeline.close()
