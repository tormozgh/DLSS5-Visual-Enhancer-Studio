"""Timeline Studio Tab: Premiere-style NLE multi-track editor, Media Pool, and DLSS 5 / ReShade FX Adjustment Layer workflow."""

from __future__ import annotations

import os
from typing import Callable

from PyQt6.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...settings.models import UISettings
from ...timeline.compositor import TimelineCompositor
from ...timeline.export_worker import TimelineExportWorker
from ...timeline.frame_cache import VideoFrameCache
from ...timeline.models import MediaAsset, SequenceSettings, TimelineClip, TimelineProject
from ..components.timeline.inspector import ClipInspectorWidget
from ..components.timeline.media_pool import MediaPoolWidget
from ..components.timeline.monitor import TimelineMonitorWidget
from ..components.timeline.time_ruler import TimeRulerWidget
from ..components.timeline.track_canvas import MultiTrackCanvas


class SequenceSettingsDialog(QDialog):
    """Modal dialog for creating a new sequence or adjusting active sequence settings."""

    def __init__(
        self,
        current_settings: SequenceSettings | None = None,
        is_new: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Sequence" if is_new else "Sequence Settings")
        self.resize(460, 340)
        self.settings = current_settings or SequenceSettings()
        self.is_new = is_new

        self.setStyleSheet(
            "QDialog { background-color: #121316; color: #f8fafc; }"
            "QLabel { color: #cbd5e1; font-size: 11px; }"
            "QLineEdit, QComboBox, QSpinBox {"
            "  background-color: #1a1c22; color: #f8fafc; border: 1px solid #2d313b; border-radius: 4px; padding: 4px 6px; font-size: 11px;"
            "}"
            "QPushButton { background: #2563eb; color: white; border-radius: 4px; padding: 6px 14px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: #1d4ed8; }"
        )

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        hdr = QLabel("New Video Sequence Setup" if self.is_new else "Sequence Settings")
        hdr.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(hdr)

        form = QFormLayout()
        form.setSpacing(8)

        # Sequence Name
        self.txt_name = QLineEdit(self.settings.name)
        form.addRow("Sequence Name:", self.txt_name)

        # Presets
        self.cmb_preset = QComboBox()
        self.presets = [
            ("1080p Full HD (1920x1080 16:9)", 1920, 1080, 30.0),
            ("4K UHD (3840x2160 16:9)", 3840, 2160, 30.0),
            ("2K QHD (2560x1440 16:9)", 2560, 1440, 30.0),
            ("Vertical Social (1080x1920 9:16)", 1080, 1920, 30.0),
            ("720p HD (1280x720 16:9)", 1280, 720, 30.0),
            ("8K Ultra HD (7680x4320 16:9)", 7680, 4320, 30.0),
            ("Custom Dimensions", 0, 0, 0),
        ]
        for name, _, _, _ in self.presets:
            self.cmb_preset.addItem(name)
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        form.addRow("Preset:", self.cmb_preset)

        # Resolution spinboxes
        res_row = QHBoxLayout()
        self.spn_w = QSpinBox()
        self.spn_w.setRange(256, 16384)
        self.spn_w.setSingleStep(2)
        self.spn_w.setValue(self.settings.width)

        self.spn_h = QSpinBox()
        self.spn_h.setRange(256, 16384)
        self.spn_h.setSingleStep(2)
        self.spn_h.setValue(self.settings.height)

        res_row.addWidget(QLabel("Width:"))
        res_row.addWidget(self.spn_w, 1)
        res_row.addWidget(QLabel("Height:"))
        res_row.addWidget(self.spn_h, 1)
        form.addRow("Frame Size:", res_row)

        # Timebase / Frame Rate
        self.cmb_fps = QComboBox()
        self.fps_options = [
            ("23.976 fps (Cinema 24p)", 23.976),
            ("24.0 fps (Standard Film)", 24.0),
            ("25.0 fps (PAL Broadcast)", 25.0),
            ("29.97 fps (NTSC Broadcast)", 29.97),
            ("30.0 fps (Standard Web)", 30.0),
            ("50.0 fps (PAL High Speed)", 50.0),
            ("59.94 fps (NTSC 60p Broadcast)", 59.94),
            ("60.0 fps (Gaming / High 60p)", 60.0),
            ("120.0 fps (Ultra High Speed)", 120.0),
        ]
        cur_fps = self.settings.fps
        matched_fps_idx = 4
        for i, (label, val) in enumerate(self.fps_options):
            self.cmb_fps.addItem(label, val)
            if abs(val - cur_fps) < 0.05:
                matched_fps_idx = i
        self.cmb_fps.setCurrentIndex(matched_fps_idx)
        form.addRow("Timebase (FPS):", self.cmb_fps)

        lbl_pixel = QLabel("Square Pixels (1.0)")
        lbl_pixel.setStyleSheet("color: #94a3b8;")
        form.addRow("Pixel Aspect:", lbl_pixel)

        layout.addLayout(form)
        layout.addStretch()

        # Dialog Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background: #334155; color: white;")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_ok = QPushButton("Create Sequence" if self.is_new else "Apply Settings")
        btn_ok.clicked.connect(self.accept)
        btn_row.addWidget(btn_ok)
        layout.addLayout(btn_row)

        # Match initial preset
        for i, (p_name, pw, ph, _) in enumerate(self.presets):
            if pw == self.settings.width and ph == self.settings.height:
                self.cmb_preset.setCurrentIndex(i)
                break

    def _on_preset_changed(self, idx: int) -> None:
        if idx < len(self.presets) - 1:
            _, w, h, _ = self.presets[idx]
            self.spn_w.setValue(w)
            self.spn_h.setValue(h)

    def get_settings(self) -> SequenceSettings:
        fps_val = float(self.cmb_fps.currentData() or 30.0)
        return SequenceSettings(
            name=self.txt_name.text().strip() or "Sequence 01",
            width=self.spn_w.value(),
            height=self.spn_h.value(),
            fps=fps_val,
            preset_name=self.cmb_preset.currentText(),
        )


class TimelineExportDialog(QDialog):
    """Export modal dialog for configuring output range, resolution, and encoding format."""

    def __init__(self, project: TimelineProject, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Timeline Video")
        self.resize(460, 360)
        self.project = project

        self.setStyleSheet(
            "QDialog { background-color: #121316; color: #f8fafc; }"
            "QLabel { color: #cbd5e1; font-size: 11px; }"
            "QComboBox, QSpinBox, QLineEdit {"
            "  background-color: #1a1c22; color: #f8fafc; border: 1px solid #2d313b; border-radius: 4px; padding: 4px; font-size: 11px;"
            "}"
            "QPushButton { background: #2563eb; color: white; border-radius: 4px; padding: 6px 14px; font-weight: 600; font-size: 11px; }"
            "QPushButton:hover { background: #1d4ed8; }"
        )

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        lbl_hdr = QLabel("Export DLSS 5 Timeline")
        lbl_hdr.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(lbl_hdr)

        form = QFormLayout()
        form.setSpacing(10)

        # Export Range
        self.cmb_range = QComboBox()
        self.cmb_range.addItems(["Entire Timeline", "In to Out Work Area"])
        form.addRow("Export Range:", self.cmb_range)

        # Target Resolution
        self.cmb_res = QComboBox()
        self.cmb_res.addItems([
            "3840x2160 (4K UHD)",
            "2560x1440 (2K QHD)",
            "1920x1080 (Full HD)",
            "7680x4320 (8K)",
        ])
        form.addRow("Output Resolution:", self.cmb_res)

        # Encoder Codec
        self.cmb_codec = QComboBox()
        self.cmb_codec.addItems([
            "H.264 (NVENC GPU Accelerated)",
            "H.265 / HEVC (NVENC 10-bit)",
            "Apple ProRes 422 HQ",
            "H.264 (CPU libx264)",
        ])
        form.addRow("Video Codec:", self.cmb_codec)

        # Target Bitrate
        self.spn_bitrate = QSpinBox()
        self.spn_bitrate.setRange(5, 200)
        self.spn_bitrate.setValue(35)
        self.spn_bitrate.setSuffix(" Mbps")
        form.addRow("Target Bitrate:", self.spn_bitrate)

        layout.addLayout(form)

        # Output Path Selector
        row_path = QHBoxLayout()
        self.lbl_path = QLabel("Select destination file...")
        self.lbl_path.setStyleSheet("color: #94a3b8; font-family: monospace; font-size: 10px;")
        btn_browse = QPushButton("Browse...")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._on_browse)
        row_path.addWidget(self.lbl_path, 1)
        row_path.addWidget(btn_browse)
        layout.addLayout(row_path)

        # Action Buttons
        row_btn = QHBoxLayout()
        row_btn.addStretch()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background: #334155;")
        btn_cancel.clicked.connect(self.reject)
        row_btn.addWidget(btn_cancel)

        self.btn_start = QPushButton("Start Render & Export")
        self.btn_start.clicked.connect(self._on_accept)
        row_btn.addWidget(self.btn_start)
        layout.addLayout(row_btn)

        self.selected_output_path = ""

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Output Video",
            os.path.expanduser("~/Videos/DLSS5_Timeline_Export.mp4"),
            "MP4 Video (*.mp4);;MOV Video (*.mov);;All Files (*.*)",
        )
        if path:
            self.selected_output_path = path
            self.lbl_path.setText(os.path.basename(path))

    def _on_accept(self) -> None:
        if not self.selected_output_path:
            QMessageBox.warning(self, "No Destination", "Please select an output destination file first.")
            return
        self.accept()

    def get_export_config(self) -> dict:
        res_text = self.cmb_res.currentText()
        if "3840x2160" in res_text:
            res = (3840, 2160)
        elif "2560x1440" in res_text:
            res = (2560, 1440)
        elif "7680x4320" in res_text:
            res = (7680, 4320)
        else:
            res = (1920, 1080)

        codec_text = self.cmb_codec.currentText()
        if "HEVC" in codec_text:
            codec = "hevc_nvenc"
        elif "ProRes" in codec_text:
            codec = "prores"
        elif "CPU" in codec_text:
            codec = "libx264"
        else:
            codec = "h264_nvenc"

        range_mode = "in_out" if "In to Out" in self.cmb_range.currentText() else "entire"

        return {
            "output_path": self.selected_output_path,
            "resolution": res,
            "codec": codec,
            "bitrate": float(self.spn_bitrate.value()),
            "range_mode": range_mode,
        }


class TimelineStudioTab(QWidget):
    """Main Timeline Studio Tab implementing a Premiere Pro workflow for DLSS 5 enhancement."""

    statusMessage = pyqtSignal(str, bool)

    def __init__(self, settings: UISettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings

        # Core State
        self._sequence_initialized: bool = False
        self.project = TimelineProject.create_default(fps=30.0, width=1920, height=1080, sequence_name="Sequence 01")
        self.frame_cache = VideoFrameCache()
        self.compositor = TimelineCompositor(self.project, self.frame_cache)
        self._current_split_ratio: float | None = None
        self._export_worker: TimelineExportWorker | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.stack = QStackedWidget(self)

        # ---------------- PAGE 0: SPLASH / NO SEQUENCE PROMPT ----------------
        self.splash_page = QWidget()
        splash_layout = QVBoxLayout(self.splash_page)
        splash_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setFixedSize(540, 320)
        card.setStyleSheet(
            "QFrame {"
            "  background-color: #16181f;"
            "  border: 1px solid #2d313b;"
            "  border-radius: 12px;"
            "  padding: 24px;"
            "}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(14)
        card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lbl_icon = QLabel("Timeline Studio")
        lbl_icon.setStyleSheet("color: #38bdf8; font-size: 20px; font-weight: 800;")
        lbl_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(lbl_icon)

        lbl_title = QLabel("No Active Sequence")
        lbl_title.setStyleSheet("color: #f8fafc; font-size: 15px; font-weight: 700;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Before placing media clips or applying DLSS 5 and ReShade FX layers,\n"
            "please create a sequence to define your resolution, aspect ratio, and frame rate."
        )
        lbl_desc.setStyleSheet("color: #94a3b8; font-size: 12px; line-height: 1.4;")
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_desc.setWordWrap(True)
        card_layout.addWidget(lbl_desc)

        card_layout.addSpacing(10)

        btn_create = QPushButton("Create Sequence to Begin Editing")
        btn_create.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_create.setStyleSheet(
            "QPushButton {"
            "  background: #2563eb;"
            "  color: white;"
            "  border-radius: 6px;"
            "  padding: 10px 22px;"
            "  font-size: 13px;"
            "  font-weight: 700;"
            "}"
            "QPushButton:hover { background: #1d4ed8; }"
        )
        btn_create.clicked.connect(self.prompt_initial_sequence)
        card_layout.addWidget(btn_create, 0, Qt.AlignmentFlag.AlignCenter)

        splash_layout.addWidget(card)
        self.stack.addWidget(self.splash_page)

        # ---------------- PAGE 1: WORKSPACE ----------------
        vert_splitter = QSplitter(Qt.Orientation.Vertical)
        vert_splitter.setHandleWidth(4)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(4)

        # 1. Left: Media Pool
        self.media_pool = MediaPoolWidget(self.frame_cache, self)
        self.media_pool.assetDoubleClicked.connect(self._on_asset_double_clicked)
        self.media_pool.addAdjustmentLayerRequested.connect(self.add_fx_layer)
        top_splitter.addWidget(self.media_pool)

        # 2. Center: Program Monitor Viewport
        self.monitor = TimelineMonitorWidget(fps=self.project.fps, parent=self)
        self.monitor.seekRequested.connect(self.seek_frame)
        self.monitor.markInRequested.connect(self.mark_in)
        self.monitor.markOutRequested.connect(self.mark_out)
        self.monitor.splitRatioChanged.connect(self._on_split_ratio_changed)
        top_splitter.addWidget(self.monitor)

        # 3. Right: Clip Inspector
        self.inspector = ClipInspectorWidget(self)
        self.inspector.parametersChanged.connect(self._on_clip_parameters_changed)
        top_splitter.addWidget(self.inspector)

        top_splitter.setSizes([260, 680, 320])
        vert_splitter.addWidget(top_splitter)

        # ---------------- BOTTOM TIMELINE SECTION ----------------
        bottom_container = QWidget()
        bottom_layout = QVBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        # 1. Sequence Info & Setup Bar (Premiere-style)
        seq_bar = QFrame()
        seq_bar.setStyleSheet("background-color: #0f1013; border-top: 1px solid #23252a; border-bottom: 1px solid #1a1c22; padding: 3px 8px;")
        sb_layout = QHBoxLayout(seq_bar)
        sb_layout.setContentsMargins(6, 2, 6, 2)
        sb_layout.setSpacing(8)

        self.lbl_seq_badge = QLabel("Sequence: Sequence 01 [1920x1080 @ 30.00 fps]")
        self.lbl_seq_badge.setStyleSheet("color: #38bdf8; font-weight: 700; font-size: 11px;")
        sb_layout.addWidget(self.lbl_seq_badge)

        sb_layout.addStretch()

        self.btn_seq_settings = QPushButton("Sequence Settings...")
        self.btn_seq_settings.setStyleSheet("background: #1e293b; color: #cbd5e1; border-radius: 4px; padding: 3px 8px; font-size: 11px; font-weight: 600;")
        self.btn_seq_settings.clicked.connect(self.open_sequence_settings)
        sb_layout.addWidget(self.btn_seq_settings)

        self.btn_new_seq = QPushButton("New Sequence...")
        self.btn_new_seq.setStyleSheet("background: #1e293b; color: #cbd5e1; border-radius: 4px; padding: 3px 8px; font-size: 11px; font-weight: 600;")
        self.btn_new_seq.clicked.connect(self.create_new_sequence)
        sb_layout.addWidget(self.btn_new_seq)

        bottom_layout.addWidget(seq_bar)

        # 2. Timeline Tools Bar
        tools_bar = QFrame()
        tools_bar.setStyleSheet("background-color: #14161b; border-bottom: 1px solid #23252a; padding: 4px 8px;")
        tb_layout = QHBoxLayout(tools_bar)
        tb_layout.setContentsMargins(4, 2, 4, 2)
        tb_layout.setSpacing(8)

        # Action Buttons
        self.btn_split_clip = QPushButton("Split Clip (C)")
        self.btn_split_clip.setStyleSheet("background: #1e293b; color: #f8fafc; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        self.btn_split_clip.clicked.connect(self._on_split_tool_clicked)
        tb_layout.addWidget(self.btn_split_clip)

        self.btn_delete_clip = QPushButton("Delete Clip (Del)")
        self.btn_delete_clip.setStyleSheet("background: #1e293b; color: #f8fafc; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        self.btn_delete_clip.clicked.connect(self._on_delete_tool_clicked)
        tb_layout.addWidget(self.btn_delete_clip)

        # FX Layer Buttons
        self.btn_add_dlss_fx = QPushButton("+ Add DLSS 5 Layer")
        self.btn_add_dlss_fx.setStyleSheet("background: #7c3aed; color: white; border-radius: 4px; padding: 4px 8px; font-weight: 600; font-size: 11px;")
        self.btn_add_dlss_fx.clicked.connect(lambda: self.add_fx_layer("dlss5"))
        tb_layout.addWidget(self.btn_add_dlss_fx)

        self.btn_add_reshade_fx = QPushButton("+ Add ReShade Layer")
        self.btn_add_reshade_fx.setStyleSheet("background: #0284c7; color: white; border-radius: 4px; padding: 4px 8px; font-weight: 600; font-size: 11px;")
        self.btn_add_reshade_fx.clicked.connect(lambda: self.add_fx_layer("reshade"))
        tb_layout.addWidget(self.btn_add_reshade_fx)

        self.btn_add_track = QPushButton("+ Add Track")
        self.btn_add_track.setStyleSheet("background: #1e293b; color: #38bdf8; border-radius: 4px; padding: 4px 8px; font-weight: 600; font-size: 11px;")
        self.btn_add_track.clicked.connect(self.add_video_track)
        tb_layout.addWidget(self.btn_add_track)

        self.btn_delete_track = QPushButton("- Delete Track")
        self.btn_delete_track.setStyleSheet("background: #1e293b; color: #f87171; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
        self.btn_delete_track.clicked.connect(self.delete_top_track)
        tb_layout.addWidget(self.btn_delete_track)

        tb_layout.addStretch()

        # Timeline Zoom Slider
        lbl_zoom = QLabel("Zoom:")
        lbl_zoom.setStyleSheet("color: #94a3b8; font-size: 11px;")
        tb_layout.addWidget(lbl_zoom)

        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(2, 40)
        self.slider_zoom.setValue(20)
        self.slider_zoom.setFixedWidth(110)
        self.slider_zoom.valueChanged.connect(self._on_zoom_changed)
        tb_layout.addWidget(self.slider_zoom)

        tb_layout.addSpacing(16)

        # Export Button
        self.btn_export = QPushButton("Export Timeline...")
        self.btn_export.setStyleSheet("background: #16a34a; color: white; font-weight: 700; border-radius: 4px; padding: 5px 14px; font-size: 11px;")
        self.btn_export.clicked.connect(self._on_export_clicked)
        tb_layout.addWidget(self.btn_export)

        bottom_layout.addWidget(tools_bar)

        # Synchronized Scroll Container for Ruler & Multi-Track Canvas
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.timeline_scroll.setStyleSheet("QScrollArea { background-color: #121316; border: none; }")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)

        # Time Ruler
        self.ruler = TimeRulerWidget(fps=self.project.fps, total_frames=self.project.get_total_frames(), pixels_per_frame=2.0, parent=self)
        self.ruler.seekRequested.connect(self.seek_frame)
        self.ruler.inPointChanged.connect(self._on_ruler_in_changed)
        self.ruler.outPointChanged.connect(self._on_ruler_out_changed)
        scroll_layout.addWidget(self.ruler)

        # Multi-Track Canvas
        self.canvas = MultiTrackCanvas(self.project, pixels_per_frame=2.0, parent=self)
        self.canvas.clipSelected.connect(self.inspector.inspect_clip)
        self.canvas.projectModified.connect(self._on_project_modified)
        self.canvas.seekRequested.connect(self.seek_frame)
        self.canvas.mediaDropped.connect(self._on_media_dropped)
        scroll_layout.addWidget(self.canvas)
        scroll_layout.addStretch()

        self.timeline_scroll.setWidget(scroll_widget)
        bottom_layout.addWidget(self.timeline_scroll, 1)

        vert_splitter.addWidget(bottom_container)
        vert_splitter.setSizes([460, 360])

        self.stack.addWidget(vert_splitter)
        self.stack.setCurrentIndex(0)
        main_layout.addWidget(self.stack)

        # Render initial empty frame
        self.seek_frame(0)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._sequence_initialized:
            QTimer.singleShot(50, self.prompt_initial_sequence)

    def prompt_initial_sequence(self) -> None:
        """Prompt user with the New Sequence setup dialog before activating the timeline workspace."""
        diag = SequenceSettingsDialog(None, is_new=True, parent=self)
        if diag.exec() == QDialog.DialogCode.Accepted:
            new_seq = diag.get_settings()
            self.project = TimelineProject.create_default(
                fps=new_seq.fps,
                width=new_seq.width,
                height=new_seq.height,
                sequence_name=new_seq.name,
            )
            self.compositor.project = self.project
            self.canvas.project = self.project
            self.canvas.selected_clip_id = None
            self.inspector.inspect_clip(None)
            self._sequence_initialized = True
            self.stack.setCurrentIndex(1)
            self._apply_sequence_update()
            self.statusMessage.emit(f"Created sequence: {new_seq.name} [{new_seq.width}x{new_seq.height} @ {new_seq.fps:.2f} fps]", False)
        else:
            self._sequence_initialized = False
            self.stack.setCurrentIndex(0)

    def closeEvent(self, event) -> None:
        if hasattr(self, "compositor") and hasattr(self.compositor, "processor"):
            self.compositor.processor.close()
        super().closeEvent(event)

    def open_sequence_settings(self) -> None:
        """Open settings dialog to adjust the current sequence resolution, framerate, and name."""
        diag = SequenceSettingsDialog(self.project.sequence, is_new=False, parent=self)
        if diag.exec() == QDialog.DialogCode.Accepted:
            new_seq = diag.get_settings()
            self.project.update_sequence(new_seq)
            self._apply_sequence_update()
            self.statusMessage.emit(f"Updated sequence: {new_seq.name} ({new_seq.width}x{new_seq.height} @ {new_seq.fps:.2f} fps)", False)

    def create_new_sequence(self) -> None:
        """Create a fresh sequence with clear project dimensions and empty tracks."""
        diag = SequenceSettingsDialog(None, is_new=True, parent=self)
        if diag.exec() == QDialog.DialogCode.Accepted:
            new_seq = diag.get_settings()
            self.project = TimelineProject.create_default(
                fps=new_seq.fps,
                width=new_seq.width,
                height=new_seq.height,
                sequence_name=new_seq.name,
            )
            self.compositor.project = self.project
            self.canvas.project = self.project
            self.canvas.selected_clip_id = None
            self.inspector.inspect_clip(None)
            self._sequence_initialized = True
            self.stack.setCurrentIndex(1)
            self._apply_sequence_update()
            self.statusMessage.emit(f"Created new sequence: {new_seq.name}", False)

    def _apply_sequence_update(self) -> None:
        seq = self.project.sequence
        self.lbl_seq_badge.setText(f"Sequence: {seq.name} [{seq.width}x{seq.height} @ {seq.fps:.2f} fps]")
        self.ruler.fps = seq.fps
        self.monitor.fps = seq.fps
        self.canvas._update_min_size()
        self.canvas.update()
        self.seek_frame(self.project.playhead_frame)

    def _on_media_dropped(self, asset_id: str, track_id: int, drop_frame: int) -> None:
        """Callback from track canvas when an asset is dragged and dropped from MediaPool."""
        asset = self.media_pool.assets.get(asset_id)
        if not asset:
            return

        target_track = next((t for t in self.project.video_tracks if t.track_id == track_id), None)
        if not target_track:
            target_track = self.project.video_tracks[0] if self.project.video_tracks else None

        if target_track:
            clip = TimelineClip.create_media_clip(
                asset=asset,
                track_id=target_track.track_id,
                timeline_in=drop_frame,
                timeline_fps=self.project.fps,
            )
            target_track.add_clip(clip)
            self.canvas.selected_clip_id = clip.clip_id
            self.inspector.inspect_clip(clip)
            self._on_project_modified()
            self.statusMessage.emit(f"Added {asset.name} to track {target_track.name}", False)

    def _on_asset_double_clicked(self, asset: MediaAsset) -> None:
        """Double clicking a media asset appends it to track V1 at current playhead."""
        v1 = next((t for t in self.project.video_tracks if t.track_id == 1), None)
        if not v1:
            v1 = self.project.video_tracks[-1]

        clip = TimelineClip.create_media_clip(
            asset=asset,
            track_id=v1.track_id,
            timeline_in=self.project.playhead_frame,
            timeline_fps=self.project.fps,
        )
        v1.add_clip(clip)
        self.canvas.selected_clip_id = clip.clip_id
        self.inspector.inspect_clip(clip)
        self._on_project_modified()
        self.statusMessage.emit(f"Placed {asset.name} on track {v1.name}", False)

    def add_fx_layer(self, fx_type: str = "dlss5") -> None:
        """Create a new FX Adjustment Layer (DLSS 5 or ReShade) on track V2 (or topmost video track)."""
        v_track = next((t for t in self.project.video_tracks if t.track_id == 2), None)
        if not v_track and self.project.video_tracks:
            v_track = self.project.video_tracks[0]

        if not v_track:
            return

        duration = int(round(5.0 * self.project.fps))
        clip = TimelineClip.create_fx_layer(
            track_id=v_track.track_id,
            timeline_in=self.project.playhead_frame,
            duration_frames=duration,
            fx_type=fx_type,
        )
        v_track.add_clip(clip)
        self.canvas.selected_clip_id = clip.clip_id
        self.inspector.inspect_clip(clip)
        self._on_project_modified()
        fx_label = "ReShade FX Layer" if fx_type == "reshade" else "DLSS 5 Neural Layer"
        self.statusMessage.emit(f"Created {fx_label}", False)

    def add_dlss_adjustment_layer(self) -> None:
        """Compatibility alias."""
        self.add_fx_layer("dlss5")

    def add_video_track(self) -> None:
        """Add a new video track to the timeline."""
        track = self.project.add_video_track()
        self.canvas._update_min_size()
        self.canvas.update()
        self.statusMessage.emit(f"Added video track {track.name}", False)

    def delete_top_track(self) -> None:
        """Remove topmost video track (if more than 1 track exists)."""
        if len(self.project.video_tracks) <= 1:
            QMessageBox.information(self, "Cannot Delete", "At least one video track is required.")
            return
        top_track = max(self.project.video_tracks, key=lambda t: t.track_id)
        self.project.remove_video_track(top_track.track_id)
        self.canvas.selected_clip_id = None
        self.inspector.inspect_clip(None)
        self.canvas._update_min_size()
        self.canvas.update()
        self.seek_frame(self.project.playhead_frame)
        self.statusMessage.emit(f"Removed track {top_track.name}", False)

    def seek_frame(self, frame: int) -> None:
        """Update playhead and re-render composite frame in Program Monitor."""
        total_f = self.project.get_total_frames()
        frame = max(0, min(total_f, frame))
        self.project.playhead_frame = frame

        self.ruler.set_current_frame(frame)
        self.canvas.update()

        # Composite frame
        res = self.compositor.render_frame(
            frame_idx=frame,
            is_export=False,
            split_ratio=self._current_split_ratio,
            return_raw=True,
        )
        if isinstance(res, tuple):
            composite, raw = res
        else:
            composite, raw = res, res
        self.monitor.display_frame(composite, raw, frame, total_f)

    def mark_in(self) -> None:
        frame = self.project.playhead_frame
        self.project.work_area_in = frame
        self.ruler.set_in_point(frame)
        self.statusMessage.emit(f"Set In Point to frame {frame}", False)

    def mark_out(self) -> None:
        frame = self.project.playhead_frame
        self.project.work_area_out = frame
        self.ruler.set_out_point(frame)
        self.statusMessage.emit(f"Set Out Point to frame {frame}", False)

    def _on_ruler_in_changed(self, frame: int) -> None:
        self.project.work_area_in = frame

    def _on_ruler_out_changed(self, frame: int) -> None:
        self.project.work_area_out = frame

    def _on_split_ratio_changed(self, ratio: float) -> None:
        self._current_split_ratio = ratio if ratio > 0.0 else None
        self.seek_frame(self.project.playhead_frame)

    def _on_clip_parameters_changed(self, clip: TimelineClip) -> None:
        self.canvas.update()
        self.seek_frame(self.project.playhead_frame)

    def _on_project_modified(self) -> None:
        tot = self.project.get_total_frames()
        self.ruler.total_frames = tot
        self.canvas._update_min_size()
        self.seek_frame(self.project.playhead_frame)

    def _on_zoom_changed(self, val: int) -> None:
        ppf = val / 10.0
        self.ruler.set_zoom(ppf)
        self.canvas.set_zoom(ppf)

    def _on_split_tool_clicked(self) -> None:
        self.canvas.split_at_playhead()

    def _on_delete_tool_clicked(self) -> None:
        self.canvas.delete_selected_clip()

    def _on_export_clicked(self) -> None:
        """Open export modal dialog and spawn background render worker."""
        diag = TimelineExportDialog(self.project, self)
        if diag.exec() != QDialog.DialogCode.Accepted:
            return

        cfg = diag.get_export_config()
        if cfg["range_mode"] == "in_out" and (self.project.work_area_out > self.project.work_area_in):
            start_f = self.project.work_area_in
            end_f = self.project.work_area_out
        else:
            start_f = 0
            end_f = self.project.get_total_frames()

        # Progress dialog modal
        progress_diag = QDialog(self)
        progress_diag.setWindowTitle("Rendering Timeline Export...")
        progress_diag.resize(400, 150)
        p_layout = QVBoxLayout(progress_diag)
        lbl_info = QLabel("Exporting DLSS 5 Timeline...")
        p_layout.addWidget(lbl_info)

        p_bar = QProgressBar()
        p_bar.setRange(0, 100)
        p_layout.addWidget(p_bar)

        lbl_stats = QLabel("Preparing pipeline...")
        p_layout.addWidget(lbl_stats)

        btn_cancel = QPushButton("Cancel Export")
        btn_cancel.setStyleSheet("background: #991b1b; color: white;")
        p_layout.addWidget(btn_cancel)

        # Worker setup
        worker = TimelineExportWorker(
            compositor=self.compositor,
            start_frame=start_f,
            end_frame=end_f,
            output_path=cfg["output_path"],
            target_resolution=cfg["resolution"],
            target_fps=self.project.fps,
            codec=cfg["codec"],
            bitrate_mbps=cfg["bitrate"],
            parent=self,
        )
        self._export_worker = worker

        btn_cancel.clicked.connect(worker.cancel)

        def on_prog(pct: int, msg: str):
            p_bar.setValue(pct)
            lbl_info.setText(msg)

        def on_stats(fps: float, elapsed: float, eta: str):
            lbl_stats.setText(f"Speed: {fps:.1f} FPS | Elapsed: {int(elapsed)}s | ETA: {eta}")

        def on_done(path: str):
            progress_diag.accept()
            QMessageBox.information(self, "Export Complete", f"Timeline exported successfully to:\n{path}")
            self.statusMessage.emit(f"Export finished: {path}", False)

        def on_err(err: str):
            progress_diag.reject()
            QMessageBox.critical(self, "Export Failed", f"Timeline export failed:\n{err}")
            self.statusMessage.emit(f"Export error: {err}", True)

        worker.progressChanged.connect(on_prog)
        worker.statsUpdated.connect(on_stats)
        worker.finishedExport.connect(on_done)
        worker.failedExport.connect(on_err)

        worker.start()
        progress_diag.exec()

    def shutdown(self) -> None:
        """Gracefully release threads, monitor playback, and frame caches."""
        self.monitor.pause()
        if self._export_worker and self._export_worker.isRunning():
            self._export_worker.cancel()
            self._export_worker.wait(2000)
        self.frame_cache.clear()
