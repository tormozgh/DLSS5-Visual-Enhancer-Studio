"""Timeline Studio Tab: Premiere-style NLE multi-track editor, Media Pool, and DLSS 5 Adjustment Layer workflow."""

from __future__ import annotations

import os
from typing import Callable

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...settings.models import UISettings
from ...timeline.compositor import TimelineCompositor
from ...timeline.export_worker import TimelineExportWorker
from ...timeline.frame_cache import VideoFrameCache
from ...timeline.models import MediaAsset, TimelineClip, TimelineProject
from ..components.timeline.inspector import ClipInspectorWidget
from ..components.timeline.media_pool import MediaPoolWidget
from ..components.timeline.monitor import TimelineMonitorWidget
from ..components.timeline.time_ruler import TimeRulerWidget
from ..components.timeline.track_canvas import MultiTrackCanvas


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
        self.project = TimelineProject.create_default(fps=30.0, width=1920, height=1080)
        self.frame_cache = VideoFrameCache()
        self.compositor = TimelineCompositor(self.project, self.frame_cache)
        self._current_split_ratio: float | None = None
        self._export_worker: TimelineExportWorker | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Vertical Splitter: Top (Workspace panels) vs Bottom (Multi-track timeline)
        vert_splitter = QSplitter(Qt.Orientation.Vertical)
        vert_splitter.setHandleWidth(4)

        # ---------------- TOP WORKSPACE SECTION ----------------
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(4)

        # 1. Left: Media Pool
        self.media_pool = MediaPoolWidget(self.frame_cache, self)
        self.media_pool.assetDoubleClicked.connect(self._on_asset_double_clicked)
        self.media_pool.addAdjustmentLayerRequested.connect(self.add_dlss_adjustment_layer)
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

        # Timeline Header / Tools Bar
        tools_bar = QFrame()
        tools_bar.setStyleSheet("background-color: #14161b; border-top: 1px solid #23252a; border-bottom: 1px solid #23252a; padding: 4px 8px;")
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

        self.btn_add_adj = QPushButton("+ Add DLSS 5 Layer")
        self.btn_add_adj.setStyleSheet("background: #7c3aed; color: white; border-radius: 4px; padding: 4px 8px; font-weight: 600; font-size: 11px;")
        self.btn_add_adj.clicked.connect(self.add_dlss_adjustment_layer)
        tb_layout.addWidget(self.btn_add_adj)

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

        main_layout.addWidget(vert_splitter)

        # Render initial empty frame
        self.seek_frame(0)

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
        )
        v1.add_clip(clip)
        self.canvas.selected_clip_id = clip.clip_id
        self.inspector.inspect_clip(clip)
        self._on_project_modified()
        self.statusMessage.emit(f"Placed {asset.name} on track {v1.name}", False)

    def add_dlss_adjustment_layer(self) -> None:
        """Create a new DLSS 5 Adjustment Layer on track V2 (or topmost video track)."""
        # Place on V2 by default, or V3 if V2 is busy
        v_track = next((t for t in self.project.video_tracks if t.track_id == 2), None)
        if not v_track and self.project.video_tracks:
            v_track = self.project.video_tracks[0]

        if not v_track:
            return

        clip = TimelineClip.create_adjustment_layer(
            track_id=v_track.track_id,
            timeline_in=self.project.playhead_frame,
            duration_frames=150,  # 5 seconds at 30 fps
            name="DLSS 5 Adjustment Layer",
        )
        v_track.add_clip(clip)
        self.canvas.selected_clip_id = clip.clip_id
        self.inspector.inspect_clip(clip)
        self._on_project_modified()
        self.statusMessage.emit("Created DLSS 5 Neural Adjustment Layer", False)

    def seek_frame(self, frame: int) -> None:
        """Update playhead and re-render composite frame in Program Monitor."""
        total_f = self.project.get_total_frames()
        frame = max(0, min(total_f, frame))
        self.project.playhead_frame = frame

        self.ruler.set_current_frame(frame)
        self.canvas.update()

        # Composite frame
        composite = self.compositor.render_frame(
            frame_idx=frame,
            is_export=False,
            split_ratio=self._current_split_ratio,
        )
        self.monitor.display_frame(composite, frame, total_f)

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
        # Scale: 2 to 40 -> 0.2 to 4.0 px/frame
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
