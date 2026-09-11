"""Frame Interpolation Tab: NVIDIA DLSS Frame Generation (DLSSG) controls and rendering."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ...core.paths import OUTPUTS
from ...settings.models import UISettings
from ..workers import InterpolationWorker
from ...frame_interpolation.models import ENGINE_CHOICES, FPS_CHOICES, FrameInterpolationOptions
from ...frame_interpolation.preview import describe_frame_interpolation_plan, frame_interpolation_capability_text


class FrameInterpolationTab(QWidget):
    """NVIDIA DLSS Frame Generation (DLSSG) multi-frame video interpolation interface."""

    statusMessage = pyqtSignal(str, bool)

    def __init__(self, settings: UISettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._current_video_path: Path | None = None
        self._source_w: int = 0
        self._source_h: int = 0
        self._source_fps: float = 0.0
        self._source_frames: int = 0
        self._worker: InterpolationWorker | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 1. Video Source Selection Card
        source_box = QGroupBox("Video Source")
        sb_layout = QVBoxLayout(source_box)

        picker_layout = QHBoxLayout()
        self.btn_select_video = QPushButton("Choose Video File")
        self.btn_select_video.setProperty("class", "primary")
        self.btn_select_video.clicked.connect(self._on_choose_video)
        picker_layout.addWidget(self.btn_select_video)

        self.btn_clear_video = QPushButton("Clear")
        self.btn_clear_video.setProperty("class", "mini-btn")
        self.btn_clear_video.clicked.connect(self._on_clear_video)
        picker_layout.addWidget(self.btn_clear_video)
        sb_layout.addLayout(picker_layout)

        self.lbl_file_path = QLabel("No video loaded. Select a video file to begin frame interpolation.")
        self.lbl_file_path.setStyleSheet("color: #9ca3af; font-size: 11px;")
        self.lbl_file_path.setWordWrap(True)
        sb_layout.addWidget(self.lbl_file_path)

        self.info_frame = QFrame()
        self.info_frame.setProperty("class", "studio-card")
        info_layout = QVBoxLayout(self.info_frame)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setSpacing(4)

        self.lbl_source_info = QLabel("Source: None")
        self.lbl_source_info.setStyleSheet("color: #d0d4dc; font-weight: 600; font-size: 12px;")
        info_layout.addWidget(self.lbl_source_info)

        self.lbl_plan_info = QLabel("Interpolation Plan: --")
        self.lbl_plan_info.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        self.lbl_plan_info.setWordWrap(True)
        info_layout.addWidget(self.lbl_plan_info)

        self.info_frame.setVisible(False)
        sb_layout.addWidget(self.info_frame)

        layout.addWidget(source_box)

        # 2. Hardware Capability Box
        cap_box = QGroupBox("Hardware Acceleration & Engine Status")
        cap_layout = QVBoxLayout(cap_box)
        try:
            cap_text = frame_interpolation_capability_text()
        except Exception:
            cap_text = "NVIDIA Optical Flow Accelerator (OFA) / DLSSG Ready"
        lbl_cap = QLabel(cap_text)
        lbl_cap.setStyleSheet("color: #a0a4b0; font-size: 11px; line-height: 1.4;")
        lbl_cap.setWordWrap(True)
        cap_layout.addWidget(lbl_cap)
        layout.addWidget(cap_box)

        # 3. DLSS Frame Generation Settings Box
        box = QGroupBox("NVIDIA DLSS Frame Generation (DLSSG)")
        box_layout = QVBoxLayout(box)
        desc = QLabel(
            "Multi-frame neural interpolation powered by NVIDIA Optical Flow Accelerator (OFA).\n"
            "Boosts video framerates smoothly up to 240+ FPS with AI-generated intermediary frames."
        )
        desc.setStyleSheet("color: #9ca3af; line-height: 1.4;")
        box_layout.addWidget(desc)

        # Target FPS
        fps_layout = QHBoxLayout()
        fps_label = QLabel("Target Framerate:")
        fps_label.setStyleSheet("font-weight: 600;")
        fps_label.setFixedWidth(130)
        self.combo_fps = QComboBox()
        self.combo_fps.addItem("60 FPS (Standard Smooth)", "60")
        self.combo_fps.addItem("120 FPS (High Refresh Rate)", "120")
        self.combo_fps.addItem("144 FPS (Esports Display)", "144")
        self.combo_fps.addItem("240 FPS (Ultra Smooth)", "240")
        self.combo_fps.addItem("30 FPS", "30")
        self.combo_fps.addItem("50 FPS", "50")
        self.combo_fps.addItem("59.94 FPS", "59.94")
        self.combo_fps.addItem("90 FPS", "90")
        self.combo_fps.addItem("119.88 FPS", "119.88")
        self.combo_fps.addItem("165 FPS", "165")
        self.combo_fps.addItem("180 FPS", "180")
        self.combo_fps.addItem("360 FPS", "360")
        self.combo_fps.addItem("480 FPS", "480")
        self.combo_fps.setCurrentIndex(1)  # 120 FPS default
        self.combo_fps.currentIndexChanged.connect(self._update_plan_info)
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.combo_fps, 1)
        box_layout.addLayout(fps_layout)

        # Engine Mode
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Interpolation Engine:")
        mode_label.setStyleSheet("font-weight: 600;")
        mode_label.setFixedWidth(130)
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("Auto (Optimal Quality & Speed)", "Auto")
        self.combo_mode.addItem("Native DLSSG", "Native DLSSG")
        self.combo_mode.addItem("Cascade Multi-Pass", "Cascade")
        self.combo_mode.setCurrentIndex(0)
        self.combo_mode.currentIndexChanged.connect(self._update_plan_info)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.combo_mode, 1)
        box_layout.addLayout(mode_layout)

        layout.addWidget(box)

        # 4. Encoding Options Box
        enc_box = QGroupBox("Video Encoding Options")
        enc_layout = QVBoxLayout(enc_box)

        codec_layout = QHBoxLayout()
        lbl_codec = QLabel("Codec:")
        lbl_codec.setFixedWidth(80)
        self.combo_codec = QComboBox()
        self.combo_codec.addItems([
            "H.264 (NVIDIA NVENC)",
            "H.265 (NVIDIA NVENC)",
            "AV1 (NVIDIA NVENC)",
            "H.264",
            "H.265",
        ])
        codec_layout.addWidget(lbl_codec)
        codec_layout.addWidget(self.combo_codec, 1)
        enc_layout.addLayout(codec_layout)

        cont_layout = QHBoxLayout()
        lbl_cont = QLabel("Container:")
        lbl_cont.setFixedWidth(80)
        self.combo_container = QComboBox()
        self.combo_container.addItems(["MP4", "MKV", "MOV"])
        cont_layout.addWidget(lbl_cont)
        cont_layout.addWidget(self.combo_container, 1)
        enc_layout.addLayout(cont_layout)

        self.chk_hdr = QCheckBox("Preserve 10-bit HDR Pipeline")
        self.chk_hdr.setChecked(False)
        enc_layout.addWidget(self.chk_hdr)

        layout.addWidget(enc_box)

        # 5. Processing & Export Actions Card
        action_box = QGroupBox("Processing & Export")
        ab_layout = QVBoxLayout(action_box)

        # Progress elements
        self.progress_container = QWidget()
        pc_layout = QVBoxLayout(self.progress_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_progress = QLabel("Ready")
        self.lbl_progress.setStyleSheet("color: #d0d4dc; font-size: 11px;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        pc_layout.addWidget(self.lbl_progress)
        pc_layout.addWidget(self.progress_bar)
        self.progress_container.setVisible(False)
        ab_layout.addWidget(self.progress_container)

        # Action Buttons Layout
        btn_layout = QHBoxLayout()

        self.btn_preview_3s = QPushButton("Preview 3s Clip")
        self.btn_preview_3s.clicked.connect(self._on_preview_3s)
        btn_layout.addWidget(self.btn_preview_3s)

        self.btn_render_full = QPushButton("Render Full Interpolated Video")
        self.btn_render_full.setProperty("class", "primary")
        self.btn_render_full.clicked.connect(self._on_render_full)
        btn_layout.addWidget(self.btn_render_full)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setProperty("class", "danger")
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_cancel.setVisible(False)
        btn_layout.addWidget(self.btn_cancel)

        ab_layout.addLayout(btn_layout)
        layout.addWidget(action_box)

        layout.addStretch()

        scroll_area.setWidget(container)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

        self._update_action_state()

    def _on_choose_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video for DLSS Frame Generation",
            "",
            "Video Files (*.mp4 *.mkv *.mov *.avi *.webm);;All Files (*.*)",
        )
        if file_path:
            self.load_video(Path(file_path))

    def _on_clear_video(self) -> None:
        self._current_video_path = None
        self._source_w = 0
        self._source_h = 0
        self._source_fps = 0.0
        self._source_frames = 0
        self.lbl_file_path.setText("No video loaded. Select a video file to begin frame interpolation.")
        self.info_frame.setVisible(False)
        self._update_action_state()
        self.statusMessage.emit("Cleared video input", False)

    def load_video(self, path: Path) -> None:
        if not path.is_file():
            return
        try:
            cap = cv2.VideoCapture(str(path))
            self._source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self._source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            self._current_video_path = path
            duration_s = (self._source_frames / self._source_fps) if self._source_fps > 0 else 0.0

            self.lbl_file_path.setText(str(path))
            self.lbl_source_info.setText(
                f"{path.name} ({self._source_w} x {self._source_h} | {self._source_fps:.2f} FPS | "
                f"{self._source_frames} frames | {duration_s:.1f}s)"
            )
            self.info_frame.setVisible(True)
            self._update_plan_info()
            self._update_action_state()
            self.statusMessage.emit(f"Loaded {path.name} for frame generation", False)
        except Exception as exc:
            self.statusMessage.emit(f"Failed to inspect video: {exc}", True)

    def _update_plan_info(self) -> None:
        if not self._current_video_path:
            self.lbl_plan_info.setText("Interpolation Plan: --")
            return

        target_fps = str(self.combo_fps.currentData() or "120")
        engine = str(self.combo_mode.currentData() or "Auto")

        try:
            plan_str = describe_frame_interpolation_plan([str(self._current_video_path)], target_fps, engine)
            self.lbl_plan_info.setText(plan_str)
        except Exception:
            try:
                target_f = float(target_fps)
                multiplier = (target_f / self._source_fps) if self._source_fps > 0 else 2.0
                est_frames = int(round(self._source_frames * multiplier))
                self.lbl_plan_info.setText(
                    f"Plan: {self._source_fps:.2f} FPS -> {target_fps} FPS ({multiplier:.2f}x) | Estimated Output: ~{est_frames} frames"
                )
            except Exception:
                self.lbl_plan_info.setText(f"Plan: -> {target_fps} FPS via {engine}")

    def _update_action_state(self) -> None:
        has_file = self._current_video_path is not None
        self.btn_preview_3s.setVisible(has_file)
        self.btn_render_full.setVisible(has_file)

    def _on_preview_3s(self) -> None:
        self._start_render(preview_seconds=3.0, task_name="3s Video Preview")

    def _on_render_full(self) -> None:
        self._start_render(preview_seconds=None, task_name="Full Interpolated Video")

    def update_settings(self, settings: UISettings) -> None:
        self._settings = settings

    def _start_render(self, preview_seconds: float | None, task_name: str) -> None:
        if not self._current_video_path:
            return

        target_fps = str(self.combo_fps.currentData() or "120")
        engine = str(self.combo_mode.currentData() or "Auto")
        codec = self.combo_codec.currentText()
        container = self.combo_container.currentText()
        hdr_mode = self.chk_hdr.isChecked()

        options = FrameInterpolationOptions(
            ai_gpu_uuid=self._settings.ai_gpu_uuid,
            video_gpu_uuid=self._settings.video_gpu_uuid,
            target_fps=target_fps,
            engine=engine,
            codec=codec,
            container=container,
            quality="Auto (Default)",
            hdr_mode=hdr_mode,
            rename_mode="Auto",
            custom_suffix="_Frame_Interpolation",
            preview_seconds=preview_seconds,
        )

        out_dir = self._settings.get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.progress_container.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress.setText(f"Initializing {task_name}...")
        self.btn_cancel.setVisible(True)

        self.btn_preview_3s.setEnabled(False)
        self.btn_render_full.setEnabled(False)

        self._worker = InterpolationWorker(
            source_path=self._current_video_path,
            options=options,
            output_dir=out_dir,
            parent=self,
        )
        self._worker.progressChanged.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.start()

    def _on_progress(self, fraction: float, message: str) -> None:
        percent = int(fraction * 100)
        self.progress_bar.setValue(percent)
        self.lbl_progress.setText(f"{percent}%: {message}")
        self.statusMessage.emit(f"Interpolating: {percent}% - {message}", False)

    def _on_finished(self, out_path: str, elapsed: float) -> None:
        self._reset_ui_state()
        p = Path(out_path)
        self.statusMessage.emit(f"Interpolation complete in {elapsed:.1f}s: {p.name}", False)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Interpolation Complete")
        msg_box.setText(f"Video successfully interpolated in {elapsed:.1f}s:\n\n{out_path}")
        btn_open = msg_box.addButton("Open Video", QMessageBox.ButtonRole.ActionRole)
        btn_folder = msg_box.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Close)
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == btn_open:
            os.startfile(out_path)
        elif clicked == btn_folder:
            os.startfile(str(p.parent))

    def _on_failed(self, error: str) -> None:
        self._reset_ui_state()
        self.statusMessage.emit(f"Interpolation error: {error}", True)
        QMessageBox.critical(self, "Interpolation Error", f"Failed to interpolate video:\n\n{error}")

    def _on_cancelled(self) -> None:
        self._reset_ui_state()
        self.statusMessage.emit("Interpolation job cancelled by user", True)

    def _on_cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.lbl_progress.setText("Cancelling interpolation operation...")
            self._worker.cancel()

    def _reset_ui_state(self) -> None:
        self.progress_container.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_preview_3s.setEnabled(True)
        self.btn_render_full.setEnabled(True)
