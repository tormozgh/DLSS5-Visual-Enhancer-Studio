"""Upscale Tab: NVIDIA RTX Video Super Resolution (VSR) and RTX Video HDR processing."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
from PIL import Image
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
from ..components.sliders import LabeledSlider
from ..workers import UpscaleWorker
from ...upscale.image.models import ImageUpscaleOptions
from ...upscale.video.models import UpscaleOptions

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tga"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}


class UpscaleTab(QWidget):
    """NVIDIA RTX Video Super Resolution (VSR) & RTX Video HDR processing interface."""

    statusMessage = pyqtSignal(str, bool)

    def __init__(self, settings: UISettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._current_file_path: Path | None = None
        self._is_video: bool = False
        self._source_w: int = 0
        self._source_h: int = 0
        self._source_fps: float = 0.0
        self._source_frames: int = 0
        self._worker: UpscaleWorker | None = None

        self._init_ui()

    def _init_ui(self) -> None:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # 1. Source Media Selection Card
        source_box = QGroupBox("Media Source")
        sb_layout = QVBoxLayout(source_box)

        picker_layout = QHBoxLayout()
        self.btn_select_file = QPushButton("Choose Media File (Image / Video)")
        self.btn_select_file.setProperty("class", "primary")
        self.btn_select_file.clicked.connect(self._on_choose_file)
        picker_layout.addWidget(self.btn_select_file)

        self.btn_clear_file = QPushButton("Clear")
        self.btn_clear_file.setProperty("class", "mini-btn")
        self.btn_clear_file.clicked.connect(self._on_clear_file)
        picker_layout.addWidget(self.btn_clear_file)
        sb_layout.addLayout(picker_layout)

        self.lbl_file_path = QLabel("No media loaded. Select an image or video to begin upscaling.")
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

        self.lbl_target_info = QLabel("Target Resolution: --")
        self.lbl_target_info.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        info_layout.addWidget(self.lbl_target_info)

        self.info_frame.setVisible(False)
        sb_layout.addWidget(self.info_frame)

        layout.addWidget(source_box)

        # 2. VSR Quality & Scale Settings
        vsr_box = QGroupBox("NVIDIA RTX Video Super Resolution (VSR)")
        tb_layout = QVBoxLayout(vsr_box)
        desc = QLabel(
            "Hardware-accelerated AI upscaling using NVIDIA RTX Tensor Cores.\n"
            "Enhances fine details, removes compression artifacts, and sharpens edges."
        )
        desc.setStyleSheet("color: #9ca3af; line-height: 1.4;")
        tb_layout.addWidget(desc)

        # VSR Quality
        q_layout = QHBoxLayout()
        q_label = QLabel("VSR Quality Level:")
        q_label.setStyleSheet("font-weight: 600;")
        self.combo_quality = QComboBox()
        self.combo_quality.addItem("Level 1 (Fast)", 1)
        self.combo_quality.addItem("Level 2", 2)
        self.combo_quality.addItem("Level 3", 3)
        self.combo_quality.addItem("Level 4 (Ultra Quality - Recommended)", 4)
        self.combo_quality.setCurrentIndex(3)
        self.combo_quality.currentIndexChanged.connect(self._update_target_info)
        q_layout.addWidget(q_label)
        q_layout.addWidget(self.combo_quality, 1)
        tb_layout.addLayout(q_layout)

        # Upscale Scale
        s_layout = QHBoxLayout()
        s_label = QLabel("Scale Factor:")
        s_label.setStyleSheet("font-weight: 600;")
        self.combo_scale = QComboBox()
        self.combo_scale.addItem("1.5x", 1.5)
        self.combo_scale.addItem("2.0x (Standard 4K/1440p Target)", 2.0)
        self.combo_scale.addItem("3.0x", 3.0)
        self.combo_scale.addItem("4.0x (Maximum)", 4.0)
        self.combo_scale.setCurrentIndex(1)
        self.combo_scale.currentIndexChanged.connect(self._update_target_info)
        s_layout.addWidget(s_label)
        s_layout.addWidget(self.combo_scale, 1)
        tb_layout.addLayout(s_layout)

        layout.addWidget(vsr_box)

        # 3. RTX Video HDR
        hdr_box = QGroupBox("NVIDIA RTX Video HDR")
        hdr_layout = QVBoxLayout(hdr_box)

        self.chk_enable_hdr = QCheckBox("Enable RTX Video HDR (Converts SDR to 10-bit HDR)")
        self.chk_enable_hdr.setChecked(False)
        hdr_layout.addWidget(self.chk_enable_hdr)

        self.slider_contrast = LabeledSlider("HDR Contrast", 0.0, 2.0, 1.0, step=0.05)
        self.slider_saturation = LabeledSlider("HDR Saturation", 0.0, 2.0, 1.0, step=0.05)
        self.slider_peak_nits = LabeledSlider("Peak Luminance", 400.0, 2000.0, 1000.0, step=50.0, decimals=0, suffix=" nits")

        hdr_layout.addWidget(self.slider_contrast)
        hdr_layout.addWidget(self.slider_saturation)
        hdr_layout.addWidget(self.slider_peak_nits)

        layout.addWidget(hdr_box)

        # 4. Video Codec / Container (visible for video source)
        self.video_format_box = QGroupBox("Video Encoding Options")
        vf_layout = QVBoxLayout(self.video_format_box)

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
        vf_layout.addLayout(codec_layout)

        cont_layout = QHBoxLayout()
        lbl_cont = QLabel("Container:")
        lbl_cont.setFixedWidth(80)
        self.combo_container = QComboBox()
        self.combo_container.addItems(["MP4", "MKV", "MOV"])
        cont_layout.addWidget(lbl_cont)
        cont_layout.addWidget(self.combo_container, 1)
        vf_layout.addLayout(cont_layout)

        self.video_format_box.setVisible(False)
        layout.addWidget(self.video_format_box)

        # 5. Execution & Export Actions Card
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

        # Image actions
        self.btn_render_image = QPushButton("Render & Save Upscaled Image")
        self.btn_render_image.setProperty("class", "primary")
        self.btn_render_image.clicked.connect(self._on_render_image)
        btn_layout.addWidget(self.btn_render_image)

        # Video actions
        self.btn_preview_frame = QPushButton("Preview 1-Frame")
        self.btn_preview_frame.clicked.connect(self._on_preview_video_frame)
        btn_layout.addWidget(self.btn_preview_frame)

        self.btn_render_video = QPushButton("Render Upscaled Video")
        self.btn_render_video.setProperty("class", "primary")
        self.btn_render_video.clicked.connect(self._on_render_video)
        btn_layout.addWidget(self.btn_render_video)

        # Cancel button
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

    def _on_choose_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image or Video for RTX Upscaling",
            "",
            "All Supported Media (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.mp4 *.mkv *.mov *.avi *.webm);;"
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff);;"
            "Videos (*.mp4 *.mkv *.mov *.avi *.webm);;"
            "All Files (*.*)",
        )
        if file_path:
            self.load_file(Path(file_path))

    def _on_clear_file(self) -> None:
        self._current_file_path = None
        self._source_w = 0
        self._source_h = 0
        self._source_fps = 0.0
        self._source_frames = 0
        self.lbl_file_path.setText("No media loaded. Select an image or video to begin upscaling.")
        self.info_frame.setVisible(False)
        self.video_format_box.setVisible(False)
        self._update_action_state()
        self.statusMessage.emit("Cleared upscale media input", False)

    def load_file(self, path: Path) -> None:
        if not path.is_file():
            return
        self._current_file_path = path
        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            self._is_video = False
            try:
                with Image.open(path) as img:
                    self._source_w, self._source_h = img.size
                self.lbl_file_path.setText(str(path))
                self.lbl_source_info.setText(f"Image: {path.name} ({self._source_w} x {self._source_h})")
                self.info_frame.setVisible(True)
                self.video_format_box.setVisible(False)
            except Exception as exc:
                self.statusMessage.emit(f"Failed to inspect image: {exc}", True)
                return

        elif suffix in VIDEO_EXTENSIONS:
            self._is_video = True
            try:
                cap = cv2.VideoCapture(str(path))
                self._source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self._source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                self._source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                self.lbl_file_path.setText(str(path))
                self.lbl_source_info.setText(
                    f"Video: {path.name} ({self._source_w} x {self._source_h} | {self._source_fps:.2f} FPS | {self._source_frames} frames)"
                )
                self.info_frame.setVisible(True)
                self.video_format_box.setVisible(True)
            except Exception as exc:
                self.statusMessage.emit(f"Failed to inspect video: {exc}", True)
                return
        else:
            self.statusMessage.emit(f"Unsupported media format: {suffix}", True)
            return

        self._update_target_info()
        self._update_action_state()
        self.statusMessage.emit(f"Loaded {path.name} for upscaling", False)

    def _update_target_info(self) -> None:
        if not self._current_file_path or self._source_w <= 0 or self._source_h <= 0:
            self.lbl_target_info.setText("Target Resolution: --")
            return

        scale = float(self.combo_scale.currentData() or 2.0)
        target_w = int(round(self._source_w * scale))
        target_h = int(round(self._source_h * scale))
        quality = self.combo_quality.currentText()

        res_tag = ""
        if target_w >= 3840 and target_h >= 2160:
            res_tag = " [4K UHD]"
        elif target_w >= 2560 and target_h >= 1440:
            res_tag = " [1440p QHD]"
        elif target_w >= 1920 and target_h >= 1080:
            res_tag = " [1080p FHD]"

        self.lbl_target_info.setText(
            f"Target: {target_w} x {target_h}{res_tag} | Factor: {scale:.1f}x | VSR: {quality}"
        )

    def _update_action_state(self) -> None:
        has_file = self._current_file_path is not None
        if not has_file:
            self.btn_render_image.setVisible(False)
            self.btn_preview_frame.setVisible(False)
            self.btn_render_video.setVisible(False)
        elif self._is_video:
            self.btn_render_image.setVisible(False)
            self.btn_preview_frame.setVisible(True)
            self.btn_render_video.setVisible(True)
        else:
            self.btn_render_image.setVisible(True)
            self.btn_preview_frame.setVisible(False)
            self.btn_render_video.setVisible(False)

    def _on_render_image(self) -> None:
        if not self._current_file_path or self._is_video:
            return

        scale = float(self.combo_scale.currentData() or 2.0)
        quality = int(self.combo_quality.currentData() or 4)

        options = ImageUpscaleOptions(
            vsr_quality=quality,
            size_mode="Scale factor",
            scale_factor=scale,
            output_format="PNG",
            quality=95,
            preserve_metadata=True,
            rename_mode="Auto",
            custom_suffix="_Upscale",
            ai_gpu_uuid=self._settings.ai_gpu_uuid,
        )

        self._start_worker(is_video=False, options=options, task_name="Image Upscale")

    def _on_preview_video_frame(self) -> None:
        if not self._current_file_path or not self._is_video:
            return
        self._run_video_upscale(preview_frames=1, task_name="1-Frame Video Preview")

    def _on_render_video(self) -> None:
        if not self._current_file_path or not self._is_video:
            return
        self._run_video_upscale(preview_frames=None, task_name="Full Video Upscale")

    def _run_video_upscale(self, preview_frames: int | None, task_name: str) -> None:
        scale = float(self.combo_scale.currentData() or 2.0)
        quality = int(self.combo_quality.currentData() or 4)
        contrast = int(round(self.slider_contrast.value() * 100))
        saturation = int(round(self.slider_saturation.value() * 100))
        peak_nits = int(round(self.slider_peak_nits.value()))

        options = UpscaleOptions(
            vsr_enabled=True,
            vsr_quality=quality,
            size_mode="Scale factor",
            scale_factor=scale,
            hdr_enabled=self.chk_enable_hdr.isChecked(),
            hdr_contrast=contrast,
            hdr_saturation=saturation,
            hdr_middle_gray=50,
            hdr_peak_luminance=peak_nits,
            hdr_precision="Packed 10-bit",
            codec=self.combo_codec.currentText(),
            container=self.combo_container.currentText(),
            quality="Auto (Default)",
            rename_mode="Auto",
            custom_suffix="_Upscale",
            ai_gpu_uuid=self._settings.ai_gpu_uuid,
            video_gpu_uuid=self._settings.video_gpu_uuid,
            preview_frames=preview_frames,
        )

        self._start_worker(is_video=True, options=options, task_name=task_name)

    def update_settings(self, settings: UISettings) -> None:
        self._settings = settings

    def _start_worker(self, is_video: bool, options: ImageUpscaleOptions | UpscaleOptions, task_name: str) -> None:
        if not self._current_file_path:
            return

        out_dir = self._settings.get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.progress_container.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_progress.setText(f"Initializing {task_name}...")
        self.btn_cancel.setVisible(True)

        self.btn_render_image.setEnabled(False)
        self.btn_preview_frame.setEnabled(False)
        self.btn_render_video.setEnabled(False)

        self._worker = UpscaleWorker(
            source_path=self._current_file_path,
            is_video=is_video,
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
        self.statusMessage.emit(f"Upscaling: {percent}% - {message}", False)

    def _on_finished(self, out_path: str, elapsed: float) -> None:
        self._reset_ui_state()
        p = Path(out_path)
        self.statusMessage.emit(f"Upscaling complete in {elapsed:.1f}s: {p.name}", False)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Upscale Complete")
        msg_box.setText(f"Media successfully upscaled in {elapsed:.1f}s:\n\n{out_path}")
        btn_open = msg_box.addButton("Open File", QMessageBox.ButtonRole.ActionRole)
        btn_folder = msg_box.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Close)
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == btn_open:
            try:
                os.startfile(out_path)
            except Exception:
                pass
        elif clicked == btn_folder:
            try:
                os.startfile(str(p.parent))
            except Exception:
                pass

    def _on_failed(self, error: str) -> None:
        self._reset_ui_state()
        self.statusMessage.emit(f"Upscale error: {error}", True)
        QMessageBox.critical(self, "Upscale Error", f"Failed to upscale media:\n\n{error}")

    def _on_cancelled(self) -> None:
        self._reset_ui_state()
        self.statusMessage.emit("Upscale job cancelled by user", True)

    def _on_cancel(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self.lbl_progress.setText("Cancelling upscale operation...")
            self._worker.cancel()

    def _reset_ui_state(self) -> None:
        self.progress_container.setVisible(False)
        self.btn_cancel.setVisible(False)
        self.btn_render_image.setEnabled(True)
        self.btn_preview_frame.setEnabled(True)
        self.btn_render_video.setEnabled(True)
