"""Neural Rendering Tab: Dual Image & Video DLSS 5 workflows with Split and Side-by-Side views."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.ffmpeg import CODEC_CHOICES, ENCODING_QUALITIES
from ...core.paths import OUTPUTS
from ...neural_rendering.video.models import ConversionOptions
from ...settings.models import CONTAINER_CHOICES, UISettings
from ..components.sliders import LabeledSlider
from ..components.split_canvas import CanvasViewMode, SplitCanvas
from ..components.timeline import VideoTimelineBar
from ..preview_engine import PreviewEngine, PreviewParameters, VIDEO_EXTENSIONS
from ..workers import VideoRenderWorker


class NeuralRenderingTab(QWidget):
    """Interactive DLSS 5 workspace supporting Image and Video workflows."""

    statusMessage = pyqtSignal(str, bool)
    latencyUpdated = pyqtSignal(float)

    def __init__(
        self,
        settings: UISettings,
        preview_engine: PreviewEngine,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._settings = settings
        self._engine = preview_engine
        self._current_file_path: Path | None = None
        self._is_video: bool = False
        self._video_frames: int = 0
        self._video_fps: float = 30.0
        self._latest_preview_qimage: QImage | None = None
        self._video_worker: VideoRenderWorker | None = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -------------------------------------------------------------
        # Left Area: Viewport Canvas + Video Timeline Scrubber + Toolbar
        # -------------------------------------------------------------
        viewport_container = QWidget()
        viewport_layout = QVBoxLayout(viewport_container)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(6)

        # Interactive Split & Side-by-Side Canvas
        self.canvas = SplitCanvas()
        viewport_layout.addWidget(self.canvas, 1)

        # Professional Video Timeline Scrubber Bar
        self.timeline = VideoTimelineBar(self)
        self.timeline.setVisible(False)
        self.timeline.frameChanged.connect(self._on_timeline_seek)
        viewport_layout.addWidget(self.timeline)

        # Viewport Toolbar: View Modes & Zoom
        view_bar = QHBoxLayout()
        view_bar.setContentsMargins(4, 0, 4, 0)
        view_bar.setSpacing(6)

        # View Mode Toggle: Split Slider vs Side-by-Side
        self.btn_view_split = QPushButton("Split View")
        self.btn_view_split.setProperty("class", "mini-btn")
        self.btn_view_split.setCheckable(True)
        self.btn_view_split.setChecked(True)
        self.btn_view_split.clicked.connect(lambda: self._set_canvas_mode(CanvasViewMode.SPLIT))

        self.btn_view_sbs = QPushButton("Side-by-Side")
        self.btn_view_sbs.setProperty("class", "mini-btn")
        self.btn_view_sbs.setCheckable(True)
        self.btn_view_sbs.clicked.connect(lambda: self._set_canvas_mode(CanvasViewMode.SIDE_BY_SIDE))

        self.view_mode_group = QButtonGroup(self)
        self.view_mode_group.addButton(self.btn_view_split)
        self.view_mode_group.addButton(self.btn_view_sbs)

        # View quick actions
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setProperty("class", "mini-btn")
        self.btn_fit.clicked.connect(self.canvas.fit_to_view)

        self.btn_1to1 = QPushButton("1:1")
        self.btn_1to1.setProperty("class", "mini-btn")
        self.btn_1to1.clicked.connect(self.canvas.reset_1to1)

        self.btn_show_orig = QPushButton("Original")
        self.btn_show_orig.setProperty("class", "mini-btn")
        self.btn_show_orig.clicked.connect(lambda: self.canvas.set_split_ratio(1.0))

        self.btn_show_dlss = QPushButton("DLSS 5")
        self.btn_show_dlss.setProperty("class", "mini-btn")
        self.btn_show_dlss.clicked.connect(lambda: self.canvas.set_split_ratio(0.0))

        view_bar.addWidget(QLabel("View:"))
        view_bar.addWidget(self.btn_view_split)
        view_bar.addWidget(self.btn_view_sbs)
        view_bar.addSpacing(8)
        view_bar.addWidget(self.btn_fit)
        view_bar.addWidget(self.btn_1to1)
        view_bar.addWidget(self.btn_show_orig)
        view_bar.addWidget(self.btn_show_dlss)
        view_bar.addStretch()

        viewport_layout.addLayout(view_bar)

        # -------------------------------------------------------------
        # Right Area: Scrollable Parameter & Control Cards
        # -------------------------------------------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(380)
        scroll_area.setMaximumWidth(440)

        controls_container = QWidget()
        self.controls_layout = QVBoxLayout(controls_container)
        self.controls_layout.setContentsMargins(8, 8, 8, 8)
        self.controls_layout.setSpacing(10)

        # 1. Mode Switcher (Image vs Video)
        self._build_mode_card()

        # 2. Source Selection Card
        self._build_source_card()

        # 3. Presets Card
        self._build_presets_card()

        # 4. DLSS 5 Core Parameters
        self._build_dlss_core_card()

        # 5. Tone & Structure
        self._build_tone_structure_card()

        # 6. Neural Composition
        self._build_composition_card()

        # 7. Video-Specific Controls (Shimmer, Codec, Container)
        self._build_video_controls_card()

        # 8. Export Actions
        self._build_actions_card()

        self.controls_layout.addStretch()
        scroll_area.setWidget(controls_container)

        splitter.addWidget(viewport_container)
        splitter.addWidget(scroll_area)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        main_layout.addWidget(splitter)

    def _build_mode_card(self) -> None:
        mode_box = QHBoxLayout()
        mode_box.setSpacing(6)

        self.btn_mode_image = QPushButton("Image Mode")
        self.btn_mode_image.setCheckable(True)
        self.btn_mode_image.setChecked(True)
        self.btn_mode_image.clicked.connect(lambda: self._set_workflow_mode(False))

        self.btn_mode_video = QPushButton("Video Mode")
        self.btn_mode_video.setCheckable(True)
        self.btn_mode_video.clicked.connect(lambda: self._set_workflow_mode(True))

        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.btn_mode_image)
        self.mode_group.addButton(self.btn_mode_video)

        mode_box.addWidget(self.btn_mode_image)
        mode_box.addWidget(self.btn_mode_video)
        self.controls_layout.addLayout(mode_box)

    def _build_source_card(self) -> None:
        card = QGroupBox("Media Source")
        layout = QVBoxLayout(card)
        layout.setSpacing(6)

        btn_layout = QHBoxLayout()
        self.btn_open_file = QPushButton("Browse Media...")
        self.btn_open_file.clicked.connect(self._on_browse_file)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setProperty("class", "mini-btn")
        self.btn_clear.clicked.connect(self._on_clear_source)

        btn_layout.addWidget(self.btn_open_file, 1)
        btn_layout.addWidget(self.btn_clear)

        self.lbl_file_name = QLabel("No file loaded (Drag & Drop supported)")
        self.lbl_file_name.setStyleSheet("color: #9ca0ab; font-size: 11px;")
        self.lbl_file_name.setWordWrap(True)

        layout.addLayout(btn_layout)
        layout.addWidget(self.lbl_file_name)
        self.controls_layout.addWidget(card)

    def _build_presets_card(self) -> None:
        card = QGroupBox("Presets")
        layout = QHBoxLayout(card)
        layout.setSpacing(6)

        self.btn_preset_default = QPushButton("Default")
        self.btn_preset_default.clicked.connect(self._preset_default)

        self.btn_preset_detail = QPushButton("Detail-Only")
        self.btn_preset_detail.setToolTip("Sets Color Strength=0 and Tone Preservation=1 to keep original colors")
        self.btn_preset_detail.clicked.connect(self._preset_detail_only)

        self.btn_reset_all = QPushButton("Reset All")
        self.btn_reset_all.clicked.connect(self._reset_all)

        layout.addWidget(self.btn_preset_default)
        layout.addWidget(self.btn_preset_detail)
        layout.addWidget(self.btn_reset_all)
        self.controls_layout.addWidget(card)

    def _build_dlss_core_card(self) -> None:
        card = QGroupBox("DLSS 5 Neural Rendering")
        layout = QVBoxLayout(card)
        layout.setSpacing(6)

        # NR Style
        style_layout = QHBoxLayout()
        style_label = QLabel("NR Style:")
        style_label.setStyleSheet("font-weight: 600;")
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Default", "Natural", "Cinematic"])
        self.combo_style.currentTextChanged.connect(self._trigger_preview)
        style_layout.addWidget(style_label)
        style_layout.addWidget(self.combo_style, 1)
        layout.addLayout(style_layout)

        # Scale
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Scale:")
        scale_label.setStyleSheet("font-weight: 600;")
        self.combo_scale = QComboBox()
        self.combo_scale.addItem("Source (100%)", 1.0)
        self.combo_scale.addItem("75%", 0.75)
        self.combo_scale.addItem("50%", 0.50)
        self.combo_scale.addItem("25%", 0.25)
        self.combo_scale.currentIndexChanged.connect(self._trigger_preview)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.combo_scale, 1)
        layout.addLayout(scale_layout)

        # Intensity & Passes
        self.slider_intensity = LabeledSlider("NR Intensity", 0.0, 2.0, self._settings.nr_intensity, step=0.05)
        self.slider_intensity.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_intensity)

        self.slider_passes = LabeledSlider("NR Passes", 1.0, 4.0, float(self._settings.nr_passes), step=1.0, decimals=0)
        self.slider_passes.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_passes)

        self.controls_layout.addWidget(card)

    def _build_tone_structure_card(self) -> None:
        card = QGroupBox("Tone & Structure")
        layout = QVBoxLayout(card)
        layout.setSpacing(6)

        self.slider_tone = LabeledSlider("Local Tone Strength", 0.0, 2.0, self._settings.local_tone_strength, step=0.05)
        self.slider_tone.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_tone)

        self.slider_structure = LabeledSlider("Local Structure Strength", 0.0, 2.0, self._settings.local_structure_strength, step=0.05)
        self.slider_structure.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_structure)

        self.slider_skin = LabeledSlider("Skin Structure Strength", -1.0, 2.0, self._settings.skin_structure_strength, step=0.05)
        self.slider_skin.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_skin)

        self.controls_layout.addWidget(card)

    def _build_composition_card(self) -> None:
        card = QGroupBox("Neural Composition")
        layout = QVBoxLayout(card)
        layout.setSpacing(6)

        self.slider_color_strength = LabeledSlider("NR Color Strength", 0.0, 1.0, self._settings.nr_color_strength, step=0.05)
        self.slider_color_strength.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_color_strength)

        self.slider_tone_preservation = LabeledSlider("Tone Preservation", 0.0, 1.0, self._settings.tone_preservation, step=0.05)
        self.slider_tone_preservation.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_tone_preservation)

        self.slider_face_protection = LabeledSlider("Face/Skin Protection", 0.0, 1.0, self._settings.face_skin_protection, step=0.05)
        self.slider_face_protection.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_face_protection)

        self.slider_grain = LabeledSlider("Grain Preservation", 0.0, 1.0, self._settings.grain_preservation, step=0.05)
        self.slider_grain.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_grain)

        self.slider_feather = LabeledSlider("Mask Feather", 0.0, 128.0, float(self._settings.mask_feather), step=1.0, decimals=0, suffix=" px")
        self.slider_feather.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_feather)

        self.chk_auto_mask = QCheckBox("Automatic Mask")
        self.chk_auto_mask.setChecked(self._settings.automatic_mask)
        self.chk_auto_mask.toggled.connect(self._trigger_preview)
        layout.addWidget(self.chk_auto_mask)

        self.controls_layout.addWidget(card)

    def _build_video_controls_card(self) -> None:
        self.video_card = QGroupBox("Video Encoding & Stabilization")
        layout = QVBoxLayout(self.video_card)
        layout.setSpacing(6)

        # Shimmer Suppression
        self.slider_shimmer = LabeledSlider("Shimmer Suppression", 0.0, 1.0, self._settings.shimmer_suppression, step=0.05)
        self.slider_shimmer.setToolTip("Temporal residual stabilizer. Keep at 0.0 for clean motion without ghosting; increase only for static shots with high-frequency shimmering.")
        self.slider_shimmer.valueChanged.connect(self._trigger_preview)
        layout.addWidget(self.slider_shimmer)

        # Codec
        codec_layout = QHBoxLayout()
        codec_label = QLabel("Codec:")
        self.combo_codec = QComboBox()
        for c in CODEC_CHOICES:
            self.combo_codec.addItem(c)
        self.combo_codec.setCurrentText(self._settings.codec)
        codec_layout.addWidget(codec_label)
        codec_layout.addWidget(self.combo_codec, 1)
        layout.addLayout(codec_layout)

        # Container
        cont_layout = QHBoxLayout()
        cont_label = QLabel("Container:")
        self.combo_container = QComboBox()
        for c in CONTAINER_CHOICES:
            self.combo_container.addItem(c)
        self.combo_container.setCurrentText(self._settings.container)
        cont_layout.addWidget(cont_label)
        cont_layout.addWidget(self.combo_container, 1)
        layout.addLayout(cont_layout)

        # HDR Mode
        self.chk_hdr = QCheckBox("Preserve HDR (10-bit output)")
        self.chk_hdr.setChecked(self._settings.hdr_mode)
        layout.addWidget(self.chk_hdr)

        self.video_card.setVisible(False)
        self.controls_layout.addWidget(self.video_card)

    def _build_actions_card(self) -> None:
        card = QGroupBox("Render & Export Actions")
        layout = QVBoxLayout(card)
        layout.setSpacing(6)

        # Image Actions
        self.image_actions_box = QWidget()
        img_act_layout = QVBoxLayout(self.image_actions_box)
        img_act_layout.setContentsMargins(0, 0, 0, 0)
        img_act_layout.setSpacing(6)

        self.btn_save_image = QPushButton("Save Enhanced Image")
        self.btn_save_image.setProperty("class", "primary")
        self.btn_save_image.clicked.connect(self._on_save_image)
        img_act_layout.addWidget(self.btn_save_image)

        # Video Actions
        self.video_actions_box = QWidget()
        vid_act_layout = QVBoxLayout(self.video_actions_box)
        vid_act_layout.setContentsMargins(0, 0, 0, 0)
        vid_act_layout.setSpacing(6)

        self.btn_preview_frame = QPushButton("1-Frame Preview")
        self.btn_preview_frame.clicked.connect(self._on_preview_1_frame)

        self.btn_preview_3s = QPushButton("3s Video Preview")
        self.btn_preview_3s.clicked.connect(self._on_preview_3s_video)

        self.btn_render_video = QPushButton("Render Full Video")
        self.btn_render_video.setProperty("class", "primary")
        self.btn_render_video.clicked.connect(self._on_render_full_video)

        self.btn_cancel_video = QPushButton("Cancel Rendering")
        self.btn_cancel_video.setProperty("class", "danger")
        self.btn_cancel_video.setVisible(False)
        self.btn_cancel_video.clicked.connect(self._on_cancel_video)

        vid_act_layout.addWidget(self.btn_preview_frame)
        vid_act_layout.addWidget(self.btn_preview_3s)
        vid_act_layout.addWidget(self.btn_render_video)
        vid_act_layout.addWidget(self.btn_cancel_video)

        self.video_actions_box.setVisible(False)

        # Progress bar & status container
        self.progress_container = QWidget()
        self.progress_container.setVisible(False)
        prog_layout = QVBoxLayout(self.progress_container)
        prog_layout.setContentsMargins(0, 0, 0, 0)
        prog_layout.setSpacing(4)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.lbl_progress = QLabel("Rendering video...")
        self.lbl_progress.setStyleSheet("color: #9ca0ab; font-size: 11px;")

        prog_layout.addWidget(self.progress_bar)
        prog_layout.addWidget(self.lbl_progress)

        layout.addWidget(self.image_actions_box)
        layout.addWidget(self.video_actions_box)
        layout.addWidget(self.progress_container)

        self.controls_layout.addWidget(card)

    def _connect_signals(self) -> None:
        self.canvas.imageDropped.connect(self.load_file)
        self._engine.sourceLoaded.connect(self._on_source_loaded)
        self._engine.videoLoaded.connect(self._on_video_loaded)
        self._engine.previewReady.connect(self._on_preview_ready)
        self._engine.errorOccurred.connect(self._on_engine_error)

    def _set_canvas_mode(self, mode: CanvasViewMode) -> None:
        self.canvas.set_view_mode(mode)
        if mode == CanvasViewMode.SPLIT:
            self.btn_view_split.setChecked(True)
            self.btn_view_sbs.setChecked(False)
        elif mode == CanvasViewMode.SIDE_BY_SIDE:
            self.btn_view_sbs.setChecked(True)
            self.btn_view_split.setChecked(False)

    def _set_workflow_mode(self, is_video: bool) -> None:
        self._is_video = is_video
        self.btn_mode_image.setChecked(not is_video)
        self.btn_mode_video.setChecked(is_video)
        self.video_card.setVisible(is_video)
        self.timeline.setVisible(is_video and self._video_frames > 0)
        self.video_actions_box.setVisible(is_video)
        self.image_actions_box.setVisible(not is_video)

    def load_file(self, file_path_str: str) -> None:
        path = Path(file_path_str)
        if not path.is_file():
            return

        self._current_file_path = path
        is_video = path.suffix.lower() in VIDEO_EXTENSIONS
        self._set_workflow_mode(is_video)

        size_kb = path.stat().st_size // 1024
        self.lbl_file_name.setText(f"{path.name} ({size_kb} KB)")
        self.statusMessage.emit(f"Loading {path.name}...", False)
        self._engine.load_source(str(path))

    def _on_browse_file(self) -> None:
        filter_str = (
            "All Supported Media (*.png *.jpg *.jpeg *.webp *.avif *.tiff *.bmp *.mp4 *.mkv *.mov *.avi *.webm);;"
            "Images (*.png *.jpg *.jpeg *.webp *.avif *.tiff *.bmp);;"
            "Videos (*.mp4 *.mkv *.mov *.avi *.webm);;"
            "All Files (*.*)"
        )
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Media File", "", filter_str)
        if file_path:
            self.load_file(file_path)

    def _on_clear_source(self) -> None:
        self._current_file_path = None
        self.lbl_file_name.setText("No file loaded")
        self.canvas.set_images(None, None)
        self.timeline.pause()
        self.timeline.setVisible(False)
        self.statusMessage.emit("Cleared source", False)

    def _on_video_loaded(self, total_frames: int, fps: float, duration_sec: float) -> None:
        self._video_frames = total_frames
        self._video_fps = fps
        self.timeline.set_media_info(total_frames, fps)
        self.timeline.setVisible(True)

    def _on_timeline_seek(self, frame_idx: int) -> None:
        self._engine.seek_video(frame_idx)

    def _on_source_loaded(self, qimg: QImage, w: int, h: int) -> None:
        if self.canvas.has_images():
            self.canvas.set_before_image(qimg)
        else:
            self.canvas.set_images(qimg, None)
        self.statusMessage.emit(f"Source: {w}×{h} | Drag sliders for instant preview", False)

    def _on_preview_ready(self, qimg: QImage, latency_ms: float, details: str) -> None:
        self._latest_preview_qimage = qimg
        self.canvas.set_after_image(qimg)
        self.timeline.notify_frame_ready()
        self.latencyUpdated.emit(latency_ms)
        self.statusMessage.emit(f"{details} ({latency_ms:.1f}ms)", False)

    def _on_engine_error(self, message: str) -> None:
        self.statusMessage.emit(message, True)

    def _get_current_params(self) -> PreviewParameters:
        scale_factor = float(self.combo_scale.currentData() or 1.0)
        return PreviewParameters(
            nr_style=self.combo_style.currentText(),
            upscaling_factor=scale_factor,
            nr_intensity=self.slider_intensity.value(),
            nr_passes=int(round(self.slider_passes.value())),
            local_tone_strength=self.slider_tone.value(),
            local_structure_strength=self.slider_structure.value(),
            skin_structure_strength=self.slider_skin.value(),
            nr_color_strength=self.slider_color_strength.value(),
            tone_preservation=self.slider_tone_preservation.value(),
            face_skin_protection=self.slider_face_protection.value(),
            grain_preservation=self.slider_grain.value(),
            mask_feather=int(round(self.slider_feather.value())),
            automatic_mask=self.chk_auto_mask.isChecked(),
            ai_gpu_uuid=self._settings.ai_gpu_uuid,
            nr_gpu_mode=self._settings.nr_gpu_mode,
            shimmer_suppression=self.slider_shimmer.value(),
            codec=self.combo_codec.currentText(),
            container=self.combo_container.currentText(),
            hdr_mode=self.chk_hdr.isChecked(),
        )

    def _trigger_preview(self) -> None:
        params = self._get_current_params()
        self._engine.update_params(params)

    def _preset_default(self) -> None:
        self.slider_intensity.setValue(1.0)
        self.slider_passes.setValue(1.0)
        self.slider_tone.setValue(1.0)
        self.slider_structure.setValue(1.0)
        self.slider_skin.setValue(-1.0)
        self.slider_color_strength.setValue(1.0)
        self.slider_tone_preservation.setValue(0.0)
        self.slider_face_protection.setValue(0.0)
        self.slider_grain.setValue(0.0)
        self.slider_feather.setValue(0.0)
        self.chk_auto_mask.setChecked(False)
        self.combo_style.setCurrentText("Default")
        self.combo_scale.setCurrentIndex(0)
        self.slider_shimmer.setValue(0.0)
        self._trigger_preview()

    def _preset_detail_only(self) -> None:
        self.slider_color_strength.setValue(0.0)
        self.slider_tone_preservation.setValue(1.0)
        self._trigger_preview()

    def _reset_all(self) -> None:
        self._preset_default()

    def _on_save_image(self) -> None:
        if self._latest_preview_qimage is None:
            QMessageBox.information(self, "Save Image", "No enhanced preview available to save.")
            return

        default_name = "DLSS5_Enhanced.png"
        if self._current_file_path:
            stem = self._current_file_path.stem
            default_name = f"{stem}_Neural_Rendering.png"

        out_dir = self._settings.get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        default_path = str(out_dir / default_name)

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Enhanced Image",
            default_path,
            "PNG Image (*.png);;JPEG Image (*.jpg);;WebP Image (*.webp);;TIFF Image (*.tiff)",
        )

        if save_path:
            success = self._latest_preview_qimage.save(save_path)
            if success:
                self.statusMessage.emit(f"Saved: {Path(save_path).name}", False)
                QMessageBox.information(self, "Saved", f"Enhanced image saved successfully:\n{save_path}")
            else:
                self.statusMessage.emit("Failed to save image", True)

    def _on_preview_1_frame(self) -> None:
        self._start_video_render(is_1_frame=True, is_3s=False)

    def _on_preview_3s_video(self) -> None:
        self._start_video_render(is_1_frame=False, is_3s=True)

    def _on_render_full_video(self) -> None:
        if not self._current_file_path:
            QMessageBox.warning(self, "Render Video", "Please load a video first.")
            return

        reply = QMessageBox.question(
            self,
            "Render Video",
            f"Render complete video with DLSS 5 Neural Rendering?\n\nSource: {self._current_file_path.name}\n"
            f"Destination: {OUTPUTS}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._start_video_render(is_1_frame=False, is_3s=False)

    def update_settings(self, settings: UISettings) -> None:
        self._settings = settings

    def _start_video_render(self, is_1_frame: bool = False, is_3s: bool = False) -> None:
        if not self._current_file_path:
            QMessageBox.warning(self, "Video Render", "Please load a video first.")
            return

        if self._video_worker is not None and self._video_worker.isRunning():
            QMessageBox.information(self, "Video Render", "A video rendering task is already in progress.")
            return

        scale_factor = float(self.combo_scale.currentData() or 1.0)
        options = ConversionOptions(
            ai_gpu_uuid=self._settings.ai_gpu_uuid,
            video_gpu_uuid=self._settings.video_gpu_uuid,
            nr_style=self.combo_style.currentText(),
            nr_intensity=self.slider_intensity.value(),
            nr_passes=int(round(self.slider_passes.value())),
            local_tone_strength=self.slider_tone.value(),
            local_structure_strength=self.slider_structure.value(),
            skin_structure_strength=self.slider_skin.value(),
            nr_color_strength=self.slider_color_strength.value(),
            tone_preservation=self.slider_tone_preservation.value(),
            face_skin_protection=self.slider_face_protection.value(),
            grain_preservation=self.slider_grain.value(),
            shimmer_suppression=self.slider_shimmer.value(),
            mask_feather=int(round(self.slider_feather.value())),
            nr_mask=None,
            upscaling_factor=scale_factor,
            codec=self.combo_codec.currentText(),
            container=self.combo_container.currentText(),
            quality="Auto (Default)",
            preserve_hdr=self.chk_hdr.isChecked(),
            preview_frames=1 if is_1_frame else None,
            preview_seconds=3.0 if is_3s else None,
            automatic_mask=self.chk_auto_mask.isChecked(),
            nr_gpu_mode=self._settings.nr_gpu_mode,
        )

        out_dir = self._settings.get_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        self.progress_container.setVisible(True)
        self.progress_bar.setValue(0)
        task_name = "1-Frame Preview" if is_1_frame else ("3s Video Preview" if is_3s else "Full Video Render")
        self.lbl_progress.setText(f"Initializing {task_name}...")
        self.btn_cancel_video.setVisible(True)
        self.btn_preview_frame.setEnabled(False)
        self.btn_preview_3s.setEnabled(False)
        self.btn_render_video.setEnabled(False)

        self._video_worker = VideoRenderWorker(self._current_file_path, options, out_dir)
        self._video_worker.progressChanged.connect(self._on_video_progress)
        self._video_worker.finished.connect(self._on_video_finished)
        self._video_worker.failed.connect(self._on_video_failed)
        self._video_worker.cancelled.connect(self._on_video_cancelled)
        self._video_worker.start()

    def _on_video_progress(self, fraction: float, message: str) -> None:
        percent = int(fraction * 100)
        self.progress_bar.setValue(percent)
        self.lbl_progress.setText(f"{percent}%: {message}")
        self.statusMessage.emit(f"Rendering: {percent}% — {message}", False)

    def _on_video_finished(self, out_path: str, elapsed: float) -> None:
        self._reset_video_ui()
        p = Path(out_path)
        self.statusMessage.emit(f"Video saved in {elapsed:.1f}s: {p.name}", False)

        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Video Render Complete")
        msg_box.setText(f"Video successfully rendered via DLSS 5 in {elapsed:.1f}s:\n\n{out_path}")
        btn_open = msg_box.addButton("Open Video", QMessageBox.ButtonRole.ActionRole)
        btn_folder = msg_box.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QMessageBox.StandardButton.Close)
        msg_box.exec()

        clicked = msg_box.clickedButton()
        if clicked == btn_open:
            import os
            os.startfile(out_path)
        elif clicked == btn_folder:
            import os
            os.startfile(str(p.parent))

    def _on_video_failed(self, error: str) -> None:
        self._reset_video_ui()
        self.statusMessage.emit(f"Video rendering error: {error}", True)
        QMessageBox.critical(self, "Video Render Error", f"Failed to render video:\n\n{error}")

    def _on_video_cancelled(self) -> None:
        self._reset_video_ui()
        self.statusMessage.emit("Video rendering cancelled by user", True)

    def _on_cancel_video(self) -> None:
        if self._video_worker is not None and self._video_worker.isRunning():
            self.lbl_progress.setText("Cancelling video render...")
            self._video_worker.cancel()

    def _reset_video_ui(self) -> None:
        self.progress_container.setVisible(False)
        self.btn_cancel_video.setVisible(False)
        self.btn_preview_frame.setEnabled(True)
        self.btn_preview_3s.setEnabled(True)
        self.btn_render_video.setEnabled(True)

