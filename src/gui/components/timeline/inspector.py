"""Clip Inspector and DLSS 5 Neural Rendering Effect Controls Widget."""

from __future__ import annotations

import json
import os
from PyQt6.QtCore import Qt, pyqtSignal
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
    QVBoxLayout,
    QWidget,
)

from ....timeline.models import DLSSConfig, TimelineClip
from ..sliders import LabeledSlider


class ClipInspectorWidget(QFrame):
    """Inspector panel displaying clip metadata and full Neural Rendering parameters matching Image 1."""

    parametersChanged = pyqtSignal(object)  # TimelineClip

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("clipInspectorWidget")
        self._current_clip: TimelineClip | None = None
        self._is_updating_ui = False

        self.setStyleSheet(
            "QFrame#clipInspectorWidget {"
            "  background-color: #121316;"
            "  border-left: 1px solid #23252a;"
            "}"
            "QGroupBox {"
            "  color: #e2e8f0;"
            "  font-size: 11px;"
            "  font-weight: 700;"
            "  border: 1px solid #262932;"
            "  border-radius: 4px;"
            "  margin-top: 10px;"
            "  padding-top: 10px;"
            "}"
            "QGroupBox::title {"
            "  subcontrol-origin: margin;"
            "  subcontrol-position: top left;"
            "  padding: 0 5px;"
            "  color: #94a3b8;"
            "  font-size: 11px;"
            "  font-weight: 600;"
            "}"
            "QLabel { color: #94a3b8; font-size: 11px; }"
            "QComboBox, QLineEdit {"
            "  background-color: #1a1c22;"
            "  color: #f8fafc;"
            "  border: 1px solid #2d313b;"
            "  border-radius: 4px;"
            "  padding: 4px 6px;"
            "  font-size: 11px;"
            "}"
            "QComboBox:focus, QLineEdit:focus {"
            "  border-color: #38bdf8;"
            "}"
            "QSlider::groove:horizontal { height: 4px; background: #262932; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #3b82f6; border-radius: 2px; }"
            "QSlider::handle:horizontal { width: 12px; height: 12px; margin: -4px 0; background: #93c5fd; border-radius: 6px; }"
            "QPushButton {"
            "  background: #1a1d24;"
            "  color: #e2e8f0;"
            "  border: 1px solid #2d313b;"
            "  border-radius: 4px;"
            "  padding: 5px 10px;"
            "  font-size: 11px;"
            "  font-weight: 600;"
            "}"
            "QPushButton:hover {"
            "  background: #262a35;"
            "  border-color: #38bdf8;"
            "}"
            "QPushButton:pressed {"
            "  background: #13151a;"
            "}"
            "QCheckBox { color: #cbd5e1; font-size: 11px; spacing: 6px; }"
            "QCheckBox::indicator { width: 14px; height: 14px; border-radius: 3px; border: 1px solid #334155; background: #1a1c22; }"
            "QCheckBox::indicator:checked { background: #2563eb; border-color: #38bdf8; }"
        )

        self._init_ui()

    def _init_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(6)

        # Header Title
        self.lbl_title = QLabel("Clip & Effect Inspector")
        self.lbl_title.setStyleSheet("font-weight: 700; font-size: 12px; color: #f8fafc;")
        self.main_layout.addWidget(self.lbl_title)

        # Empty State Message
        self.lbl_empty = QLabel("Select a clip or DLSS 5 Adjustment Layer on the timeline to edit its properties.")
        self.lbl_empty.setWordWrap(True)
        self.lbl_empty.setStyleSheet("color: #64748b; font-style: italic; padding: 20px 0;")
        self.main_layout.addWidget(self.lbl_empty)

        # Scroll Area for the extensive Inspector controls
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 4, 0)
        container_layout.setSpacing(10)

        # ---------------- 1. General Clip Properties ----------------
        grp_clip = QGroupBox("General Clip Properties")
        form_clip = QVBoxLayout(grp_clip)
        form_clip.setContentsMargins(8, 8, 8, 8)
        form_clip.setSpacing(6)

        row_name = QHBoxLayout()
        row_name.addWidget(QLabel("Name:"))
        self.txt_clip_name = QLineEdit()
        self.txt_clip_name.textChanged.connect(self._on_name_changed)
        row_name.addWidget(self.txt_clip_name, 1)
        form_clip.addLayout(row_name)

        row_info = QHBoxLayout()
        self.lbl_clip_type = QLabel("Media Clip")
        self.lbl_clip_duration = QLabel("0 frames")
        row_info.addWidget(self.lbl_clip_type)
        row_info.addStretch()
        row_info.addWidget(self.lbl_clip_duration)
        form_clip.addLayout(row_info)

        # Opacity slider
        row_op = QHBoxLayout()
        row_op.addWidget(QLabel("Opacity:"))
        self.slider_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(0, 100)
        self.slider_opacity.setValue(100)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        row_op.addWidget(self.slider_opacity, 1)
        self.lbl_opacity_val = QLabel("100%")
        self.lbl_opacity_val.setFixedWidth(36)
        row_op.addWidget(self.lbl_opacity_val)
        form_clip.addLayout(row_op)

        # Enable DLSS checkbox
        self.chk_dlss_enable = QCheckBox("Enable DLSS 5 Neural Processing")
        self.chk_dlss_enable.setStyleSheet("color: #38bdf8; font-weight: 600;")
        self.chk_dlss_enable.toggled.connect(self._on_dlss_toggled)
        form_clip.addWidget(self.chk_dlss_enable)

        container_layout.addWidget(grp_clip)

        # ---------------- 2. Presets Settings Card (Matching Image 1) ----------------
        self.grp_presets = QGroupBox("Presets Settings")
        presets_layout = QVBoxLayout(self.grp_presets)
        presets_layout.setContentsMargins(8, 8, 8, 8)
        presets_layout.setSpacing(6)

        # Top preset buttons: Default, Detail-Only, Reset All
        row_quick = QHBoxLayout()
        row_quick.setSpacing(6)
        self.btn_preset_default = QPushButton("Default")
        self.btn_preset_default.clicked.connect(self._preset_default)
        self.btn_preset_detail = QPushButton("Detail-Only")
        self.btn_preset_detail.clicked.connect(self._preset_detail_only)
        self.btn_reset_all = QPushButton("Reset All")
        self.btn_reset_all.clicked.connect(self._reset_all)
        row_quick.addWidget(self.btn_preset_default)
        row_quick.addWidget(self.btn_preset_detail)
        row_quick.addWidget(self.btn_reset_all)
        presets_layout.addLayout(row_quick)

        # Save Settings... and Load Settings...
        row_save_load = QHBoxLayout()
        row_save_load.setSpacing(6)
        self.btn_save_settings = QPushButton("Save Settings...")
        self.btn_save_settings.clicked.connect(self._on_save_settings)
        self.btn_load_settings = QPushButton("Load Settings...")
        self.btn_load_settings.clicked.connect(self._on_load_settings)
        row_save_load.addWidget(self.btn_save_settings)
        row_save_load.addWidget(self.btn_load_settings)
        presets_layout.addLayout(row_save_load)

        container_layout.addWidget(self.grp_presets)

        # ---------------- 3. DLSS 5 Neural Rendering Card (Matching Image 1) ----------------
        self.grp_dlss_core = QGroupBox("DLSS 5 Neural Rendering")
        dlss_layout = QVBoxLayout(self.grp_dlss_core)
        dlss_layout.setContentsMargins(8, 8, 8, 8)
        dlss_layout.setSpacing(6)

        # NR Style
        row_style = QHBoxLayout()
        lbl_style = QLabel("NR Style:")
        lbl_style.setStyleSheet("font-weight: 600; color: #f8fafc;")
        self.combo_style = QComboBox()
        self.combo_style.addItems(["Default", "Natural", "Cinematic"])
        self.combo_style.currentTextChanged.connect(self._on_style_changed)
        row_style.addWidget(lbl_style)
        row_style.addWidget(self.combo_style, 1)
        dlss_layout.addLayout(row_style)

        # Scale
        row_scale = QHBoxLayout()
        lbl_scale = QLabel("Scale:")
        lbl_scale.setStyleSheet("font-weight: 600; color: #f8fafc;")
        self.combo_scale = QComboBox()
        self.combo_scale.addItem("Source (100%)", 1.0)
        self.combo_scale.addItem("75%", 0.75)
        self.combo_scale.addItem("50%", 0.50)
        self.combo_scale.addItem("25%", 0.25)
        self.combo_scale.currentIndexChanged.connect(self._on_scale_changed)
        row_scale.addWidget(lbl_scale)
        row_scale.addWidget(self.combo_scale, 1)
        dlss_layout.addLayout(row_scale)

        # NR Intensity
        self.slider_intensity = LabeledSlider("NR Intensity", 0.0, 2.0, 1.0, step=0.05, decimals=2)
        self.slider_intensity.valueChanged.connect(self._on_slider_param_changed)
        dlss_layout.addWidget(self.slider_intensity)

        # NR Passes
        self.slider_passes = LabeledSlider("NR Passes", 1.0, 4.0, 1.0, step=1.0, decimals=0)
        self.slider_passes.valueChanged.connect(self._on_slider_param_changed)
        dlss_layout.addWidget(self.slider_passes)

        container_layout.addWidget(self.grp_dlss_core)

        # ---------------- 4. Tone Structure Card (Matching Image 1) ----------------
        self.grp_tone = QGroupBox("Tone  Structure")
        tone_layout = QVBoxLayout(self.grp_tone)
        tone_layout.setContentsMargins(8, 8, 8, 8)
        tone_layout.setSpacing(6)

        # Local Tone Strength
        self.slider_tone = LabeledSlider("Local Tone Strength", 0.0, 2.0, 1.0, step=0.05, decimals=2)
        self.slider_tone.valueChanged.connect(self._on_slider_param_changed)
        tone_layout.addWidget(self.slider_tone)

        # Local Structure Strength
        self.slider_structure = LabeledSlider("Local Structure Strength", 0.0, 2.0, 1.0, step=0.05, decimals=2)
        self.slider_structure.valueChanged.connect(self._on_slider_param_changed)
        tone_layout.addWidget(self.slider_structure)

        # Skin Structure Strength
        self.slider_skin = LabeledSlider("Skin Structure Strength", -1.0, 1.0, -1.0, step=0.05, decimals=2)
        self.slider_skin.valueChanged.connect(self._on_slider_param_changed)
        tone_layout.addWidget(self.slider_skin)

        container_layout.addWidget(self.grp_tone)

        # ---------------- 5. Neural Composition Card (Matching Image 1) ----------------
        self.grp_comp = QGroupBox("Neural Composition")
        comp_layout = QVBoxLayout(self.grp_comp)
        comp_layout.setContentsMargins(8, 8, 8, 8)
        comp_layout.setSpacing(6)

        # NR Color Strength
        self.slider_color_strength = LabeledSlider("NR Color Strength", 0.0, 1.0, 1.0, step=0.05, decimals=2)
        self.slider_color_strength.valueChanged.connect(self._on_slider_param_changed)
        comp_layout.addWidget(self.slider_color_strength)

        # Tone Preservation
        self.slider_tone_preservation = LabeledSlider("Tone Preservation", 0.0, 1.0, 0.0, step=0.05, decimals=2)
        self.slider_tone_preservation.valueChanged.connect(self._on_slider_param_changed)
        comp_layout.addWidget(self.slider_tone_preservation)

        # Face/Skin Protection
        self.slider_face_protection = LabeledSlider("Face/Skin Protection", 0.0, 1.0, 0.0, step=0.05, decimals=2)
        self.slider_face_protection.valueChanged.connect(self._on_slider_param_changed)
        comp_layout.addWidget(self.slider_face_protection)

        # Grain Preservation
        self.slider_grain = LabeledSlider("Grain Preservation", 0.0, 1.0, 0.0, step=0.05, decimals=2)
        self.slider_grain.valueChanged.connect(self._on_slider_param_changed)
        comp_layout.addWidget(self.slider_grain)

        # Mask Feather
        self.slider_feather = LabeledSlider("Mask Feather", 0.0, 128.0, 0.0, step=1.0, decimals=0, suffix=" px")
        self.slider_feather.valueChanged.connect(self._on_slider_param_changed)
        comp_layout.addWidget(self.slider_feather)

        # Automatic Mask Checkbox
        self.chk_auto_mask = QCheckBox("Automatic Mask")
        self.chk_auto_mask.toggled.connect(self._on_slider_param_changed)
        comp_layout.addWidget(self.chk_auto_mask)

        container_layout.addWidget(self.grp_comp)
        container_layout.addStretch()

        self.scroll_area.setWidget(self.container)
        self.main_layout.addWidget(self.scroll_area)
        self.scroll_area.setVisible(False)

    def inspect_clip(self, clip: TimelineClip | None) -> None:
        """Bind selected clip to inspector controls."""
        self._current_clip = clip
        if clip is None:
            self.lbl_empty.setVisible(True)
            self.scroll_area.setVisible(False)
            return

        self._is_updating_ui = True
        self.lbl_empty.setVisible(False)
        self.scroll_area.setVisible(True)

        self.txt_clip_name.setText(clip.name)
        is_adj = clip.clip_type == "adjustment_layer"
        self.lbl_clip_type.setText("DLSS 5 Adjustment Layer" if is_adj else "Media Video Clip")
        self.lbl_clip_type.setStyleSheet("color: #c084fc; font-weight: bold;" if is_adj else "color: #38bdf8; font-weight: bold;")
        self.lbl_clip_duration.setText(f"{clip.duration_frames} frames ({clip.timeline_in} -> {clip.timeline_out})")

        op_int = int(clip.opacity * 100.0)
        self.slider_opacity.setValue(op_int)
        self.lbl_opacity_val.setText(f"{op_int}%")

        # Configure DLSS settings
        cfg = clip.dlss_params
        self.chk_dlss_enable.setChecked(cfg.enabled)

        # NR Style
        idx = self.combo_style.findText(cfg.nr_style)
        if idx >= 0:
            self.combo_style.setCurrentIndex(idx)

        # Scale
        scale_val = cfg.scale
        matched_scale = False
        for i in range(self.combo_scale.count()):
            val = float(self.combo_scale.itemData(i) or 1.0)
            if abs(val - scale_val) < 0.01:
                self.combo_scale.setCurrentIndex(i)
                matched_scale = True
                break
        if not matched_scale:
            self.combo_scale.setCurrentIndex(0)

        # Sliders
        self.slider_intensity.setValue(cfg.nr_intensity)
        self.slider_passes.setValue(float(cfg.nr_passes))
        self.slider_tone.setValue(cfg.local_tone_strength)
        self.slider_structure.setValue(cfg.local_structure_strength)
        self.slider_skin.setValue(cfg.skin_structure_strength)
        self.slider_color_strength.setValue(cfg.nr_color_strength)
        self.slider_tone_preservation.setValue(cfg.tone_preservation)
        self.slider_face_protection.setValue(cfg.face_skin_protection)
        self.slider_grain.setValue(cfg.grain_preservation)
        self.slider_feather.setValue(float(cfg.mask_feather))
        self.chk_auto_mask.setChecked(cfg.automatic_mask)

        self._is_updating_ui = False

    def _on_name_changed(self, text: str) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.name = text
            self.parametersChanged.emit(self._current_clip)

    def _on_opacity_changed(self, val: int) -> None:
        self.lbl_opacity_val.setText(f"{val}%")
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.opacity = val / 100.0
            self.parametersChanged.emit(self._current_clip)

    def _on_dlss_toggled(self, checked: bool) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.dlss_params.enabled = checked
            self.parametersChanged.emit(self._current_clip)

    def _on_style_changed(self, text: str) -> None:
        if self._current_clip and not self._is_updating_ui:
            self._current_clip.dlss_params.nr_style = text
            self.parametersChanged.emit(self._current_clip)

    def _on_scale_changed(self, index: int) -> None:
        if self._current_clip and not self._is_updating_ui:
            scale_val = float(self.combo_scale.itemData(index) or 1.0)
            self._current_clip.dlss_params.scale = scale_val
            self.parametersChanged.emit(self._current_clip)

    def _on_slider_param_changed(self) -> None:
        if not self._current_clip or self._is_updating_ui:
            return

        cfg = self._current_clip.dlss_params
        cfg.nr_intensity = self.slider_intensity.value()
        cfg.nr_passes = int(round(self.slider_passes.value()))
        cfg.local_tone_strength = self.slider_tone.value()
        cfg.local_structure_strength = self.slider_structure.value()
        cfg.skin_structure_strength = self.slider_skin.value()
        cfg.nr_color_strength = self.slider_color_strength.value()
        cfg.tone_preservation = self.slider_tone_preservation.value()
        cfg.face_skin_protection = self.slider_face_protection.value()
        cfg.grain_preservation = self.slider_grain.value()
        cfg.mask_feather = int(round(self.slider_feather.value()))
        cfg.automatic_mask = self.chk_auto_mask.isChecked()

        # Update legacy parameters for maximum pipeline compatibility
        cfg.sharpness = max(0.0, min(100.0, cfg.local_structure_strength * 50.0))
        cfg.denoise = max(0.0, min(100.0, (1.0 - cfg.grain_preservation) * 50.0))

        self.parametersChanged.emit(self._current_clip)

    def _preset_default(self) -> None:
        self._is_updating_ui = True
        self.combo_style.setCurrentText("Default")
        self.combo_scale.setCurrentIndex(0)
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
        self._is_updating_ui = False
        self._on_slider_param_changed()

    def _preset_detail_only(self) -> None:
        self._is_updating_ui = True
        self.slider_color_strength.setValue(0.0)
        self.slider_tone_preservation.setValue(1.0)
        self._is_updating_ui = False
        self._on_slider_param_changed()

    def _reset_all(self) -> None:
        self._preset_default()

    def _on_save_settings(self) -> None:
        if not self._current_clip:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Neural Rendering Settings",
            os.path.expanduser("~/DLSS5_Preset.json"),
            "JSON Presets (*.json);;All Files (*.*)",
        )
        if not path:
            return
        try:
            data = self._current_clip.dlss_params.to_dict()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Preset Saved", f"Settings successfully saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save preset: {e}")

    def _on_load_settings(self) -> None:
        if not self._current_clip:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Neural Rendering Settings",
            os.path.expanduser("~/"),
            "JSON Presets (*.json);;All Files (*.*)",
        )
        if not path or not os.path.isfile(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._current_clip.dlss_params = DLSSConfig.from_dict(data)
            self.inspect_clip(self._current_clip)
            self.parametersChanged.emit(self._current_clip)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load preset: {e}")
